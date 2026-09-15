import numpy as np
import requests
from flask import Flask, render_template, request
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAKE_STORE_URL = "https://fakestoreapi.com/products"
RRF_K = 60
TOP_K = 5

# Size of the RRF shortlist handed to the cross-encoder for reranking.
# Wider than TOP_K so the reranker (which judges query+product jointly, and
# is more accurate than the bi-encoder/TF-IDF retrieval stage) has room to
# promote a relevant item that RRF ranked outside the top 5.
SHORTLIST_K = 15

# Minimum cosine similarity a query must have to a class's seed phrases to be
# considered a match at all. Tune by encoding real queries, printing
# male_score/female_score, and adjusting so borderline/unrelated queries fall
# below this line.
GENDER_INTENT_THRESHOLD = 0.30

# Required lead of the winning class's score over the other class's score,
# to avoid classifying ambiguous queries (where both classes score similarly)
# as gendered. Raise this to make the classifier more conservative.
GENDER_INTENT_MARGIN = 0.05

MALE_SEED_PHRASES = [
    "gift for my dad", "gift for my husband", "gift for my boyfriend",
    "gift for grandpa", "something for him", "his birthday present",
    "present for a man", "for the guy in my life", "men's style",
    "men's fashion", "shirt for my husband", "clothes for my son",
    "outfit for my brother",
]

FEMALE_SEED_PHRASES = [
    "gift for my mom", "gift for my wife", "gift for my girlfriend",
    "gift for grandma", "something for her", "her birthday present",
    "present for a woman", "for the girl in my life", "women's style",
    "women's fashion", "dress for my wife", "clothes for my daughter",
    "outfit for my sister",
]

app = Flask(__name__)


def fetch_products():
    response = requests.get(FAKE_STORE_URL, timeout=10)
    response.raise_for_status()
    return [
        {
            "id": p["id"],
            "title": p["title"],
            "description": p["description"],
            "price": p["price"],
            "category": p["category"],
            "image": p["image"],
        }
        for p in response.json()
    ]


print("Fetching products from Fake Store API...")
PRODUCTS = fetch_products()
CORPUS = [
    f"Category: {p['category']}. {p['title']}. {p['description']}" for p in PRODUCTS
]
print(f"Loaded {len(PRODUCTS)} products.")

print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
PRODUCT_EMBEDDINGS = MODEL.encode(CORPUS, normalize_embeddings=True)
print(f"Computed semantic embeddings: {PRODUCT_EMBEDDINGS.shape}")

MALE_SEED_EMBEDDINGS = MODEL.encode(MALE_SEED_PHRASES, normalize_embeddings=True)
FEMALE_SEED_EMBEDDINGS = MODEL.encode(FEMALE_SEED_PHRASES, normalize_embeddings=True)
print(
    f"Computed gender-intent seed embeddings: "
    f"{len(MALE_SEED_PHRASES)} male, {len(FEMALE_SEED_PHRASES)} female"
)

print("Fitting TF-IDF vectorizer...")
VECTORIZER = TfidfVectorizer(stop_words="english")
PRODUCT_TFIDF = VECTORIZER.fit_transform(CORPUS)
print(f"Computed TF-IDF matrix: {PRODUCT_TFIDF.shape}")

print("Loading cross-encoder reranker (ms-marco-MiniLM-L-6-v2)...")
CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _ranks_from_scores(scores):
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def _excluded_category(query_embedding):
    """If the query embedding is strongly and unambiguously closer to one
    gender's seed phrases than the other, return the clothing category for
    the opposite gender so it can be excluded from candidates."""
    male_score = cosine_similarity(query_embedding, MALE_SEED_EMBEDDINGS).max()
    female_score = cosine_similarity(query_embedding, FEMALE_SEED_EMBEDDINGS).max()

    is_male = (
        male_score >= GENDER_INTENT_THRESHOLD
        and (male_score - female_score) >= GENDER_INTENT_MARGIN
    )
    is_female = (
        female_score >= GENDER_INTENT_THRESHOLD
        and (female_score - male_score) >= GENDER_INTENT_MARGIN
    )

    if is_male:
        return "women's clothing"
    if is_female:
        return "men's clothing"
    return None


def search(query, top_k=TOP_K):
    query_embedding = MODEL.encode([query], normalize_embeddings=True)

    exclude_category = _excluded_category(query_embedding)
    if exclude_category:
        keep = np.array([p["category"] != exclude_category for p in PRODUCTS])
    else:
        keep = np.ones(len(PRODUCTS), dtype=bool)
    candidates = np.where(keep)[0]

    semantic_scores = cosine_similarity(query_embedding, PRODUCT_EMBEDDINGS[candidates])[0]

    query_tfidf = VECTORIZER.transform([query])
    lexical_scores = cosine_similarity(query_tfidf, PRODUCT_TFIDF[candidates])[0]

    semantic_ranks = _ranks_from_scores(semantic_scores)
    lexical_ranks = _ranks_from_scores(lexical_scores)

    rrf_scores = 1.0 / (RRF_K + semantic_ranks) + 1.0 / (RRF_K + lexical_ranks)

    shortlist_local = np.argsort(-rrf_scores)[:SHORTLIST_K]
    shortlist_ids = candidates[shortlist_local]

    pairs = [(query, CORPUS[i]) for i in shortlist_ids]
    rerank_scores = CROSS_ENCODER.predict(pairs)
    reranked_ids = shortlist_ids[np.argsort(-rerank_scores)][:top_k]

    return [PRODUCTS[i] for i in reranked_ids]


EXAMPLE_QUERIES = [
    "something warm for winter",
    "gift for my dad",
    "gear for outdoor hiking",
    "everyday jewelry",
]


@app.route("/")
def index():
    query = request.args.get("q", "").strip()
    results = search(query) if query else []
    return render_template(
        "index.html", query=query, results=results, example_queries=EXAMPLE_QUERIES
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001, use_reloader=False)
