import numpy as np
import requests
from flask import Flask, render_template, request
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAKE_STORE_URL = "https://fakestoreapi.com/products"
RRF_K = 60
TOP_K = 5  # Hard cap on results returned; the actual count can be lower --
           # see RERANK_MARGIN -- when the reranker isn't confident beyond
           # the top match(es).

# Cross-encoder score margin (raw model-score units) below the shortlist's
# top reranked score, within which a candidate is still considered
# "confidently relevant" and kept. Relative to each query's own top score,
# not an absolute cutoff, because this cross-encoder's raw scores are not
# well-calibrated across query types: a clean query like "gift for dad"
# tops out around 0.5, while "everyday jewelry" tops out around -5.1 even
# though that top jewelry match is fully valid. Widen to keep more
# borderline results; narrow to be stricter. 5.0 was chosen empirically:
# smaller values (e.g. 2.0) over-pruned "gear for outdoor hiking" down to
# just the reranker's single (questionable) top pick, dropping an
# obviously-relevant backpack that scored 4.76 points lower -- a case
# where the cross-encoder's own ranking, not the margin logic, was
# shakier than usual. Note: for ambiguous queries where relevant and
# irrelevant items sit close together in score, no value here will be
# perfectly precise -- that's a real calibration limitation of this model
# on short e-commerce queries, not a bug to tune away.
RERANK_MARGIN = 5.0

# Minimum standard deviation across the shortlist's cross-encoder scores,
# below which search() returns no results at all instead of forcing a
# top-N pick. Unlike RERANK_MARGIN/GENDER_INTENT_THRESHOLD, this isn't a
# floor on score *magnitude* -- cross-encoder magnitude isn't comparable
# across queries (see RERANK_MARGIN above), and neither is bi-encoder
# cosine similarity: e.g. "halloween costume" scores a 0.313 top semantic
# match, on par with "gift for my dad"'s 0.324, even though this catalog
# has nothing costume-related. A per-category classifier (query vs.
# "electronics"/"jewelery"/"men's clothing"/"women's clothing" seed
# phrases, same max-cosine-over-seeds approach as the gender classifier)
# was tried and fails the same way: "halloween costume" scores 0.560
# against the jewelery seeds, *higher* than genuinely valid queries like
# "gift for my dad" (0.469) or "gear for outdoor hiking" (0.393) -- there
# is no cut point in the sorted score list that separates real matches
# from queries this catalog has nothing for. Neither threshold nor
# top-class-vs-runner-up margin (the gender classifier's GENDER_INTENT_
# MARGIN approach) fixes this: margin tracks how cleanly a query splits
# across the 4 known categories, not whether it belongs to any of them --
# e.g. "t-shirt" (valid, genuinely unisex) has a tiny 0.029 margin, while
# "kitchen appliance" (nothing in the catalog) has a comfortable 0.135
# margin toward "electronics" purely from shared retail vocabulary.
#
# What does separate the two cases is the *spread* of cross-encoder
# scores across the whole 15-item shortlist, independent of their
# absolute level: when a real match exists, the reranker pulls it away
# from the rest of the shortlist (high variance); when nothing matches,
# every candidate gets a similarly bad score (flat, low-variance noise),
# regardless of whether that noise floor sits at -5 or -11 for a given
# query.
#
# 0.5 is not hand-picked -- it's the F1-optimal cut point from a sweep
# over a 23-valid/16-junk labeled query set, see calibrate_std_floor.py
# (run it to reproduce or extend the set). The sweep's optimal region is
# actually the whole interval (0.486, 0.543] -- every threshold in that
# range gives identical classification results on the calibration set,
# so 0.5 is just a clean value inside it, not a precise optimum. At this
# floor: 22/23 valid queries are kept, including all 5 eval_relevance.py
# labels (lowest is "storage for my laptop" at 0.947), and 13/16 junk
# queries are correctly rejected (halloween costume: 0.196, kitchen
# appliance: 0.085, cooking pot: 0.070, etc). As with LABELED_QUERIES in
# eval_relevance.py, the valid/junk labels in calibrate_std_floor.py are
# a judgment call made while reading the catalog, not independently
# verified ground truth -- review before extending.
#
# Known, accepted misses that no value here fixes (see
# calibrate_std_floor.py output for the full list): "video game
# console" (std 2.18), "warm gloves" (2.15), and "sunglasses" (0.68)
# slip through as false positives; "birthday present for my sister"
# (0.05) is dropped as a false negative -- these sit outside the
# (0.486, 0.543] safe interval entirely, so they're not a
# threshold-tuning gap, they're queries where this signal itself
# doesn't discriminate correctly.
SHORTLIST_STD_FLOOR = 0.5

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


def _confident_count(sorted_scores, top_k):
    """Given cross-encoder scores sorted descending, count how many leading
    scores are within RERANK_MARGIN of the top score -- i.e., the reranker
    is nearly as confident about them as its single best match -- capped at
    top_k. Always returns at least 1, since the top score is trivially
    within its own margin of itself."""
    top_score = sorted_scores[0]
    within_margin = sorted_scores >= (top_score - RERANK_MARGIN)
    return min(int(np.count_nonzero(within_margin)), top_k)


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


def _shortlist_rerank_scores(query):
    """Retrieve-then-rerank up through the cross-encoder scoring step,
    stopping short of the final confident-count cutoff. Shared by
    search() and calibrate_std_floor.py so both use the exact same
    pipeline -- the calibration script scores SHORTLIST_STD_FLOOR
    candidates against real retrieval behavior, not a reimplementation
    of it that could drift out of sync."""
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
    rerank_scores = np.asarray(CROSS_ENCODER.predict(pairs))

    return shortlist_ids, rerank_scores


def search(query, top_k=TOP_K):
    shortlist_ids, rerank_scores = _shortlist_rerank_scores(query)

    if np.std(rerank_scores) < SHORTLIST_STD_FLOOR:
        return []

    order = np.argsort(-rerank_scores)
    sorted_rerank_scores = rerank_scores[order]
    n_confident = _confident_count(sorted_rerank_scores, top_k)
    reranked_ids = shortlist_ids[order][:n_confident]

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
