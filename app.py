import json

import numpy as np
from flask import Flask, render_template, request
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Source of PRODUCTS_FILE's snapshot -- see load_products() for why this is
# read from disk instead of fetched live on every startup.
FAKE_STORE_URL = "https://fakestoreapi.com/products"
PRODUCTS_FILE = "products.json"
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
#
# Second documented instance of that limitation: "winter clothes" keeps a
# genuinely irrelevant summer T-shirt (score 2.697) alongside two real
# winter jackets (6.891, 6.197), because 6.891-2.697=4.194 is inside the
# 5.0 margin. Tightening the margin to exclude it would need <4.194, which
# also drops the "gear for outdoor hiking" backpack above (needs >=4.763)
# -- no single value satisfies both (4.194 < 4.763). Five alternative
# cutoff approaches were tried, using the shortlist's score shape/spread
# instead of a flat number -- see calibrate_rerank_cutoff.py for the
# reproducible numbers:
#   1. Margin scaled by shortlist std (z-score-style): backwards -- the
#      T-shirt's standardized gap (0.712) is *smaller* than the backpack's
#      (1.626), so any cutoff generous enough to keep the backpack keeps
#      the T-shirt too.
#   2. Kneedle knee detection: the curve's knee lands one step past the
#      T-shirt, not before it -- still includes it, and regresses 2 other
#      currently-perfect queries when checked against the full labeled set
#      (mean precision across LABELED_QUERIES drops from 0.74 to 0.56).
#   3. Tail noise-floor z-test (z=2,2.5,3): the T-shirt isn't remotely
#      close to the noise floor, so no z excludes it (mean precision 0.62).
#   4. Cross-signal corroboration (does semantic/lexical retrieval agree
#      with the cross-encoder?): the T-shirt is independently rank 2-3/20
#      on both the semantic and lexical signals, not just the cross-
#      encoder -- it's a genuine top match on every signal this pipeline
#      computes, so no corroboration threshold excludes it without also
#      excluding real jewelry results in the backpack case above (those
#      score similarly poorly, 12-14/20, on the lexical signal).
#   5. Raising SHORTLIST_STD_FLOOR to also catch a different bad query
#      ("all products") was considered and rejected in favor of a
#      dedicated browse-all-intent classifier (see BROWSE_ALL_THRESHOLD)
#      -- it would have traded 2 correctly-working queries for 2 different
#      fixes, not a net improvement.
# Conclusion: none of the signals this pipeline computes (cross-encoder
# score, semantic similarity, lexical similarity) encode "seasonally
# appropriate," only topical relevance to "clothes" in general -- this is
# a genuine model/embedding limitation, not an unexplored corner of
# cutoff-tuning. RERANK_MARGIN stays at 5.0.
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

# Minimum cosine similarity a query must have to the browse-all seed phrases
# to be treated as "show me the whole catalog" rather than a search for
# something specific. Unlike GENDER_INTENT_THRESHOLD, no margin-against-a-
# second-class is needed -- this is one yes/no direction, not a choice
# between two competing classes. Calibrated empirically: 8 true browse-all
# phrasings ("all products", "show me everything", "what do you have", ...)
# scored between 0.692 and 1.000 (min 0.692); every specific-product query
# tested, including tricky category-specific-but-browse-phrased ones like
# "show me your jewelry" (0.473) and "what electronics do you sell" (0.523),
# scored at most 0.523. 0.6 sits centered in that real 0.17-point gap. This
# works cleanly, unlike an earlier attempt at a broad "which of our 4
# catalog categories does this query belong to" classifier (see the
# no-match-detection comment on SHORTLIST_STD_FLOOR above) -- that failed
# because it tested a broad multi-category question where nearly any
# shopping-flavored query lands close to some category; this tests one
# narrow, specific direction ("does this mean literally everything"),
# closer in spirit to the gender classifier than the category one.
BROWSE_ALL_THRESHOLD = 0.6

BROWSE_ALL_SEED_PHRASES = [
    "show me everything", "all products", "browse the full catalog",
    "what do you have", "see all items", "show your entire catalog",
    "everything you sell", "browse all products", "show me your whole inventory",
    "what is in stock", "see the full range", "show all items", "list everything",
]

app = Flask(__name__)


def load_products():
    # Vendored snapshot of the Fake Store API catalog (PRODUCTS_FILE), not a
    # live fetch. Two independent reasons, not just one: (1) Fake Store API
    # 403s requests from at least Render's outbound IP range regardless of
    # User-Agent -- a browser-like header didn't fix it, so this looks like
    # IP-range blocking, not header filtering, and isn't something a retry
    # would get past. (2) Even setting that aside, every calibrated constant
    # in this app (SHORTLIST_STD_FLOOR, RERANK_MARGIN, BROWSE_ALL_THRESHOLD,
    # GENDER_INTENT_THRESHOLD, every labeled eval query, every product id/
    # price referenced in CLAUDE.md) already implicitly assumes this exact
    # 20-product catalog never changes -- a live fetch was always one upstream
    # data change away from silently invalidating all of it. Pinning to a
    # snapshot fixes the deployment issue and removes that fragility at the
    # same time. Regenerate with the snippet in CLAUDE.md if the catalog is
    # deliberately meant to be refreshed.
    with open(PRODUCTS_FILE) as f:
        data = json.load(f)
    return [
        {
            "id": p["id"],
            "title": p["title"],
            "description": p["description"],
            "price": p["price"],
            "category": p["category"],
            "image": p["image"],
        }
        for p in data
    ]


print("Loading products from vendored catalog snapshot...")
PRODUCTS = load_products()
CORPUS = [
    f"Category: {p['category']}. {p['title']}. {p['description']}" for p in PRODUCTS
]
print(f"Loaded {len(PRODUCTS)} products.")

# Known limitation: this bi-encoder judges broad topical/paraphrase
# similarity, not fine-grained product-type identity, so a query for an
# item this catalog doesn't carry (e.g. "winter scarf") can still score
# highest against a topically-related item (a winter jacket) -- see
# "Item-type discrimination" in CLAUDE.md for the full investigation
# (three fixes tried and rejected, including swapping this model).
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

BROWSE_ALL_SEED_EMBEDDINGS = MODEL.encode(BROWSE_ALL_SEED_PHRASES, normalize_embeddings=True)
print(f"Computed browse-all-intent seed embeddings: {len(BROWSE_ALL_SEED_PHRASES)} phrases")

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


def _is_browse_all(query_embedding):
    """True if the query means "show me the whole catalog" rather than a
    search for something specific -- see BROWSE_ALL_THRESHOLD for the
    calibration evidence behind the cutoff."""
    return bool(
        cosine_similarity(query_embedding, BROWSE_ALL_SEED_EMBEDDINGS).max()
        >= BROWSE_ALL_THRESHOLD
    )


def _price_in_range(product, min_price, max_price):
    if min_price is not None and product["price"] < min_price:
        return False
    if max_price is not None and product["price"] > max_price:
        return False
    return True


def _shortlist_rerank_scores(query, query_embedding=None, min_price=None, max_price=None):
    """Retrieve-then-rerank up through the cross-encoder scoring step,
    stopping short of the final confident-count cutoff. Shared by
    search() and calibrate_std_floor.py so both use the exact same
    pipeline -- the calibration script scores SHORTLIST_STD_FLOOR
    candidates against real retrieval behavior, not a reimplementation
    of it that could drift out of sync. Accepts an optional precomputed
    query_embedding so search() can reuse the one it already encoded for
    the browse-all/gender-intent checks instead of encoding twice."""
    if query_embedding is None:
        query_embedding = MODEL.encode([query], normalize_embeddings=True)

    exclude_category = _excluded_category(query_embedding)
    if exclude_category:
        keep = np.array([p["category"] != exclude_category for p in PRODUCTS])
    else:
        keep = np.ones(len(PRODUCTS), dtype=bool)

    if min_price is not None or max_price is not None:
        keep = keep & np.array([_price_in_range(p, min_price, max_price) for p in PRODUCTS])

    candidates = np.where(keep)[0]
    if len(candidates) == 0:
        # Price range can genuinely empty the pool (e.g. min_price above every
        # product's price) -- gender exclusion alone never can (electronics/
        # jewelry are never excluded by gender, so >=7 products always remain).
        # Without this guard, np.std([]) below would be nan, and a
        # `nan < SHORTLIST_STD_FLOOR` comparison is always False in numpy, so
        # search() would skip the no-match path and crash later in
        # _confident_count instead of returning [] cleanly.
        return np.array([], dtype=int), np.array([])

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


def search(query, min_price=None, max_price=None, top_k=TOP_K):
    query_embedding = MODEL.encode([query], normalize_embeddings=True)

    if _is_browse_all(query_embedding):
        # Deliberate scope simplification: bypasses gender-category
        # exclusion too. A query combining both intents (e.g. "show me
        # everything for my dad") is out of scope for this demo. Price IS
        # honored here (cheap to apply, and "show me everything under $50"
        # is a natural combination unlike the gender case above).
        products = PRODUCTS
        if min_price is not None or max_price is not None:
            products = [p for p in products if _price_in_range(p, min_price, max_price)]
        return list(products)

    shortlist_ids, rerank_scores = _shortlist_rerank_scores(
        query, query_embedding, min_price, max_price
    )
    if len(shortlist_ids) == 0:
        return []

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
    min_price = request.args.get("min_price", type=float)
    max_price = request.args.get("max_price", type=float)
    results = search(query, min_price=min_price, max_price=max_price) if query else []
    return render_template(
        "index.html",
        query=query,
        results=results,
        example_queries=EXAMPLE_QUERIES,
        min_price=min_price,
        max_price=max_price,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001, use_reloader=False)
