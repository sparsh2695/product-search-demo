# product-search-demo

A small Flask demo of hybrid product search over the [Fake Store
API](https://fakestoreapi.com/products) catalog (20 products).

## Running it

```bash
source venv/bin/activate
python app.py   # http://127.0.0.1:5001
```

Dependencies are pinned in `requirements.txt` (flask, requests,
sentence-transformers, scikit-learn, numpy). No other files/deps needed —
keep it that way; this is a demo, not a production service.

## Architecture

Single file, `app.py`. On startup:
1. Fetches all products from the Fake Store API into `PRODUCTS` / `CORPUS`.
2. Loads `sentence-transformers/all-MiniLM-L6-v2` once as `MODEL` and
   embeds the product corpus into `PRODUCT_EMBEDDINGS`.
3. Fits a `TfidfVectorizer` over the same corpus into `PRODUCT_TFIDF`.

`search(query)` is a two-stage **retrieve-then-rerank** pipeline:

1. **Retrieval** — combines two signals via **Reciprocal Rank Fusion
   (RRF)**: semantic similarity (`MODEL` embeddings, cosine similarity)
   and lexical similarity (TF-IDF, cosine similarity). Each candidate is
   ranked by both signals; `1/(RRF_K + rank)` is summed across the two
   rankings. `RRF_K = 60` is a standard default from the RRF literature —
   trust it unless there's a concrete reason to retune. This stage
   produces a shortlist of `SHORTLIST_K` (15) candidates, not the final 5.
2. **Rerank** — `CROSS_ENCODER` (`cross-encoder/ms-marco-MiniLM-L-6-v2`,
   loaded once at startup, same `sentence-transformers` package as
   `MODEL` — no extra dependency) scores each `(query, product_text)`
   pair jointly, which is more accurate than the bi-encoder's
   independently-computed embeddings because it actually attends across
   query and product together. The shortlist is re-sorted by this score
   and the top `TOP_K` (5) become the final result.

**Why bother with a shortlist funnel for only 20 products:** at this
catalog size you could cross-encode the whole catalog directly and skip
RRF as a pre-filter. The two-stage structure is kept anyway to demonstrate
the pattern used at real scale, where cross-encoding every product per
query would be too slow — RRF stays as the cheap, recall-oriented first
pass. Measured effect on the seeded eval: retrieval-only Precision@5 was
0.48, with reranking it's 0.52 (see `eval_relevance.py`).

## Gender-intent category filtering

Some queries imply a gendered gift/audience (e.g. "gift for my dad"), in
which case the opposite gender's clothing category should be excluded from
results before scoring, so it doesn't pollute the top-5.

**Decision:** this used to be hardcoded keyword matching (`MALE_TERMS`/
`FEMALE_TERMS` sets + a regex tokenizer). It was replaced with an
**embedding-based intent classifier** that reuses the already-loaded
`MODEL` instead of adding a new dependency or model:

- `MALE_SEED_PHRASES` / `FEMALE_SEED_PHRASES`: ~13 example phrases per
  class, embedded once at startup into `MALE_SEED_EMBEDDINGS` /
  `FEMALE_SEED_EMBEDDINGS`.
- `_excluded_category(query_embedding)` scores the query against each
  class via **max cosine similarity over that class's seed embeddings**
  (not a mean centroid — with a small, semantically varied seed set, a
  centroid can land in a point that isn't close to any real phrasing; max
  similarity asks "is this close to *at least one* example," which is more
  robust here).
- Classification requires both `GENDER_INTENT_THRESHOLD` (0.30, absolute
  similarity floor) and `GENDER_INTENT_MARGIN` (0.05, lead over the other
  class) to trip — this avoids flagging ambiguous/unrelated queries.
- `search()` embeds the query once and reuses that embedding for both
  intent classification and semantic scoring (don't re-encode the query
  twice).

**To retune:** print `male_score`/`female_score` inside
`_excluded_category` for real queries and adjust the threshold/margin, or
add a closer seed phrase to the relevant list, rather than reverting to
keyword matching.

**To verify a change to this logic:** re-run the check below (existing
queries should be unaffected; new phrasings should now classify correctly
without being in any hardcoded list):

```python
from app import MODEL, _excluded_category
for q in ["something warm for winter", "gift for my dad",
          "something for my grandpa", "his birthday present",
          "something for my grandma", "dress for the girl in my life"]:
    emb = MODEL.encode([q], normalize_embeddings=True)
    print(q, "->", _excluded_category(emb))
```

## Evaluation workflow

Two separate mechanisms, because they answer different questions:

- **`tests/test_intent_classifier.py`** (pytest) — regression tests for
  `_excluded_category`. These have an objective right answer (does the
  query imply a gender, and does the classifier catch it), so they're
  real pass/fail tests. Install dev deps with
  `pip install -r requirements-dev.txt`, run with `python -m pytest tests/ -v`.
  Run this after any change to the seed phrases, threshold, or margin.

- **`eval_relevance.py`** — hand-labeled relevance eval for `search()`
  overall (RRF-ranked results, not just the gender filter). "Is product X
  relevant to query Y" isn't objective, so this isn't a pass/fail test —
  it's a small labeled dataset (`LABELED_QUERIES`: query -> relevant
  product ids) scored via Precision@5/Recall@5 against live search
  output. The seeded labels were picked by reading the product catalog,
  not by you — review/adjust `LABELED_QUERIES` before treating the
  scores as real ground truth. Run with `python eval_relevance.py`.
  Useful for catching ranking issues the classifier tests can't see (e.g.
  it already surfaced that "gift for my dad" ranks jewelry above two
  relevant men's shirts — a real gap in the RRF ranking, not the intent
  classifier).

Add new cases to both files as the catalog or use cases grow, rather than
one-off manual scripts.

## Repo state

- Git repo initialized and pushed to
  https://github.com/sparsh2695/product-search-demo (public).
- `venv/` is gitignored — not committed.
