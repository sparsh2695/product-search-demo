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

`search(query)` combines two signals via **Reciprocal Rank Fusion (RRF)**:
- Semantic similarity (`MODEL` embeddings, cosine similarity)
- Lexical similarity (TF-IDF, cosine similarity)

Each candidate gets ranked by both signals; `1/(RRF_K + rank)` is summed
across the two rankings and the top `TOP_K` (5) win. `RRF_K = 60` is a
standard default from the RRF literature — trust it unless there's a
concrete reason to retune.

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

## Repo state

- Git repo initialized and pushed to
  https://github.com/sparsh2695/product-search-demo (public).
- `venv/` is gitignored — not committed.
