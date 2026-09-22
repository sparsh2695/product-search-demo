# product-search-demo

A small Flask demo of hybrid product search over the [Fake Store
API](https://fakestoreapi.com/products) catalog (20 products).

## Running it

```bash
source venv/bin/activate
python app.py   # http://127.0.0.1:5001
```

Dependencies are pinned in `requirements.txt` (flask, requests,
sentence-transformers, scikit-learn, numpy, gunicorn). Keep additions to this
list to genuine necessities — this is a demo, not a production service.
`gunicorn` is the one exception to "no production concerns": it's only used
by `Dockerfile` for public deployment (see "Deployment" below); local dev
still runs Flask's own server via `python app.py` and is unaffected.

## Deployment

Deployed to [Render](https://render.com) as a Web Service, built from
`Dockerfile` (Render auto-detects it — no separate build/start command
configured in Render's UI). Considered Hugging Face Spaces first, but its
Docker and Gradio SDKs both now require a paid PRO plan; only Spaces' Static
SDK (no server-side execution at all, so no Flask) is free. Render's free
Web Service tier runs a real container at no cost (spins down after 15 min
idle; cold-starts on the next request). Two things this required that plain
local dev doesn't:

- **`gunicorn`, not `python app.py`, in production.** Flask's own docs say
  its built-in dev server isn't meant to be reachable by untrusted traffic.
  `Dockerfile`'s `CMD` runs gunicorn directly against the `app:app` WSGI
  object, which means the `if __name__ == "__main__": app.run(...)` block at
  the bottom of `app.py` never executes in the container at all (gunicorn
  imports the module; `__name__` is `"app"`, not `"__main__"`) — so it didn't
  need to change for this, and local dev behavior (`python app.py`,
  port 5001) is untouched.
- **Single gunicorn worker.** Each worker is a separate process with its own
  copy of the loaded models (bi-encoder + cross-encoder, ~200MB+ combined) —
  multiple workers would multiply memory on a free-tier instance with limited
  RAM. Fine for a demo; would need revisiting under real concurrent load.

`Dockerfile`'s `CMD` binds to `$PORT`, which Render assigns dynamically at
container start (not a fixed port like some other platforms use) — written
in shell form (`CMD sh -c '...'`), not exec-array form, since exec form
doesn't expand environment variables; it would pass the literal string
`${PORT:-7860}` to gunicorn instead of substituting the actual port. Non-root
container user (`useradd -m -u 1000 user`) is standard Docker practice, and
incidentally also gives `sentence-transformers` a real `$HOME` to cache
downloaded models under (`~/.cache/huggingface`).

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
pass.

**Dynamic result count.** `search()` used to always return exactly
`TOP_K` results, padded with whatever scored highest even if irrelevant
(e.g. "gift for dad" returned 5 results but only 1 was actually relevant).
`_confident_count()` now caps results to however many reranked candidates
fall within `RERANK_MARGIN` (5.0) of the shortlist's *top* cross-encoder
score — relative to each query's own top score, not a global threshold,
because this cross-encoder's raw scores aren't calibrated across query
types (see the `RERANK_MARGIN` comment in `app.py` for the full
reasoning and the "gear for outdoor hiking" case that drove the margin
value). This always keeps at least the top result. Known, accepted
limitation: for queries where a genuinely irrelevant item scores between
two genuinely relevant ones, no margin value fixes that — it's a real
calibration limit of this small MS-MARCO-trained model on short
e-commerce queries, not something to tune away. Two documented instances,
with the same symptom but different root causes (see the `RERANK_MARGIN`
comment in `app.py` for full numbers on both):
- "everyday jewelry" keeps an irrelevant backpack interleaved among real
  jewelry results — a **retrieval-signal coincidence**: the backpack is a
  spurious top-1/20 lexical (TF-IDF) match despite scoring poorly (10/20)
  on semantic similarity, while the real jewelry items are the reverse
  (strong semantic rank, weak lexical rank).
- "winter clothes" keeps an irrelevant summer T-shirt alongside two real
  winter jackets — a **genuine conceptual gap**, not a signal coincidence:
  the T-shirt is independently a top-3 match on *every* signal this
  pipeline computes (semantic, lexical, and cross-encoder), because none
  of them encode "seasonally appropriate," only topical relevance to
  "clothes." Five alternative cutoff approaches (std-scaled margin, knee
  detection, noise-floor z-test, cross-signal corroboration, and a wider
  no-match floor) were tried and evidenced to fail — see
  `calibrate_rerank_cutoff.py` for the reproducible numbers, including
  that the best alternative (knee detection) *drops* mean precision on
  `eval_relevance.py`'s labeled set from 0.74 to 0.56 by regressing 2
  other currently-perfect queries while fixing this one.

Measured effect of the dynamic cutoff itself on the seeded eval (mean
precision within each query's own returned window, across 5 queries):
0.48 retrieval-only → 0.52 fixed-top-5 reranking → 0.74 with the dynamic
cutoff (see `eval_relevance.py`). `recall_at_k` in that script is
normalized by `min(returned window, |relevant|)`, not the raw
relevant-item count, so a deliberately short but fully-correct answer
(e.g. "gift for dad" → 1 result, 1 hit) scores Recall = 1.0 instead of
being penalized for not surfacing every relevant item in the whole
catalog.

**No-match detection.** Neither the dynamic cutoff above nor any absolute
score fixes the case where the catalog genuinely has nothing relevant
(e.g. "halloween costume" — this 20-product catalog has no costumes).
Three score-magnitude approaches were tried and all failed the same way,
because none of this pipeline's scores carry an absolute "is this
actually relevant" meaning across different queries — only relative
ranking within one query's candidates:
- An absolute floor on the cross-encoder's top shortlist score: fails
  because "storage for my laptop" (valid, top score -8.605) and
  "sunglasses" (nothing in the catalog, -8.717) are indistinguishable.
- An absolute floor on bi-encoder cosine similarity to the product
  catalog: fails for the same reason — "halloween costume" scores 0.313
  semantic similarity to its best product match, on par with "gift for
  my dad"'s 0.324, even though only one of those queries has a real
  match. Sentence embeddings are anisotropic (nearly all sentences,
  related or not, cluster into a narrow cosine-similarity band), so raw
  magnitude isn't informative here.
- A category-membership classifier, structured exactly like the
  gender-intent classifier below (seed phrases per catalog category —
  electronics / jewelery / men's / women's clothing — max cosine
  similarity, threshold, and margin over the runner-up category): also
  fails, and not from a bad threshold/margin value. It's a structural
  mismatch — gender classification tests one real symmetric axis where
  an unrelated query scores low on *both* poles, but "is this query in
  any of our 4 unrelated retail categories" doesn't have an analogous
  "belongs to none of them" region nearby in a general-purpose
  embedding's space; almost any shopping-flavored query (including ones
  for products this store doesn't carry) lands moderately close to at
  least one category. Concretely, "halloween costume" scores 0.560
  against the jewelery seeds — *higher* than valid queries like "gift
  for my dad" (0.469) — and margin doesn't help either: "t-shirt"
  (valid, genuinely unisex) has a tiny 0.029 margin between its top two
  categories, while "kitchen appliance" (nothing in the catalog) has a
  comfortable 0.135 margin toward "electronics" from shared retail
  vocabulary alone.

What does work: `SHORTLIST_STD_FLOOR` (see the comment in `app.py`) —
the standard deviation of the cross-encoder scores across the whole
15-item shortlist, independent of their absolute level. When a real
match exists, the reranker pulls it away from the rest of the
shortlist (high variance); when nothing matches, every candidate gets
a similarly bad score (flat, low-variance noise), whether that noise
floor sits at -5 or -11 for a given query. `search()` returns `[]`
when this spread falls below the floor. This isn't perfectly clean —
see the `SHORTLIST_STD_FLOOR` comment for the known false
positives/negatives — but it's meaningfully better than full
interleaving: at the calibrated floor of 0.5 it keeps 22/23 valid test
queries (including all 5 `eval_relevance.py` labels) while rejecting
13/16 junk queries. The value is F1-optimal on a labeled calibration
set, not hand-picked — see `calibrate_std_floor.py` below.

## Item-type discrimination

A third, distinct search-quality gap, different in kind from the two
`RERANK_MARGIN` cases above: `"winter scarf"` returns the BIYLACLESEN
Snowboard Jacket as its top (and only) result, even though this
20-product catalog has no scarves at all. The other two documented
cases are about *how many* already-correctly-ordered results to keep
(padding a good result with a trailing bad one); this is about the
**#1 result itself being wrong** — `RERANK_MARGIN` and
`SHORTLIST_STD_FLOOR` only ever decide how many shortlist candidates
to return, never whether the top-ranked one is actually the right item
type, so neither mechanism can address this by design, not by a
missing tuning value.

`SHORTLIST_STD_FLOOR` doesn't catch it either, and isn't malfunctioning
when it doesn't: the shortlist genuinely has a "one candidate pulls
away from a flat tail" shape (std=3.415, floor=0.5) — the same shape a
real match produces. The floor can only tell "no discrimination at
all" (flat scores, e.g. "halloween costume": std=0.196) from "found
something with real signal," not whether that something is the right
something.

All three retrieval signals independently favor the wrong item, and at
higher magnitude than for a query that works correctly:

```
                         lexical   semantic   cross-encoder
winter scarf (wrong)      0.1442    0.4626      -0.642
gift for my dad (right)   0.1818    0.3243       0.469
```

Root cause for the lexical signal: `"scarf"` isn't in the fitted
TF-IDF vocabulary at all (no product ever uses that word), so
`VECTORIZER.transform(["winter scarf"])` silently drops it —
`scikit-learn` ignores out-of-vocabulary tokens rather than erroring —
and the resulting vector is identical to the one for `"winter"` alone.

Three fixes were tried and rejected, each targeting a different layer
of the pipeline:
- **Token-coverage / out-of-vocabulary check** (reject a query if any
  content word never appears in any product): also flags `"gift for my
  dad"` (`"dad"` is OOV too) — one of the most reliably-correct queries
  in the app, whose match works entirely through semantic similarity,
  not lexical overlap. Can't distinguish "missing word because we
  don't carry that item" from "missing word because the real match
  doesn't use it" — both look identical to a coverage check.
- **Per-word semantic decomposition** (embed each query word
  separately, require each to independently support the match):
  embedding `"scarf"` alone actually scores *higher* against the
  jacket (0.3341) than `"winter"` alone (0.2886) with the current
  model — the opposite of what would be needed to flag it. Scarves and
  jackets sit close together in embedding space simply as members of
  the same "cold-weather apparel" cluster; the model was never trained
  to separate specific product types within a topic, only to judge
  general topical/paraphrase relatedness.
- **Larger embedding model** (`all-mpnet-base-v2`, same
  `sentence-transformers` package, no new dependency): per-word
  discrimination improves (scarf 0.2977 < winter 0.3564, the correct
  direction) but the actual query-level similarity `search()` uses
  (`"winter scarf"` as one string) goes *up*, not down (0.4762 vs
  0.4626). Model capacity doesn't target this gap because it isn't a
  capacity problem — general-purpose sentence embeddings are trained
  for broad paraphrase/topic similarity, not fine-grained e-commerce
  product-type discrimination. Swapping `MODEL` would also invalidate
  every threshold already calibrated against its specific score
  distributions (`GENDER_INTENT_THRESHOLD`/`MARGIN`,
  `BROWSE_ALL_THRESHOLD`, `SHORTLIST_STD_FLOOR`) for no measured
  benefit.

Eight structurally different fixes have now been tried across this and
the winter-clothes case (five for the cutoff/count problem above,
three for this top-result problem), all evidenced to fail — treated as
a genuine architectural boundary (small, general-purpose models over a
tiny fixed catalog with no item of the requested type) rather than an
unfound fix. For context, this is one specific, adversarially-discovered
edge case on a system that otherwise works well: across the existing
test/calibration sets the app sits at 90-100% correctness (35/39
no-match cases, 9/9 gender-intent, 9/9 browse-intent, 0.74 mean
precision on labeled relevance) — not a sign of broad unreliability.

## Price filtering

A price bound (e.g. "only show jewelry under $20") is an exact, structured
constraint, not a fuzzy intent -- unlike gender or browse-all detection, it
isn't the kind of thing an embedding classifier should judge. **Decision:**
plain UI number inputs (`min_price`/`max_price`, alongside the existing
query box in `templates/index.html`), read by Flask with built-in numeric
coercion (`request.args.get("min_price", type=float)` -- returns `None` on
missing/invalid input, no parsing needed) and passed straight into
`search(query, min_price=None, max_price=None)`. A natural-language version
("jewelry under $20" typed as one phrase, parsed with regex) was considered
and rejected: it would only change *how the numbers get in*, since everything
downstream — the candidate-pool masking, the empty-pool guard, the
browse-all interaction — is identical either way, so the extra parsing
surface (phrasing variants, stripping the matched phrase before scoring)
wasn't worth it for a demo.

**Mechanism:** `_price_in_range(product, min_price, max_price)` is ANDed into
the same `keep` boolean mask `_shortlist_rerank_scores()` already builds for
gender-category exclusion -- no new masking mechanism, just a second
condition on the existing one. `search()`'s browse-all branch also applies
the price filter directly to the full catalog (cheap, and "show me
everything under $50" is a natural combination) but still does *not* apply
gender exclusion there, extending that already-documented scope
simplification.

**Empty-candidate-pool guard, newly required:** gender exclusion alone can
never empty the candidate pool (electronics/jewelry are never excluded by
gender, so at least 7 products always remain), but a price range can (e.g.
`min_price` above every product's price). Without an explicit guard,
`PRODUCT_EMBEDDINGS[candidates]` would get 0 rows and `np.std([])` would
return `nan` -- a `nan < SHORTLIST_STD_FLOOR` comparison is always `False`
in numpy, so `search()` would skip the existing no-match path and crash
later in `_confident_count` (`IndexError` on `sorted_scores[0]`) instead of
returning `[]` cleanly. `_shortlist_rerank_scores()` now returns empty arrays
when the post-price-filter candidate pool is empty, and `search()` checks
for that before computing `np.std(rerank_scores)`.

**Verified the "topic has zero matches in range" case separately** from the
"price range is empty" case above, since it's the main risk of filtering the
candidate pool *before* ranking rather than filtering results after the
fact: searching `"electronics"` against a pool already restricted to
`price < $50` (this catalog's cheapest electronics item is $64, so the
filtered pool has zero electronics in it) produces a shortlist std of 0.402
-- below `SHORTLIST_STD_FLOOR` (0.5). The existing no-match mechanism
already generalizes to this case with no new logic: a query with no matches
for its own topic, once forced into a same-price-band pool containing none
of that topic, produces the same flat, undifferentiated score pattern as any
other no-match query (e.g. "halloween costume", std=0.196). This is why
pre-filtering the candidate pool (rather than filtering the final ranked
results afterward, which risks silently truncating away a cheaper relevant
item before it's ever considered) was the right design.

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

## Browse-all intent

Queries like "all products" or "show me everything" have no specific
semantic target — running them through the normal retrieve-then-rerank
pipeline produces a top-5 that looks like a ranked, relevant result but is
actually close to arbitrary (whatever happens to have moderately higher
shortlist variance than pure noise; e.g. "all products" used to return a
T-shirt, a necklace, another T-shirt, earrings, and a monitor as if they
were the 5 best matches).

**Decision:** a dedicated embedding-based classifier, structured like the
gender-intent classifier (seed phrases, max cosine similarity), but with
just one threshold and no margin-against-a-second-class, since this is a
single yes/no direction rather than a choice between two competing
classes:
- `BROWSE_ALL_SEED_PHRASES`: 13 example phrases ("show me everything",
  "all products", "what do you have", ...), embedded once at startup into
  `BROWSE_ALL_SEED_EMBEDDINGS`.
- `_is_browse_all(query_embedding)` checks max cosine similarity against
  `BROWSE_ALL_THRESHOLD` (0.6). `search()` checks this immediately after
  encoding the query (before gender-exclusion/RRF/reranking) and, if
  true, returns the entire catalog directly, bypassing the rest of the
  pipeline.
- Calibrated with a real gap, unlike the failed category-membership
  classifier from "No-match detection" above: 8 true browse-all phrasings
  scored 0.692–1.000, while every specific-product query tested —
  including tricky category-specific-but-browse-phrased ones like "show
  me your jewelry" (0.473) — scored at most 0.523. This works where the
  broader 4-category classifier didn't because it tests one narrow,
  specific direction instead of a broad multi-category question.
- Deliberate scope simplification: bypasses gender-category exclusion
  too. A query combining both intents (e.g. "show me everything for my
  dad") is out of scope.

**To verify a change to this logic:** `tests/test_browse_intent.py` is a
real pass/fail regression test (this classifier separates cleanly enough
to be tested deterministically, unlike `SHORTLIST_STD_FLOOR`'s genuinely
interleaved calibration set) — run with `python -m pytest tests/ -v`.

## Evaluation workflow

Six separate mechanisms, because they answer different questions:

- **`tests/test_intent_classifier.py`** (pytest) — regression tests for
  `_excluded_category`. These have an objective right answer (does the
  query imply a gender, and does the classifier catch it), so they're
  real pass/fail tests. Install dev deps with
  `pip install -r requirements-dev.txt`, run with `python -m pytest tests/ -v`.
  Run this after any change to the seed phrases, threshold, or margin.

- **`tests/test_browse_intent.py`** (pytest) — same treatment for
  `_is_browse_all` (see "Browse-all intent" above). Run after any change
  to `BROWSE_ALL_SEED_PHRASES` or `BROWSE_ALL_THRESHOLD`.

- **`tests/test_no_match_floor.py`** (pytest) — regression coverage for
  `SHORTLIST_STD_FLOOR` (see "No-match detection" above) on a small, safe
  subset of queries (not the full `calibrate_std_floor.py` set, which
  includes documented known misses that would make a pass/fail test
  flaky). Run after any change to retrieval/reranking.

- **`tests/test_price_filter.py`** (pytest) — regression coverage for the
  `min_price`/`max_price` candidate-pool masking (see "Price filtering"
  above), including the empty-candidate-pool guard and the "topic has zero
  matches in range" no-match case. Run after any change to
  `_price_in_range()` or the candidate-masking logic in
  `_shortlist_rerank_scores()`.

- **`eval_relevance.py`** — hand-labeled relevance eval for `search()`
  overall (RRF-ranked results, not just the gender filter). "Is product X
  relevant to query Y" isn't objective, so this isn't a pass/fail test —
  it's a small labeled dataset (`LABELED_QUERIES`: query -> relevant
  product ids) scored via precision/recall against live search output,
  each computed within that query's own returned window since `search()`
  returns a dynamic count, not always `TOP_K` (see "Dynamic result count"
  above). The seeded labels were picked by reading the product catalog,
  not by you — review/adjust `LABELED_QUERIES` before treating the
  scores as real ground truth. Run with `python eval_relevance.py`.
  Useful for catching ranking issues the classifier tests can't see (e.g.
  it already surfaced that "gift for my dad" ranks jewelry above two
  relevant men's shirts — a real gap in the RRF ranking, not the intent
  classifier).

- **`calibrate_std_floor.py`** — F1 sweep for `SHORTLIST_STD_FLOOR` (see
  "No-match detection" above) against a labeled `VALID_QUERIES`/
  `JUNK_QUERIES` set. Reuses `app._shortlist_rerank_scores()` (the same
  pipeline `search()` calls) rather than reimplementing it, so it can't
  drift out of sync. Run with `python calibrate_std_floor.py`. Run this
  after any change to retrieval/reranking that could shift shortlist
  score distributions, and re-run after extending the query set.

- **`calibrate_rerank_cutoff.py`** — not a calibration to adopt a value
  from (unlike `calibrate_std_floor.py`), but checked-in reproducible
  evidence for the `RERANK_MARGIN` "winter clothes" limitation documented
  above: it prints all five attempted alternative cutoff approaches'
  actual numbers against `LABELED_QUERIES`, confirming none beats the
  `RERANK_MARGIN=5.0` baseline without regressing something. Run with
  `python calibrate_rerank_cutoff.py`.

Add new cases to these files as the catalog or use cases grow, rather
than one-off manual scripts.

## Repo state

- Git repo initialized and pushed to
  https://github.com/sparsh2695/product-search-demo (public).
- `venv/` is gitignored — not committed.
