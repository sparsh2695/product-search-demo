# Product Search Demo

**Live demo:** _add your Render URL here after deploying, e.g. https://product-search-demo.onrender.com_
_(free-tier instance — the first request after a few minutes of inactivity takes ~30-60s to wake it up)_

A small Flask demo of hybrid product search over the [Fake Store
API](https://fakestoreapi.com/products) catalog (20 products): Reciprocal
Rank Fusion over semantic (bi-encoder) and lexical (TF-IDF) similarity,
reranked by a cross-encoder, with embedding-based gender-intent, browse-all,
and no-match detection, plus price filtering.

Try queries like:
- `gift for my dad`
- `everyday jewelry`
- `all products`
- `jewelry` with a max price set
- `halloween costume` (nothing in the catalog — returns no results on purpose)

See [`CLAUDE.md`](CLAUDE.md) for the full architecture writeup, including the
documented, known limitations of the search ranking (and why they were left
as limitations rather than papered over with an unreliable fix).
