"""
Sketch of a hand-labeled relevance eval for search().

Unlike the intent-classifier tests (tests/test_intent_classifier.py), there's
no objective right answer for "is product X relevant to query Y" — it's a
judgment call. This script structures that judgment as a small labeled
dataset (query -> set of relevant product ids, chosen by manually reading
the 20-product catalog) and scores search()'s actual output against it.

The labels below are a STARTING POINT (seeded by re-reading each product's
title/category), not ground truth — review and adjust the relevant_ids sets
to match what you'd actually consider a good result before trusting the
scores. Extend LABELED_QUERIES with more queries as the catalog or use
cases grow.

search() returns a dynamic number of results (see RERANK_MARGIN in
app.py), not always TOP_K, so both metrics below are computed against
each query's own returned window rather than assuming a fixed size:
precision already does this naturally (dividing by what was actually
returned); recall is explicitly normalized by min(window, |relevant|) so
a deliberately short, fully-correct answer doesn't get penalized as if it
should have surfaced every relevant item in the whole catalog.

Usage:
    source venv/bin/activate
    python eval_relevance.py
"""

from app import search, PRODUCTS, TOP_K

# query -> set of product ids considered relevant (see fakestoreapi.com/products
# for the full catalog; ids and categories are listed in CLAUDE.md history / the
# curl output used to seed these labels)
LABELED_QUERIES = {
    "something warm for winter": {
        15,  # BIYLACLESEN Women's 3-in-1 Snowboard Jacket Winter Coats
        16,  # Lock and Love Women's ... Moto Biker Jacket
        3,   # Mens Cotton Jacket
    },
    "gift for my dad": {
        1,  # Fjallraven Backpack
        3,  # Mens Cotton Jacket
        2,  # Mens Casual Premium Slim Fit T-Shirts
        4,  # Mens Casual Slim Fit
    },
    "gear for outdoor hiking": {
        1,   # Fjallraven Backpack
        15,  # Snowboard Jacket Winter Coats
        17,  # Rain Jacket Women Windbreaker Striped Climbing Raincoats
    },
    "everyday jewelry": {
        6,  # Solid Gold Petite Micropave
        7,  # White Gold Plated Princess
        8,  # Pierced Owl earrings
    },
    "storage for my laptop": {
        9,   # WD 2TB Elements Portable External Hard Drive
        10,  # SanDisk SSD PLUS 1TB
        11,  # Silicon Power 256GB SSD
    },
}


def precision_at_k(returned_ids, relevant_ids, k):
    """Already naturally adapts to a short result list: slicing
    returned_ids[:k] when len(returned_ids) < k just yields the whole
    (shorter) list, and dividing by len(top_k) -- not by k -- means this
    already measures precision over what was actually shown, not an
    assumed k-length window. No dynamic-cutoff-specific fix needed here."""
    top_k = returned_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for pid in top_k if pid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(returned_ids, relevant_ids, k):
    """Recall@min(k, len(returned_ids)), normalized by
    min(that window, len(relevant_ids)) rather than the raw relevant-item
    count. With search()'s dynamic result cutoff, a query can legitimately
    return fewer than k results when the reranker is only confident about
    one or two of them (e.g. "gift for dad" -> 1 result). Dividing by the
    full relevant-item count would then treat a correct, deliberately
    short answer as if it should have surfaced every relevant item in the
    whole catalog (e.g. 1 hit / 4 relevant = 0.25, even though the single
    result shown was exactly right). Normalizing by min(window shown,
    relevant count) instead means recall reaches 1.0 whenever the (short)
    result list contains everything it had room to show."""
    if not relevant_ids:
        return None
    window = min(k, len(returned_ids))
    top_k = returned_ids[:window]
    denom = min(window, len(relevant_ids))
    if denom == 0:
        return 0.0
    hits = sum(1 for pid in top_k if pid in relevant_ids)
    return hits / denom


def main():
    id_to_title = {p["id"]: p["title"] for p in PRODUCTS}

    precisions = []
    for query, relevant_ids in LABELED_QUERIES.items():
        results = search(query)
        returned_ids = [p["id"] for p in results]

        n = len(returned_ids)
        precision = precision_at_k(returned_ids, relevant_ids, TOP_K)
        recall = recall_at_k(returned_ids, relevant_ids, TOP_K)
        precisions.append(precision)

        print(f"\nQuery: {query!r}")
        print(f"  Precision@{n}: {precision:.2f}   Recall@{n}: {recall:.2f}"
              f"  ({n} result{'s' if n != 1 else ''} returned)")
        for pid in returned_ids:
            mark = "✓" if pid in relevant_ids else " "
            print(f"    [{mark}] {pid:>2} {id_to_title[pid]}")
        missed = relevant_ids - set(returned_ids)
        if missed:
            print("  Missed relevant items:")
            for pid in missed:
                print(f"      {pid:>2} {id_to_title[pid]}")

    print(f"\nMean Precision (within each query's own returned window) "
          f"across {len(LABELED_QUERIES)} queries: "
          f"{sum(precisions) / len(precisions):.2f}")


if __name__ == "__main__":
    main()
