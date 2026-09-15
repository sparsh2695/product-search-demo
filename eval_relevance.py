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

Usage:
    source venv/bin/activate
    python eval_relevance.py
"""

from app import search, PRODUCTS

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
    top_k = returned_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for pid in top_k if pid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(returned_ids, relevant_ids, k):
    if not relevant_ids:
        return None
    top_k = returned_ids[:k]
    hits = sum(1 for pid in top_k if pid in relevant_ids)
    return hits / len(relevant_ids)


def main():
    id_to_title = {p["id"]: p["title"] for p in PRODUCTS}

    precisions = []
    for query, relevant_ids in LABELED_QUERIES.items():
        results = search(query)
        returned_ids = [p["id"] for p in results]

        p_at_5 = precision_at_k(returned_ids, relevant_ids, 5)
        r_at_5 = recall_at_k(returned_ids, relevant_ids, 5)
        precisions.append(p_at_5)

        print(f"\nQuery: {query!r}")
        print(f"  Precision@5: {p_at_5:.2f}   Recall@5: {r_at_5:.2f}")
        for pid in returned_ids:
            mark = "✓" if pid in relevant_ids else " "
            print(f"    [{mark}] {pid:>2} {id_to_title[pid]}")
        missed = relevant_ids - set(returned_ids)
        if missed:
            print("  Missed relevant items:")
            for pid in missed:
                print(f"      {pid:>2} {id_to_title[pid]}")

    print(f"\nMean Precision@5 across {len(LABELED_QUERIES)} queries: "
          f"{sum(precisions) / len(precisions):.2f}")


if __name__ == "__main__":
    main()
