"""
Investigation record for RERANK_MARGIN's known "winter clothes" limitation
(see the comment on RERANK_MARGIN in app.py). Unlike calibrate_std_floor.py,
this script isn't calibrating a value to adopt -- every candidate approach
tried here was evidenced to fail, and RERANK_MARGIN is staying as-is. This
is checked in as the reproducible evidence for that conclusion, the same
way calibrate_std_floor.py's output documents its own known misses instead
of just asserting them in a comment.

The motivating case: for the query "winter clothes", RERANK_MARGIN=5.0
correctly keeps two real winter jackets but also keeps an irrelevant
summer T-shirt (DANVOUY Womens T Shirt), because the T-shirt's cross-encoder
score sits only 4.194 points below the top score -- inside the margin. That
margin can't be tightened without also dropping a genuinely relevant
backpack in a different query ("gear for outdoor hiking"), which needs
>=4.763 points of margin to survive (that's the case RERANK_MARGIN=5.0 was
originally tuned to protect -- see its comment). No fixed margin satisfies
both: 4.194 < 4.763.

Five approaches were tried to replace or complement the fixed margin, using
the whole shortlist's score shape/distribution instead of a flat number.
All five failed -- either they still don't exclude the T-shirt, or they fix
it at the cost of regressing an already-correct query. See CLAUDE.md's
"Dynamic result count" section for the narrative summary; this script
prints the actual numbers behind that summary and re-checks candidates 2
and 4 (the two with clean enough logic to implement and rerun) against the
full LABELED_QUERIES set.

Usage:
    source venv/bin/activate
    python calibrate_rerank_cutoff.py
"""

import numpy as np

from app import PRODUCTS, TOP_K, _confident_count, _shortlist_rerank_scores
from eval_relevance import LABELED_QUERIES, precision_at_k, recall_at_k

WINTER_CLOTHES_QUERY = "winter clothes"
HIKING_QUERY = "gear for outdoor hiking"


def knee_count(sorted_scores, top_k):
    """Candidate 2: Kneedle-style knee detection. Normalize the sorted-
    descending score curve to [0,1]x[0,1] and cut right after the point
    of maximum vertical distance below the chord from (0,1) to (1,0) --
    the 'knee' where the curve bends hardest from real signal into flat
    noise tail."""
    n = len(sorted_scores)
    if n <= 1:
        return min(n, top_k)
    score_range = sorted_scores[0] - sorted_scores[-1]
    if score_range == 0:
        return min(n, top_k)
    x = np.arange(n) / (n - 1)
    y = (sorted_scores - sorted_scores[-1]) / score_range
    distance_below_chord = y - (1 - x)
    knee_idx = int(np.argmin(distance_below_chord))
    return min(knee_idx + 1, top_k)


def tail_noise_floor(sorted_scores, flat_eps=0.15, min_run=3):
    """Estimate the flat-tail noise floor by walking up from the bottom
    of the sorted-descending shortlist as long as each step's gap stays
    under flat_eps."""
    n = len(sorted_scores)
    run_end = n
    run_start = n - 1
    for i in range(n - 1, 0, -1):
        if sorted_scores[i - 1] - sorted_scores[i] > flat_eps:
            break
        run_start = i - 1
    if run_end - run_start < min_run:
        run_start = n // 2
    tail = sorted_scores[run_start:run_end]
    return float(np.mean(tail)), float(np.std(tail))


def noise_floor_count(sorted_scores, top_k, z):
    """Candidate 3: tail-anchored noise-floor significance test. Keep
    candidates that are z standard deviations above the estimated flat-
    tail noise floor, instead of comparing to the top score or to
    neighboring candidates."""
    mean, std = tail_noise_floor(sorted_scores)
    std = max(std, 0.05)
    within = sorted_scores >= mean + z * std
    return min(int(np.count_nonzero(within)), top_k)


CANDIDATES = {
    "baseline (RERANK_MARGIN=5.0)": lambda s, k: _confident_count(s, k),
    "knee (Kneedle)": knee_count,
    "noise_floor_z2.0": lambda s, k: noise_floor_count(s, k, z=2.0),
    "noise_floor_z3.0": lambda s, k: noise_floor_count(s, k, z=3.0),
}


def evaluate(candidate_fn):
    precisions, recalls = [], []
    per_query = {}
    for query, relevant_ids in LABELED_QUERIES.items():
        shortlist_ids, rerank_scores = _shortlist_rerank_scores(query)
        order = np.argsort(-rerank_scores)
        sorted_scores = rerank_scores[order]
        sorted_ids = shortlist_ids[order]

        n = candidate_fn(sorted_scores, TOP_K)
        returned_pids = [PRODUCTS[i]["id"] for i in sorted_ids[:n]]

        p = precision_at_k(returned_pids, relevant_ids, TOP_K)
        r = recall_at_k(returned_pids, relevant_ids, TOP_K)
        precisions.append(p)
        recalls.append(r)
        per_query[query] = (n, p, r, returned_pids)
    return precisions, recalls, per_query


def print_winter_clothes_breakdown():
    print(f"=== {WINTER_CLOTHES_QUERY!r} shortlist, all candidates ===")
    shortlist_ids, rerank_scores = _shortlist_rerank_scores(WINTER_CLOTHES_QUERY)
    order = np.argsort(-rerank_scores)
    sorted_scores = rerank_scores[order]
    sorted_ids = shortlist_ids[order]

    for name, fn in CANDIDATES.items():
        n = fn(sorted_scores, TOP_K)
        titles = [PRODUCTS[i]["title"][:40] for i in sorted_ids[:n]]
        t_shirt_included = any("DANVOUY" in PRODUCTS[i]["title"] for i in sorted_ids[:n])
        flag = "  <-- still includes the T-shirt" if t_shirt_included else "  (T-shirt excluded)"
        print(f"  {name:30s} keeps {n}: {titles}{flag}")


def main():
    print_winter_clothes_breakdown()

    print(f"\n=== Mean precision/recall across {len(LABELED_QUERIES)} LABELED_QUERIES ===")
    for name, fn in CANDIDATES.items():
        precisions, recalls, per_query = evaluate(fn)
        mean_p = sum(precisions) / len(precisions)
        mean_r = sum(recalls) / len(recalls)
        print(f"\n{name}: mean precision={mean_p:.2f}  mean recall={mean_r:.2f}")
        for query, (n, p, r, _) in per_query.items():
            print(f"    {query!r:35s} n={n} P={p:.2f} R={r:.2f}")

    print(
        "\nNote: cross-signal corroboration (semantic+lexical agreement) was also "
        "tried but isn't reproduced here as a swappable candidate function -- it "
        "failed for a different reason than a bad threshold (the T-shirt is "
        "independently rank 2-3/20 on both the semantic and lexical retrieval "
        "signals, not just the cross-encoder, so no corroboration threshold "
        "excludes it without also excluding genuinely correct jewelry results "
        "in the separately-documented backpack case). See CLAUDE.md for the "
        "full numbers from that investigation."
    )


if __name__ == "__main__":
    main()
