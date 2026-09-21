"""
Calibration sweep for SHORTLIST_STD_FLOOR (see the constant's comment in
app.py for what it does and why).

Like LABELED_QUERIES in eval_relevance.py, the VALID_QUERIES/JUNK_QUERIES
labels below are a judgment call made by reading the 20-product catalog,
not independently verified ground truth -- review/extend them before
trusting the sweep output. "Valid" means the catalog has a genuine match;
"junk" means it doesn't, so search() should return no results.

This reuses app._shortlist_rerank_scores() -- the same retrieve-then-
rerank pipeline search() calls -- rather than reimplementing it, so this
script can't silently drift out of sync with actual search() behavior.

Usage:
    source venv/bin/activate
    python calibrate_std_floor.py
"""

import numpy as np

from app import _shortlist_rerank_scores

VALID_QUERIES = [
    "something warm for winter", "gift for my dad", "gear for outdoor hiking",
    "everyday jewelry", "storage for my laptop", "red dress", "gift for my mom",
    "birthday present for my sister", "backpack", "gold necklace",
    "gaming monitor", "t-shirt", "gift for my grandma", "necklace for a wedding",
    "laptop bag", "casual shirt", "diamond ring", "winter coat",
    "external storage drive", "curved monitor", "moisture wicking shirt",
    "raincoat", "gift for my son",
]

JUNK_QUERIES = [
    "halloween costume", "sunglasses", "kitchen appliance", "phone case",
    "video game console", "birthday cake", "cooking pot", "umbrella",
    "wireless headphones", "warm gloves", "coffee mug", "running shoes",
    "yoga mat", "board game", "scented candle", "bluetooth speaker",
]


def shortlist_std(query):
    _, rerank_scores = _shortlist_rerank_scores(query)
    return float(np.std(rerank_scores))


def sweep(valid_stds, junk_stds):
    """Score every candidate threshold (each observed std value) on
    accuracy/precision/recall/F1, return the F1-best one plus the
    contiguous score interval around it that gives identical results.

    Classification only changes at an observed data point (crossing one
    flips exactly one query's in/out status), so the interval between
    two adjacent distinct values is safe to treat as one candidate --
    evaluating only at the values themselves (as opposed to a fixed-step
    grid) would otherwise collapse that whole safe range down to the
    single point it happens to land on."""
    candidates = sorted(set(valid_stds + junk_stds))
    results = []
    for i, t in enumerate(candidates):
        tp = sum(1 for v in valid_stds if v >= t)
        fn = len(valid_stds) - tp
        fp = sum(1 for j in junk_stds if j >= t)
        tn = len(junk_stds) - fp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        acc = (tp + tn) / (len(valid_stds) + len(junk_stds))
        interval_lo = candidates[i - 1] if i > 0 else t - 1.0
        results.append((t, f1, acc, tp, fn, fp, tn, interval_lo))

    best_f1 = max(r[1] for r in results)
    return [r for r in results if r[1] == best_f1]


def main():
    print("Computing shortlist score std for each query (loads models, may take a moment)...\n")

    valid_stds = {q: shortlist_std(q) for q in VALID_QUERIES}
    junk_stds = {q: shortlist_std(q) for q in JUNK_QUERIES}

    rows = [(s, q, "VALID") for q, s in valid_stds.items()]
    rows += [(s, q, "JUNK") for q, s in junk_stds.items()]
    rows.sort(key=lambda r: -r[0])

    print(f"{'std':>8s}  label   query")
    for s, q, label in rows:
        print(f"{s:8.3f}  {label:5s}   {q!r}")

    best = sweep(list(valid_stds.values()), list(junk_stds.values()))
    lo = min(r[7] for r in best)
    hi = max(r[0] for r in best)
    t, f1, acc, tp, fn, fp, tn, _ = best[0]

    print(f"\nF1-optimal threshold interval: ({lo:.3f}, {hi:.3f}]")
    print(f"  F1={f1:.3f}  accuracy={acc:.3f}")
    print(f"  valid kept: {tp}/{len(VALID_QUERIES)}   junk rejected: {tn}/{len(JUNK_QUERIES)}")

    fp_queries = [q for q, s in junk_stds.items() if s >= hi]
    fn_queries = [q for q, s in valid_stds.items() if s < hi]
    print(f"\n  False positives (junk kept) at this threshold: {fp_queries}")
    print(f"  False negatives (valid dropped) at this threshold: {fn_queries}")


if __name__ == "__main__":
    main()
