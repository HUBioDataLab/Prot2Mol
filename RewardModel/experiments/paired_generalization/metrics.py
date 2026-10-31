"""Tie-aware, within-assay diagnostics specified by the experiment protocol.

These metrics do not replace the baseline's checkpoint-selection metric.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import kendalltau, spearmanr


@dataclass(frozen=True)
class PairCounts:
    all_pairs: int
    comparable_pairs: int
    concordant: int
    discordant: int
    prediction_ties: int
    label_band_pairs: int

    @property
    def concordance_all_pairs(self):
        return (self.concordant - self.discordant) / self.all_pairs if self.all_pairs else None

    @property
    def ordering_accuracy(self):
        if not self.comparable_pairs:
            return None
        return (self.concordant + .5 * self.prediction_ties) / self.comparable_pairs


def pair_counts(labels, scores, epsilon=0.0):
    """Count exact within-group pairs in O(n log n), without a pair matrix.

    Label differences <= epsilon are ties/excluded from ordering accuracy.
    Predicted ties receive half credit on label-comparable pairs. The all-pairs
    concordance uses ALL within-group pairs in its denominator, as in Tier 1.
    """
    y = np.asarray(labels, dtype=float).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if len(y) != len(s) or not np.isfinite(y).all() or not np.isfinite(s).all():
        raise ValueError("Labels and scores must be aligned and finite")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    order = np.argsort(y, kind="stable")
    y, s = y[order], s[order]
    _, ranks = np.unique(s, return_inverse=True)
    tree = np.zeros(len(np.unique(s)) + 1, dtype=np.int64)

    def insert(rank):
        index = int(rank) + 1
        while index < len(tree):
            tree[index] += 1
            index += index & -index

    def prefix(count):
        total = 0
        index = int(count)
        while index:
            total += int(tree[index])
            index -= index & -index
        return total

    left = concordant = discordant = ties = comparable = 0
    for right, rank in enumerate(ranks):
        while left < right and y[right] - y[left] > epsilon:
            insert(ranks[left])
            left += 1
        below = prefix(rank)
        through = prefix(rank + 1)
        concordant += below
        ties += through - below
        discordant += left - through
        comparable += left
    all_pairs = len(y) * (len(y) - 1) // 2
    return PairCounts(all_pairs, comparable, concordant, discordant, ties, all_pairs - comparable)


def size_bin(size):
    if size < 2:
        return "1"
    if size in (2, 3):
        return str(size)
    if size <= 10:
        return "4-10"
    if size <= 25:
        return "11-25"
    return ">25"


def group_metrics(labels, scores, epsilon=None, original_min_size=3, original_min_span=.5):
    y = np.asarray(labels, dtype=float).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    exact = pair_counts(y, s)
    nonconstant = len(y) >= 2 and np.ptp(y) > 0 and np.ptp(s) > 0
    rho = float(spearmanr(y, s).statistic) if nonconstant else None
    tau = float(kendalltau(y, s, variant="b").statistic) if nonconstant else None
    result = {
        "size": len(y), "size_bin": size_bin(len(y)),
        "label_span": float(np.ptp(y)) if len(y) else None,
        "label_variance": float(np.var(y)) if len(y) else None,
        "spearman": rho, "kendall_tau_b": tau,
        "spearman_defined": rho is not None,
        "original_ranking_eligible": len(y) >= original_min_size and np.ptp(y) >= original_min_span,
        "pairs": asdict(exact),
        "concordance_all_pairs": exact.concordance_all_pairs,
        "ordering_accuracy": exact.ordering_accuracy,
    }
    if epsilon is not None:
        band = pair_counts(y, s, epsilon=epsilon)
        result.update(epsilon=epsilon, epsilon_pairs=asdict(band), epsilon_ordering_accuracy=band.ordering_accuracy)
    return result


def aggregate_groups(records):
    rows = list(records)
    defined = [r for r in rows if r["spearman"] is not None]
    count = lambda name: sum(r["pairs"][name] for r in rows)
    all_pairs, comparable = count("all_pairs"), count("comparable_pairs")
    concordant, discordant, ties = count("concordant"), count("discordant"), count("prediction_ties")
    original = [r for r in defined if r["original_ranking_eligible"]]
    return {
        "groups": len(rows), "defined_groups": len(defined),
        "defined_fraction": len(defined) / len(rows) if rows else None,
        "macro_spearman": float(np.mean([r["spearman"] for r in defined])) if defined else None,
        "size_weighted_spearman": float(np.average([r["spearman"] for r in defined], weights=[r["size"] for r in defined])) if defined else None,
        "original_eligible_macro_spearman": float(np.mean([r["spearman"] for r in original])) if original else None,
        "macro_kendall_tau_b": float(np.mean([r["kendall_tau_b"] for r in defined])) if defined else None,
        "pooled_within_group_concordance": (concordant - discordant) / all_pairs if all_pairs else None,
        "pooled_ordering_accuracy": (concordant + .5 * ties) / comparable if comparable else None,
        "all_pairs": all_pairs, "comparable_pairs": comparable,
    }
