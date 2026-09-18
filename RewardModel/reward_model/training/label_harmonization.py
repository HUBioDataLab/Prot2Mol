"""Harmonise contradictory activity labels by pair-median pChEMBL.

The ChEMBL source contains the same (target, compound) pair measured in several
assays. Because pChEMBL carries roughly 0.6 log units of measurement noise, a
pair whose true potency sits near the 6.0 activity cut can be recorded above the
cut in one assay and below it in another, which puts contradictory binary labels
on the identical protein-ligand interaction.

This module resolves that by taking the median pChEMBL over every measurement of
a pair -- across all splits -- and giving every row of the pair the label implied
by that median. The median is used rather than the minimum because averaging
repeated measurements reduces their noise (sigma/sqrt(n)), so it moves labels
towards the true potency instead of biasing them downwards.

Only ``binary_label`` is rewritten. Every ``pchembl_value`` is left exactly as
ChEMBL reported it, so the contrastive loss -- which selects actives by raw
pChEMBL -- keeps its original behaviour.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import pandas as pd

DEFAULT_PAIR_KEY: Tuple[str, ...] = ("target_chembl_id", "compound_id")
DEFAULT_ACTIVITY_THRESHOLD = 6.0

REQUIRED_COLUMNS = ("pchembl_value", "binary_label")


def _validate(frames: Mapping[str, pd.DataFrame], pair_key: Sequence[str]) -> None:
    if not frames:
        raise ValueError("at least one split frame is required")
    for name, frame in frames.items():
        missing = [c for c in (*pair_key, *REQUIRED_COLUMNS) if c not in frame.columns]
        if missing:
            raise ValueError(f"split '{name}' is missing columns: {missing}")
        if frame["pchembl_value"].isna().any():
            raise ValueError(f"split '{name}' contains null pchembl_value")


def harmonize_labels_by_pair_median(
    frames: Mapping[str, pd.DataFrame],
    *,
    activity_threshold: float = DEFAULT_ACTIVITY_THRESHOLD,
    pair_key: Sequence[str] = DEFAULT_PAIR_KEY,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Return relabelled copies of ``frames`` plus a summary of what changed.

    The median is pooled over every split so that a pair receives one label
    everywhere it appears. Splits themselves are never re-assigned.
    """
    pair_key = tuple(pair_key)
    _validate(frames, pair_key)

    order = list(frames)
    stacked = pd.concat(
        [frames[name].assign(_split=name) for name in order],
        ignore_index=True,
        copy=False,
    )
    grouped = stacked.groupby(list(pair_key), sort=False)["pchembl_value"]
    median = grouped.transform("median")
    pair_min = grouped.transform("min")
    pair_max = grouped.transform("max")

    new_label = (median >= activity_threshold).astype(stacked["binary_label"].dtype)
    conflicting = (pair_min < activity_threshold) & (pair_max >= activity_threshold)
    changed = new_label != stacked["binary_label"]

    pairs = stacked.drop_duplicates(list(pair_key))
    summary: Dict[str, Any] = {
        "activity_threshold": float(activity_threshold),
        "pair_key": list(pair_key),
        "rows_total": int(len(stacked)),
        "pairs_total": int(len(pairs)),
        "pairs_conflicting": int(conflicting[pairs.index].sum()),
        "rows_relabelled": int(changed.sum()),
        "rows_positive_to_negative": int((changed & (stacked["binary_label"] == 1)).sum()),
        "rows_negative_to_positive": int((changed & (stacked["binary_label"] == 0)).sum()),
        "splits": {},
    }

    out: Dict[str, pd.DataFrame] = {}
    for name in order:
        mask = (stacked["_split"] == name).to_numpy()
        frame = frames[name].copy()
        before = frame["binary_label"]
        after = pd.Series(new_label.to_numpy()[mask], index=frame.index, name="binary_label")
        frame["binary_label"] = after
        out[name] = frame
        summary["splits"][name] = {
            "rows": int(len(frame)),
            "rows_relabelled": int((after != before).sum()),
            "positive_rate_before": float(before.mean()),
            "positive_rate_after": float(after.mean()),
        }

    # After harmonisation no pair may hold two different labels anywhere.
    check = pd.concat([out[n][[*pair_key, "binary_label"]] for n in order], ignore_index=True)
    remaining = check.groupby(list(pair_key), sort=False)["binary_label"].nunique()
    leftover = int((remaining > 1).sum())
    if leftover:
        raise AssertionError(f"{leftover} pairs still hold contradictory labels")
    summary["pairs_with_contradictions_after"] = 0
    return out, summary
