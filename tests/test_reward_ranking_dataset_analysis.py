import math

import pandas as pd

from data_processing.analyze_reward_ranking_dataset import (
    _activity_type_statistics,
    build_assay_statistics,
    count_comparable_pairs,
)


def _frame():
    rows = []
    for group_id, values, activity_types in [
        ("A", [5.0, 5.4, 6.0], ["IC50", "IC50|Ki", "Ki"]),
        ("B", [4.0, 8.0], ["Kd", "Kd"]),
    ]:
        for index, (value, activity_type) in enumerate(zip(values, activity_types)):
            rows.append(
                {
                    "assay_group_id": group_id,
                    "assay_id": f"assay-{group_id}",
                    "target_chembl_id": f"target-{group_id}",
                    "compound_id": f"{group_id}-{index}",
                    "pchembl_value": value,
                    "activity_type": activity_type,
                }
            )
    return pd.DataFrame(rows)


def test_count_comparable_pairs_uses_strict_affinity_margin():
    margin = math.log10(3.0)
    assert count_comparable_pairs([5.0, 5.4, 6.0], margin) == 2
    assert count_comparable_pairs([5.0, 5.0 + margin], margin) == 0
    assert count_comparable_pairs([5.0, 5.0 + margin + 1e-9], margin) == 1


def test_assay_statistics_match_reward_ranking_eligibility():
    stats = build_assay_statistics(
        _frame(),
        boundary="test",
        split="all",
        min_ligands=3,
        min_pchembl_span=0.5,
        affinity_margin=math.log10(3.0),
    ).set_index("assay_group_id")

    assert bool(stats.loc["A", "eligible_ranking_assay"])
    assert stats.loc["A", "ligands"] == 3
    assert stats.loc["A", "pchembl_span"] == 1.0
    assert stats.loc["A", "comparable_comparisons"] == 2
    assert not bool(stats.loc["B", "eligible_ranking_assay"])
    assert stats.loc["B", "comparable_comparisons"] == 1


def test_activity_type_statistics_report_memberships_and_mixed_rows():
    records = _activity_type_statistics(_frame(), boundary="test", split="all")
    membership = {
        record["activity_type"]: record["rows"]
        for record in records
        if record["view"] == "exploded_membership"
    }
    mixed = next(record for record in records if record["view"] == "mixed_rows")

    assert membership == {"IC50": 2, "Kd": 2, "Ki": 2}
    assert mixed["rows"] == 1
