import pandas as pd

from reward_model.training import split_seen_target_assay_holdout


def _source_frame():
    rows = []
    specifications = {
        ("target-a", "AAAA", "cluster-a"): ("a1", "a2", "a3"),
        ("target-b", "BBBB", "cluster-b"): ("b1",),
        ("target-c", "CCCC", "cluster-c"): ("c1", "c2"),
    }
    for (target, sequence, cluster), assays in specifications.items():
        for assay in assays:
            for compound_index in range(2):
                rows.append(
                    {
                        "target_chembl_id": target,
                        "protein_sequence": sequence,
                        "protein_cluster_50": cluster,
                        "assay_group_id": f"{target}:{assay}",
                        "compound_id": f"{assay}-compound-{compound_index}",
                        "split": "train",
                    }
                )
    return pd.DataFrame(rows)


def test_seen_target_split_is_deterministic_and_holds_out_whole_assays():
    source = _source_frame()
    first_train, first_val2, first_summary = split_seen_target_assay_holdout(
        source,
        validation_fraction=0.25,
        seed=42,
    )
    second_train, second_val2, second_summary = split_seen_target_assay_holdout(
        source,
        validation_fraction=0.25,
        seed=42,
    )

    assert first_train.equals(second_train)
    assert first_val2.equals(second_val2)
    assert first_summary == second_summary
    assert len(first_train) + len(first_val2) == len(source)
    assert set(first_train["assay_group_id"]).isdisjoint(
        set(first_val2["assay_group_id"])
    )
    assert set(first_val2["target_chembl_id"]).issubset(
        set(first_train["target_chembl_id"])
    )
    assert set(first_val2["protein_sequence"]).issubset(
        set(first_train["protein_sequence"])
    )
    assert set(first_val2["protein_cluster_50"]).issubset(
        set(first_train["protein_cluster_50"])
    )
    assert "target-b:b1" not in set(first_val2["assay_group_id"])
    assert set(first_val2["split"]) == {"val2"}
    assert first_summary["assay_overlap"] == 0
    assert first_summary["val2_targets_absent_from_train"] == 0
    assert first_summary["val2_protein_sequences_absent_from_train"] == 0
