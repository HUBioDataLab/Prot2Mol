import sqlite3

import pandas as pd

from data_processing.build_chembl_binding_dataset import (
    assign_cluster_splits_balanced,
    build_activity_balanced_cluster_stats,
    build_activity_query,
    build_cluster_disjoint_splits,
    curate_activity_rows,
    extract_activity_rows,
    validate_dataset,
)
from data_processing.build_protein_cluster_splits import SplitConfig


def _create_toy_chembl(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE activities (
          assay_id INTEGER, molregno INTEGER, pchembl_value REAL,
          standard_type TEXT, standard_relation TEXT, data_validity_comment TEXT
        );
        CREATE TABLE assays (
          assay_id INTEGER, chembl_id TEXT, assay_group TEXT, confidence_score INTEGER,
          assay_type TEXT, tid INTEGER, variant_id INTEGER
        );
        CREATE TABLE target_dictionary (
          tid INTEGER, chembl_id TEXT, target_type TEXT, species_group_flag INTEGER
        );
        CREATE TABLE target_components (tid INTEGER, component_id INTEGER);
        CREATE TABLE component_sequences (
          component_id INTEGER, accession TEXT, sequence TEXT, component_type TEXT
        );
        CREATE TABLE molecule_dictionary (molregno INTEGER, chembl_id TEXT);
        CREATE TABLE molecule_hierarchy (molregno INTEGER, parent_molregno INTEGER);
        CREATE TABLE compound_structures (molregno INTEGER, canonical_smiles TEXT);
        """
    )
    connection.executemany(
        "INSERT INTO target_dictionary VALUES (?, ?, ?, ?)",
        [
            (1, "CHEMBL_T1", "SINGLE PROTEIN", 0),
            (2, "CHEMBL_T2", "PROTEIN COMPLEX", 0),
            (3, "CHEMBL_T3", "SINGLE PROTEIN", 1),
        ],
    )
    connection.executemany(
        "INSERT INTO target_components VALUES (?, ?)",
        [(1, 11), (2, 12), (3, 13), (3, 14)],
    )
    connection.executemany(
        "INSERT INTO component_sequences VALUES (?, ?, ?, ?)",
        [
            (11, "P00001", "MKT AAA", "PROTEIN"),
            (12, "P00002", "GGG", "PROTEIN"),
            (13, "P00003", "VVV", "PROTEIN"),
            (14, "P00004", "LLL", "PROTEIN"),
        ],
    )
    connection.executemany(
        "INSERT INTO assays VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (101, "CHEMBL_A1", "GROUP1", 9, "B", 1, None),
            (102, "CHEMBL_A2", None, 9, "F", 1, None),
            (103, "CHEMBL_A3", None, 9, "B", 2, None),
            (104, "CHEMBL_A4", None, 9, "B", 1, 99),
            (105, "CHEMBL_A5", None, 7, "B", 1, None),
            (106, "CHEMBL_A6", None, 9, "B", 3, None),
        ],
    )
    connection.executemany(
        "INSERT INTO molecule_dictionary VALUES (?, ?)",
        [(201, "CHEMBL_RAW1"), (202, "CHEMBL_PARENT1"), (203, "CHEMBL_PARENT2")],
    )
    connection.executemany(
        "INSERT INTO molecule_hierarchy VALUES (?, ?)", [(201, 202), (202, 202), (203, 203)]
    )
    connection.executemany(
        "INSERT INTO compound_structures VALUES (?, ?)",
        [(201, "CCO.Cl"), (202, "CCO"), (203, "CCN")],
    )
    connection.executemany(
        "INSERT INTO activities VALUES (?, ?, ?, ?, ?, ?)",
        [
            (101, 201, 7.0, "Ki", "=", None),
            (101, 202, 9.0, "Ki", "=", "Manually validated"),
            (101, 203, 5.0, "IC50", "=", None),
            (101, 203, 4.0, "IC50", ">", None),
            (101, 203, 3.0, "IC50", "=", "Potential author error"),
            (102, 203, 8.0, "IC50", "=", None),
            (103, 203, 8.0, "IC50", "=", None),
            (104, 203, 8.0, "IC50", "=", None),
            (105, 203, 8.0, "IC50", "=", None),
            (106, 203, 8.0, "IC50", "=", None),
        ],
    )
    connection.commit()
    connection.close()


def test_query_defines_single_protein_binding_quality_subset():
    query, parameters = build_activity_query(
        exact_only=True,
        include_variants=False,
        include_questionable=False,
        min_confidence_score=8,
        has_variant_id=True,
    )
    assert "a.assay_type = 'B'" in query
    assert "td.target_type = 'SINGLE PROTEIN'" in query
    assert "td.species_group_flag = 0" in query
    assert "HAVING COUNT(*) = 1" in query
    assert "act.pchembl_value IS NOT NULL" in query
    assert "act.standard_relation = '='" in query
    assert "a.variant_id IS NULL" in query
    assert parameters == [8]


def test_extract_and_curate_uses_parent_smiles_and_median(tmp_path):
    sqlite_path = tmp_path / "chembl.db"
    _create_toy_chembl(sqlite_path)
    raw = extract_activity_rows(sqlite_path, chunksize=2)
    assert len(raw) == 3
    assert set(raw["assay_id"]) == {"CHEMBL_A1"}
    assert set(raw["smiles"]) == {"CCO", "CCN"}

    curated, stats = curate_activity_rows(
        raw,
        selfies_encoder=lambda smiles: f"SELFIES:{smiles}",
    )
    assert len(curated) == 2
    ethanol = curated.loc[curated["compound_id"].eq("CHEMBL_PARENT1")].iloc[0]
    assert ethanol.pchembl_value == 8.0
    assert ethanol.measurement_count == 2
    assert ethanol.target_id == "P00001"
    assert ethanol.compound_selfies == "SELFIES:CCO"
    assert ethanol.binary_label == 1
    assert stats["measurements_collapsed"] == 1


def test_cluster_split_is_deterministic_and_leakage_free(tmp_path):
    rows = []
    for index in range(30):
        sequence = "M" + ("A" * (index + 2)) + "G"
        rows.append(
            {
                "source": "ChEMBL",
                "assay_id": f"A{index}",
                "target_id": f"P{index}",
                "target_chembl_id": f"T{index}",
                "protein_id": f"P{index}",
                "protein_accession": f"P{index}",
                "protein_sequence": sequence,
                "compound_id": f"C{index}",
                "molecule_chembl_id": f"C{index}",
                "smiles": "CCO",
                "pchembl_value": 5.0 + (index % 3),
                "protein_length": len(sequence),
                "compound_selfies": "[C][C][O]",
                "assay_group_id": f"ChEMBL:A{index}:T{index}",
                "binary_label": int(index % 3 != 0),
                "activity_type": "Ki",
                "measurement_count": 1,
                "confidence_score": 9,
                "depositor_assay_group": "",
                "assay_type": "B",
                "target_type": "SINGLE PROTEIN",
                "standard_relation": "=",
                "chembl_release": 37,
            }
        )
    frame = pd.DataFrame(rows)
    config = SplitConfig(
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
        min_seq_id=0.5,
        cluster_mode="exact",
    )
    cluster_stats = pd.DataFrame(
        {
            "protein_cluster_50": [f"cluster_{index}" for index in range(30)],
            "row_count": [1] * 30,
            "pos_count": [int(index % 3 != 0) for index in range(30)],
        }
    )
    cluster_stats["neg_count"] = cluster_stats["row_count"] - cluster_stats["pos_count"]
    assignment_a = assign_cluster_splits_balanced(cluster_stats, config, swap_trials=1_000)
    assignment_b = assign_cluster_splits_balanced(cluster_stats, config, swap_trials=1_000)
    assert assignment_a["split"].tolist() == assignment_b["split"].tolist()
    assert assignment_a.groupby("split")["row_count"].sum().to_dict() == {
        "test": 3,
        "train": 24,
        "val": 3,
    }

    split_frame, artifacts = build_cluster_disjoint_splits(frame, tmp_path / "out", config=config)
    assert set(split_frame["split"]) == {"train", "val", "test"}
    assert split_frame.groupby("protein_sequence")["split"].nunique().max() == 1
    assert all(value == 0 for value in artifacts["leakage_checks"].values())
    assert validate_dataset(split_frame)["binary_label_mismatches"] == 0
    assert all((tmp_path / "out" / f"{split}.parquet").exists() for split in ("train", "val", "test"))


def test_cluster_split_preserves_fixed_assignments():
    config = SplitConfig(val_ratio=0.1, test_ratio=0.1, random_seed=42)
    cluster_stats = pd.DataFrame(
        {
            "protein_cluster_50": [f"cluster_{index}" for index in range(30)],
            "row_count": [10 + index for index in range(30)],
            "pos_count": [5 + index // 2 for index in range(30)],
        }
    )
    cluster_stats["neg_count"] = (
        cluster_stats["row_count"] - cluster_stats["pos_count"]
    )
    fixed = {
        "cluster_29": "train",
        "cluster_28": "val",
        "cluster_27": "test",
    }

    assignment = assign_cluster_splits_balanced(
        cluster_stats,
        config,
        fixed_assignments=fixed,
        swap_trials=2_000,
    )

    assigned = assignment.set_index("protein_cluster_50")
    assert assigned.loc["cluster_29", "split"] == "train"
    assert assigned.loc["cluster_28", "split"] == "val"
    assert assigned.loc["cluster_27", "split"] == "test"
    assert assigned.loc["cluster_29", "fixed_split"] == "train"
    assert assigned.loc["cluster_28", "fixed_split"] == "val"
    assert assigned.loc["cluster_27", "fixed_split"] == "test"


def test_activity_balanced_stats_anchor_three_largest_potency_clusters():
    rows = []
    cluster_sizes = {
        "potency_large": 12,
        "potency_medium": 9,
        "potency_small": 6,
        "ki_a": 5,
        "ki_b": 5,
        "ki_c": 5,
    }
    for cluster_id, size in cluster_sizes.items():
        activity_type = "Potency" if cluster_id.startswith("potency") else "Ki"
        for index in range(size):
            rows.append(
                {
                    "protein_cluster_50": cluster_id,
                    "protein_sequence": f"SEQ_{cluster_id}",
                    "protein_length": 100 if index else 2_000,
                    "target_chembl_id": f"TARGET_{cluster_id}",
                    "assay_group_id": f"ASSAY_{cluster_id}",
                    "binary_label": int(index % 2 == 0),
                    "pchembl_value": 5.0 + index / 10.0,
                    "activity_type": activity_type,
                }
            )
    frame = pd.DataFrame(rows)
    config = SplitConfig(val_ratio=0.1, test_ratio=0.1, random_seed=42)

    stats, metric_weights, fixed, metadata = build_activity_balanced_cluster_stats(
        frame,
        config,
    )

    assert fixed == {
        "potency_large": "train",
        "potency_medium": "val",
        "potency_small": "test",
    }
    assert "activity_rows__Potency" in stats.columns
    assert "activity_assays__Potency" in metric_weights
    assert "activity_rows__Potency" not in metric_weights
    assert metric_weights["activity_rows__Ki"] == 40.0
    assert metric_weights["reward_protein_rows__Ki"] == 40.0
    assert "reward_protein_rows__Potency" in stats.columns
    assert "reward_protein_rows__Potency" not in metric_weights
    assert metadata["anchor_activity_type"] == "Potency"
    assert metadata["reward_protein_max_residues"] == 1022
    assert [item["activity_rows"] for item in metadata["anchor_clusters"]] == [
        12,
        9,
        6,
    ]
