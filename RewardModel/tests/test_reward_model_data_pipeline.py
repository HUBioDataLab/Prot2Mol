import json
import os
import sqlite3

from conftest import DummyTokenizer
from reward_model.data import RewardDataStore
from reward_model.data_processing import ChemblPreprocessConfig
from reward_model.data_processing.chembl import (
    preprocess_chembl_sqlite,
    assign_assay_based_splits,
    deduplicate_assay_rows,
    extract_chembl_activity_rows,
    tokenize_assay_rows,
)


def _create_test_sqlite(path):
    connection = sqlite3.connect(path)
    cursor = connection.cursor()

    cursor.execute(
        "CREATE TABLE activities ("
        "activity_id INTEGER PRIMARY KEY, "
        "assay_id INTEGER, "
        "molregno INTEGER, "
        "pchembl_value REAL, "
        "standard_relation TEXT, "
        "standard_type TEXT, "
        "data_validity_comment TEXT)"
    )
    cursor.execute(
        "CREATE TABLE assays ("
        "assay_id INTEGER PRIMARY KEY, "
        "chembl_id TEXT, "
        "tid INTEGER, "
        "assay_type TEXT, "
        "confidence_score INTEGER, "
        "variant_id INTEGER)"
    )
    cursor.execute(
        "CREATE TABLE target_dictionary ("
        "tid INTEGER PRIMARY KEY, "
        "chembl_id TEXT, "
        "pref_name TEXT, "
        "organism TEXT, "
        "target_type TEXT)"
    )
    cursor.execute("CREATE TABLE target_components (tid INTEGER, component_id INTEGER)")
    cursor.execute(
        "CREATE TABLE component_sequences ("
        "component_id INTEGER PRIMARY KEY, "
        "accession TEXT, "
        "sequence TEXT)"
    )
    cursor.execute(
        "CREATE TABLE molecule_dictionary ("
        "molregno INTEGER PRIMARY KEY, "
        "chembl_id TEXT)"
    )
    cursor.execute(
        "CREATE TABLE molecule_hierarchy ("
        "molregno INTEGER PRIMARY KEY, "
        "parent_molregno INTEGER)"
    )
    cursor.execute(
        "CREATE TABLE compound_structures ("
        "molregno INTEGER PRIMARY KEY, "
        "canonical_smiles TEXT)"
    )

    cursor.executemany(
        "INSERT INTO target_dictionary VALUES (?, ?, ?, ?, ?)",
        [
            (1, "CHEMBL_T1", "Kinase One", "Homo sapiens", "SINGLE PROTEIN"),
            (2, "CHEMBL_T2", "Mouse Target", "Mus musculus", "SINGLE PROTEIN"),
            (3, "CHEMBL_T3", "GPCR Three", "Homo sapiens", "SINGLE PROTEIN"),
            (4, "CHEMBL_T4", "Family Target", "Homo sapiens", "PROTEIN FAMILY"),
        ],
    )
    cursor.executemany(
        "INSERT INTO target_components VALUES (?, ?)",
        [(1, 101), (2, 102), (3, 103), (4, 104)],
    )
    cursor.executemany(
        "INSERT INTO component_sequences VALUES (?, ?, ?)",
        [
            (101, "P11111", "MKTAA"),
            (102, "P22222", "GGGGG"),
            (103, "P33333", "TTTTT"),
            (104, "P44444", "CCCCC"),
        ],
    )
    cursor.executemany(
        "INSERT INTO assays VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, "CHEMBL_A1", 1, "B", 8, None),
            (2, "CHEMBL_A2", 1, "B", 9, None),
            (3, "CHEMBL_A3", 1, "B", 8, 100),
            (4, "CHEMBL_A4", 2, "B", 8, None),
            (5, "CHEMBL_A5", 1, "F", 8, None),
            (6, "CHEMBL_A6", 1, "B", 7, None),
            (7, "CHEMBL_A7", 3, "B", 8, None),
            (8, "CHEMBL_A8", 3, "B", 9, None),
            (9, "CHEMBL_A9", 3, "B", 8, None),
            (10, "CHEMBL_A10", 4, "B", 8, None),
        ],
    )
    cursor.executemany(
        "INSERT INTO molecule_dictionary VALUES (?, ?)",
        [
            (10, "CHEMBL_M10"),
            (11, "CHEMBL_M11"),
            (20, "CHEMBL_M20"),
            (30, "CHEMBL_M30"),
            (31, "CHEMBL_M31"),
            (40, "CHEMBL_M40"),
            (50, "CHEMBL_M50"),
            (60, "CHEMBL_M60"),
            (61, "CHEMBL_M61"),
            (70, "CHEMBL_M70"),
            (71, "CHEMBL_M71"),
            (72, "CHEMBL_M72"),
            (73, "CHEMBL_M73"),
            (74, "CHEMBL_M74"),
            (75, "CHEMBL_M75"),
            (76, "CHEMBL_M76"),
            (77, "CHEMBL_M77"),
            (78, "CHEMBL_M78"),
            (80, "CHEMBL_M80"),
        ],
    )
    cursor.executemany(
        "INSERT INTO molecule_hierarchy VALUES (?, ?)",
        [(11, 10)],
    )
    cursor.executemany(
        "INSERT INTO compound_structures VALUES (?, ?)",
        [
            (10, "CCO"),
            (11, "CCO"),
            (20, "NCC"),
            (30, "CCC"),
            (31, "CCN"),
            (40, "COC"),
            (50, "CNC"),
            (60, "OCC"),
            (61, "OCO"),
            (70, "CCCl"),
            (71, "CCBr"),
            (72, "CN"),
            (73, "NC"),
            (74, "CO"),
            (75, "OC"),
            (76, "NN"),
            (78, "NO"),
            (80, "OO"),
        ],
    )
    cursor.executemany(
        "INSERT INTO activities VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, 10, 7.0, "=", "IC50", None),
            (2, 1, 11, 8.0, "=", "IC50", None),
            (3, 1, 20, 5.0, "=", "Ki", "Manually validated"),
            (4, 2, 30, 6.5, "=", "Kd", None),
            (5, 2, 31, 7.2, "=", "Potency", None),
            (6, 3, 40, 8.0, "=", "IC50", None),
            (7, 4, 50, 8.0, "=", "IC50", None),
            (8, 5, 60, 8.0, "=", "IC50", None),
            (9, 6, 61, 8.0, "=", "IC50", None),
            (10, 7, 70, 6.1, "=", "IC50", None),
            (11, 7, 71, 5.9, "=", "IC50", None),
            (12, 8, 72, 7.1, "=", "XC50", None),
            (13, 8, 73, 6.8, "=", "AC50", None),
            (14, 9, 74, 7.4, "=", "ED50", None),
            (15, 9, 75, 6.2, ">", "IC50", None),
            (16, 9, 76, 6.6, "=", "IC50", "Outside typical range"),
            (17, 9, 77, 6.3, "=", "IC50", None),
            (18, 9, 78, 6.9, "=", "Ki", None),
            (19, 10, 80, 7.0, "=", "IC50", None),
        ],
    )

    connection.commit()
    connection.close()


def _patched_selfies(monkeypatch):
    monkeypatch.setattr(
        "reward_model.data_processing.chembl._try_encode_selfies",
        lambda smiles: None if smiles == "BAD" else f"SELFIES<{smiles}>",
    )


def test_extract_and_deduplicate_rows_apply_chembl_filters(tmp_path, monkeypatch):
    db_path = tmp_path / "chembl_36.sqlite"
    _create_test_sqlite(str(db_path))
    _patched_selfies(monkeypatch)

    config = ChemblPreprocessConfig(sqlite_path=str(db_path), output_dir=str(tmp_path / "artifacts"))
    filtered_rows, total_raw_rows = extract_chembl_activity_rows(str(db_path), config)
    curated_rows = deduplicate_assay_rows(filtered_rows, config)

    assert total_raw_rows == 19
    assert len(filtered_rows) == 11
    assert len(curated_rows) == 10

    collapsed = next(
        row
        for row in curated_rows
        if row["group_id"] == "CHEMBL_T1__CHEMBL_A1" and row["parent_molregno"] == 10
    )
    assert collapsed["pchembl_value"] == 7.5
    assert collapsed["n_raw_rows_collapsed"] == 2
    assert json.loads(collapsed["raw_molregnos_json"]) == [10, 11]
    assert collapsed["compound_selfies"] == "SELFIES<CCO>"

    kept_groups = {row["group_id"] for row in curated_rows}
    assert "CHEMBL_T1__CHEMBL_A3" not in kept_groups
    assert "CHEMBL_T2__CHEMBL_A4" not in kept_groups
    assert "CHEMBL_T1__CHEMBL_A5" not in kept_groups
    assert "CHEMBL_T4__CHEMBL_A10" not in kept_groups


def test_assign_assay_based_splits_are_deterministic_and_assay_scoped(tmp_path, monkeypatch):
    db_path = tmp_path / "chembl_36.sqlite"
    _create_test_sqlite(str(db_path))
    _patched_selfies(monkeypatch)

    config = ChemblPreprocessConfig(sqlite_path=str(db_path), output_dir=str(tmp_path / "artifacts"), split_seed=17)
    filtered_rows, _ = extract_chembl_activity_rows(str(db_path), config)
    curated_rows = deduplicate_assay_rows(filtered_rows, config)

    first_assignment = assign_assay_based_splits(curated_rows, config)
    second_assignment = assign_assay_based_splits(curated_rows, config)
    assert first_assignment == second_assignment

    sparse_rows = [row for row in first_assignment if row["target_chembl_id"] == "CHEMBL_T1"]
    assert sparse_rows
    assert {row["split"] for row in sparse_rows} == {"train"}
    assert {row["is_sparse_train_only"] for row in sparse_rows} == {True}

    target_three_rows = [row for row in first_assignment if row["target_chembl_id"] == "CHEMBL_T3"]
    assert {row["split"] for row in target_three_rows} == {"train", "valid", "test"}

    assay_to_split = {}
    for row in first_assignment:
        key = (row["target_chembl_id"], row["assay_chembl_id"])
        assay_to_split.setdefault(key, set()).add(row["split"])
    assert all(len(splits) == 1 for splits in assay_to_split.values())


def test_preprocess_pipeline_writes_artifacts_and_runtime_grouping(tmp_path, monkeypatch):
    db_path = tmp_path / "chembl_36.sqlite"
    _create_test_sqlite(str(db_path))
    _patched_selfies(monkeypatch)

    output_dir = tmp_path / "phase3_output"
    config = ChemblPreprocessConfig(
        sqlite_path=str(db_path),
        output_dir=str(output_dir),
        split_seed=21,
        protein_max_length=6,
        molecule_max_length=7,
        write_parquet=False,
    )

    artifacts = preprocess_chembl_sqlite(
        config=config,
        protein_tokenizer=DummyTokenizer(),
        molecule_tokenizer=DummyTokenizer(),
    )

    assert os.path.exists(artifacts.curated_csv_path)
    assert os.path.exists(artifacts.tokenized_jsonl_path)
    assert os.path.exists(artifacts.metadata_path)
    assert os.path.exists(artifacts.provenance_path)
    assert artifacts.curated_parquet_path is None

    with open(artifacts.metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["total_raw_rows"] == 19
    assert metadata["rows_after_deduplication"] == 10
    assert metadata["split_counts"]["train"]["rows"] >= 1

    store = RewardDataStore.from_output_dir(str(output_dir))
    assert len(store.curated_rows) == 10
    assert len(store.tokenized_rows) == 10

    one_tokenized_row = store.tokenized_rows[0]
    assert "protein_input_ids" in one_tokenized_row
    assert "molecule_input_ids" in one_tokenized_row

    curated_groups = store.curated_groups()
    assert "CHEMBL_T1__CHEMBL_A1" in curated_groups
    assert len(curated_groups["CHEMBL_T1__CHEMBL_A1"]) == 2

    train_groups = store.tokenized_groups(split="train")
    assert all(all(row["split"] == "train" for row in rows) for rows in train_groups.values())


def test_tokenize_assay_rows_preserve_group_metadata():
    rows = [
        {
            "target_chembl_id": "CHEMBL_T1",
            "target_pref_name": "Kinase One",
            "protein_accession": "P11111",
            "protein_sequence": "MKTAA",
            "assay_chembl_id": "CHEMBL_A1",
            "assay_id": 1,
            "confidence_score": 8,
            "target_organism": "Homo sapiens",
            "parent_molregno": 10,
            "molecule_chembl_id": "CHEMBL_M10",
            "canonical_smiles": "CCO",
            "compound_selfies": "[C][C][O]",
            "pchembl_value": 7.5,
            "activity_label": 1,
            "n_raw_rows_collapsed": 2,
            "group_id": "CHEMBL_T1__CHEMBL_A1",
            "split": "train",
            "split_seed": 42,
            "split_policy": "demo",
            "is_sparse_train_only": True,
            "raw_activity_ids_json": "[1, 2]",
            "raw_molregnos_json": "[10, 11]",
            "raw_molecule_chembl_ids_json": "[\"CHEMBL_M10\", \"CHEMBL_M11\"]",
            "raw_standard_types_json": "[\"IC50\"]",
        }
    ]
    config = ChemblPreprocessConfig(output_dir="/tmp/unused", protein_max_length=6, molecule_max_length=7)
    tokenized_rows = tokenize_assay_rows(
        rows,
        protein_tokenizer=DummyTokenizer(),
        molecule_tokenizer=DummyTokenizer(),
        config=config,
    )

    assert len(tokenized_rows) == 1
    assert tokenized_rows[0]["group_id"] == "CHEMBL_T1__CHEMBL_A1"
    assert len(tokenized_rows[0]["protein_input_ids"]) == 6
    assert len(tokenized_rows[0]["molecule_input_ids"]) == 7
