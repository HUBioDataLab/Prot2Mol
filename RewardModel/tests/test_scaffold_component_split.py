import pandas as pd
import pytest

from reward_model.training.scaffold_component_split import (
    ACYCLIC_NODE,
    compute_scaffold_nodes,
    connected_components,
    split_scaffold_components,
)

BENZ = "c1ccccc1C"          # benzene scaffold
BENZ2 = "c1ccccc1CC"        # same benzene scaffold
PYR = "c1ccncc1C"           # pyridine scaffold
NAPH = "c1ccc2ccccc2c1C"    # naphthalene scaffold
ACYC = "CCCCO"              # no ring -> acyclic
ACYC2 = "CCCCCN"            # no ring -> acyclic


def _rows(triples):
    return pd.DataFrame(
        [(a, t, f"C{i}", s, y) for i, (a, t, s, y) in enumerate(triples)],
        columns=["assay_id", "target_chembl_id", "compound_id", "smiles", "binary_label"],
    )


def test_acyclic_molecules_share_one_pooled_node():
    mapping, counts = compute_scaffold_nodes([BENZ, ACYC, ACYC2], num_proc=1)
    assert mapping[ACYC] == ACYCLIC_NODE
    assert mapping[ACYC2] == ACYCLIC_NODE
    assert mapping[BENZ] != ACYCLIC_NODE
    assert counts["acyclic_molecules"] == 2
    assert counts["unparseable_molecules"] == 0


def test_stereoisomers_share_a_scaffold_since_chirality_is_ignored():
    mapping, _ = compute_scaffold_nodes(["C[C@H](N)c1ccccc1", "C[C@@H](N)c1ccccc1"], num_proc=1)
    assert mapping.nunique() == 1


def test_unparseable_molecules_get_private_nodes_and_are_counted():
    mapping, counts = compute_scaffold_nodes([BENZ, "not_a_molecule"], num_proc=1)
    assert counts["unparseable_molecules"] == 1
    assert mapping["not_a_molecule"] != mapping[BENZ]


def test_assays_sharing_a_scaffold_join_one_component():
    assays = pd.Series(["A", "A", "B", "C"])
    scaffolds = pd.Series(["X", "Y", "Y", "Z"])
    comp = connected_components(assays, scaffolds)
    assert comp["A"] == comp["B"]          # linked through scaffold Y
    assert comp["C"] != comp["A"]


def test_transitive_chaining_fuses_components():
    assays = pd.Series(["A", "A", "B", "B", "C"])
    scaffolds = pd.Series(["1", "2", "2", "3", "3"])
    comp = connected_components(assays, scaffolds)
    assert comp["A"] == comp["B"] == comp["C"]


def _toy():
    # one dominant component (target T1) plus isolated ones that can be held out
    rows = []
    for i in range(30):
        rows.append((f"big{i}", "T1", BENZ if i % 2 else BENZ2, i % 2))
    for i in range(6):
        rows.append((f"iso{i}", "T1", f"c1ccc2ccccc2c1{'C' * (i + 1)}", i % 2))
    for i in range(6):
        rows.append((f"pyr{i}", "T1", f"c1ccncc1{'C' * (i + 1)}", i % 2))
    return _rows(rows)


def test_split_is_assay_and_scaffold_disjoint():
    train, val, test, summary = split_scaffold_components(_toy(), seed=42, num_proc=1, balance_restarts=4)
    assert summary["leakage_checks"]["assay_integrity_holds"]
    assert summary["leakage_checks"]["scaffold_disjointness_holds"]
    assert summary["leakage_checks"]["assays_spanning_multiple_splits"] == 0
    assert summary["leakage_checks"]["scaffolds_spanning_multiple_splits"] == 0
    all_assays = set(train.assay_id) | set(val.assay_id) | set(test.assay_id)
    assert len(all_assays) == _toy().assay_id.nunique()
    assert set(train.assay_id) & set(val.assay_id) == set()
    assert set(val.assay_id) & set(test.assay_id) == set()


def test_no_rows_are_lost_or_duplicated():
    source = _toy()
    train, val, test, _ = split_scaffold_components(source, seed=42, num_proc=1, balance_restarts=4)
    assert len(train) + len(val) + len(test) == len(source)
    combined = pd.concat([train, val, test]).sort_values("compound_id").reset_index(drop=True)
    expected = source.sort_values("compound_id").reset_index(drop=True)
    assert set(combined.compound_id) == set(expected.compound_id)


def test_every_val_test_target_is_present_in_train():
    train, val, test, summary = split_scaffold_components(_toy(), seed=42, num_proc=1, balance_restarts=4)
    assert summary["leakage_checks"]["val_targets_absent_from_train"] == 0
    assert summary["leakage_checks"]["test_targets_absent_from_train"] == 0


def test_giant_component_goes_to_train():
    train, val, test, summary = split_scaffold_components(_toy(), seed=42, num_proc=1, balance_restarts=4)
    assert summary["giant_component_rows"] > 0
    assert len(train) >= len(val) and len(train) >= len(test)


def test_split_is_deterministic_for_a_given_seed():
    a = split_scaffold_components(_toy(), seed=42, num_proc=1, balance_restarts=8)
    b = split_scaffold_components(_toy(), seed=42, num_proc=1, balance_restarts=8)
    for x, y in zip(a[:3], b[:3]):
        pd.testing.assert_frame_equal(x, y)


def test_source_frame_is_not_mutated():
    source = _toy()
    before = source.copy()
    split_scaffold_components(source, seed=42, num_proc=1, balance_restarts=4)
    pd.testing.assert_frame_equal(source, before)


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="missing columns"):
        split_scaffold_components(pd.DataFrame({"assay_id": ["A"]}), num_proc=1)


def test_empty_frame_raises():
    empty = _rows([]).astype({"binary_label": int})
    with pytest.raises(ValueError, match="must not be empty"):
        split_scaffold_components(empty, num_proc=1)
