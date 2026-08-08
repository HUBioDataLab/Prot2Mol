import math

import pytest

from prot2mol.chem.utils import metrics_calculation
from prot2mol.training.metrics import compute_conditional_generation_metrics


def test_generation_metrics_use_valid_molecule_denominators():
    metrics, details = metrics_calculation(
        predictions=["[C]", "[C]", "not-selfies"],
        references=["C"],
        train_data=["O"],
        return_details=True,
    )

    assert metrics["validity"] == pytest.approx(2 / 3)
    assert metrics["uniqueness"] == pytest.approx(1 / 2)
    assert metrics["novelty_train"] == 1.0
    assert metrics["novelty_eval"] == 0.0
    assert metrics["similarity_eval"] == 1.0
    assert len(details) == 3
    assert details.loc[2, "smiles"] == ""
    assert math.isnan(details.loc[2, "qed"])


def test_generation_metrics_are_zero_safe_for_no_predictions():
    metrics = metrics_calculation([], references=[], train_data=[])

    assert set(metrics) == {
        "validity",
        "uniqueness",
        "novelty_train",
        "novelty_eval",
        "intdiv",
        "similarity_train",
        "similarity_eval",
        "sa",
        "qed",
        "logp",
    }
    assert all(value == 0.0 for value in metrics.values())


def test_conditional_metrics_only_use_the_matching_protein_references():
    class Tokenizer:
        @staticmethod
        def batch_decode(token_ids, **kwargs):
            del kwargs
            return [row for row in token_ids]

    metrics = compute_conditional_generation_metrics(
        generated_token_ids_by_protein={
            "protein-a": ["[C]"],
            "protein-b": ["[C]"],
        },
        mol_tokenizer=Tokenizer(),
        reference_smiles_by_protein={
            "protein-a": ["C"],
            "protein-b": ["O"],
        },
    )

    assert metrics["gen_conditional_protein_count"] == 2
    assert metrics["gen_conditional_reference_recovery_macro"] == 0.5
