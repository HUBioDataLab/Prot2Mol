from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from datasets import Dataset

from experiments.paired_generalization.audit_data import group_descriptors
from experiments.paired_generalization.export_predictions import score_dataset, train_probe_indices


def test_composition_counts_ties_and_eligibility_from_complete_groups():
    frame = pd.DataFrame({
        "target_chembl_id": ["x"] * 6,
        "assay_id": ["a"] * 3 + ["b"] * 2 + ["c"],
        "pchembl_value": [5., 5., 6., 4., 8., 7.],
        "binary_label": [0, 0, 1, 0, 1, 1],
        "measurement_count": [1, 3, 1, 2, 1, 1],
    })
    groups = group_descriptors(frame).set_index("assay_id")
    assert groups.loc["a", "group_id"] == "x__a"
    assert groups.ranking_eligible.tolist() == [True, False, False]
    assert groups.all_pairs.tolist() == [3, 1, 0]
    assert groups.tied_label_pairs.tolist() == [1, 0, 0]
    assert groups.loc["a", "variance"] == pytest.approx(np.var([5., 5., 6.]))
    assert groups.loc["c", "variance"] == 0


def test_probe_is_reproducible_and_never_splits_an_assay():
    dataset = Dataset.from_dict({"group_id": ["a", "a", "b", "c", "c", "c", "d"]})
    indices = train_probe_indices(dataset, number=2)
    np.testing.assert_array_equal(indices, train_probe_indices(dataset, number=2))
    selected = set(indices)
    for group in set(dataset["group_id"]):
        members = {i for i, g in enumerate(dataset["group_id"]) if g == group}
        assert not members & selected or members <= selected


def test_export_keeps_observation_identity_and_representations_aligned(tmp_path):
    class Model(torch.nn.Module):
        protein_tokenizer = molecule_tokenizer = SimpleNamespace(pad_token_id=0)

        def forward(self, protein_input_ids, **kwargs):
            score = protein_input_ids[:, 0].float()
            embedding = torch.stack((score, score + 1), dim=1)
            return SimpleNamespace(ranking_score=score, activity_logits=-score,
                                   cosine_similarity=score / 10, normalized_protein_embedding=embedding,
                                   normalized_molecule_embedding=embedding + 10, score_scale=torch.tensor(13.))

    dataset = Dataset.from_dict({
        "protein_input_ids": [[i, 0] for i in range(1, 6)],
        "protein_attention_mask": [[1, 0]] * 5,
        "molecule_input_ids": [[1]] * 5, "molecule_attention_mask": [[1]] * 5,
        "group_id": ["x__a"] * 3 + ["x__b"] * 2,
        "target_chembl_id": ["x"] * 5, "compound_id": [str(i) for i in range(5)],
        "pchembl_value": [4., 5., 6., 7., 8.], "binary_label": [0, 0, 1, 1, 1],
    })
    result = score_dataset(Model(), dataset, tmp_path, batch_size=2, device="cpu")
    frame = pd.read_parquet(tmp_path / "predictions.parquet")
    assert result["rows"] == 5 and result["score_scale"] == 13
    assert frame.row_index.tolist() == list(range(5))
    assert frame.compound_id.tolist() == dataset["compound_id"]
    np.testing.assert_array_equal(frame.ranking_score, np.arange(1, 6))
    np.testing.assert_array_equal(np.load(tmp_path / "protein.npy")[:, 0], frame.ranking_score)
    np.testing.assert_array_equal(np.load(tmp_path / "molecule.npy")[:, 0], frame.ranking_score + 10)
