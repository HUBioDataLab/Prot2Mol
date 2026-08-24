from types import SimpleNamespace

import pytest
import torch

from prot2mol.rewards.internal import InternalRewardModelActivityScorer


class FakeInternalRewardModel(torch.nn.Module):
    def __init__(self, *, representation="selfies"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(
            molecule_input_representation=representation,
        )
        self.calls = []

    def score_pairs(self, *, protein_sequences, molecule_sequences):
        self.calls.append((list(protein_sequences), list(molecule_sequences)))
        values = [0.1 * len(value) for value in molecule_sequences]
        return SimpleNamespace(activity_probability=torch.tensor(values))


def test_internal_reward_scorer_batches_selfies_and_freezes_model():
    model = FakeInternalRewardModel()
    scorer = InternalRewardModelActivityScorer(model, batch_size=2)

    probabilities = scorer(
        ["AAAA", "BBBB", "CCCC"],
        ["[C]", "[O][O]", "[N]"],
    )

    assert scorer.protein_representation == "sequence"
    assert scorer.molecule_representation == "selfies"
    assert probabilities.tolist() == pytest.approx([0.3, 0.6, 0.3])
    assert model.calls == [
        (["AAAA", "BBBB"], ["[C]", "[O][O]"]),
        (["CCCC"], ["[N]"]),
    ]
    assert scorer.last_diagnostics()["activity_probability"].tolist() == (
        pytest.approx([0.3, 0.6, 0.3])
    )
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_internal_reward_scorer_validates_contract():
    with pytest.raises(ValueError, match="SMILES or SELFIES"):
        InternalRewardModelActivityScorer(
            FakeInternalRewardModel(representation="inchi")
        )
    with pytest.raises(ValueError, match="batch_size"):
        InternalRewardModelActivityScorer(FakeInternalRewardModel(), batch_size=0)

    scorer = InternalRewardModelActivityScorer(FakeInternalRewardModel())
    with pytest.raises(ValueError, match="must align"):
        scorer(["AAAA"], [])
    assert scorer([], []).numel() == 0


@pytest.mark.parametrize("probability", [float("nan"), -0.1, 1.1])
def test_internal_reward_scorer_rejects_invalid_probabilities(probability):
    class InvalidModel(FakeInternalRewardModel):
        def score_pairs(self, *, protein_sequences, molecule_sequences):
            return SimpleNamespace(
                activity_probability=torch.full(
                    (len(protein_sequences),),
                    probability,
                )
            )

    scorer = InternalRewardModelActivityScorer(InvalidModel())
    with pytest.raises(ValueError, match=r"finite probabilities in \[0, 1\]"):
        scorer(["AAAA"], ["[C]"])
