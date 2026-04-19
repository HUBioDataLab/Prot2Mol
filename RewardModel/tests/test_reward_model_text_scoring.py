import torch

from conftest import DummyEncoder, DummyTokenizer
from reward_model.config import RewardModelConfig
from reward_model.encoders import LoadedEncoder
from reward_model.model import RewardModel
from reward_model.outputs import RewardModelOutput


def _build_model():
    protein_tokenizer = DummyTokenizer()
    molecule_tokenizer = DummyTokenizer()
    protein_bundle = LoadedEncoder(
        name_or_path="protein/dummy",
        tokenizer=protein_tokenizer,
        model=DummyEncoder(hidden_size=6),
        hidden_size=6,
    )
    molecule_bundle = LoadedEncoder(
        name_or_path="molecule/dummy",
        tokenizer=molecule_tokenizer,
        model=DummyEncoder(hidden_size=8),
        hidden_size=8,
    )
    config = RewardModelConfig(
        protein_model_name_or_path="protein/dummy",
        molecule_model_name_or_path="molecule/dummy",
        protein_max_length=5,
        molecule_max_length=7,
        fusion_hidden_dim=10,
        fusion_num_heads=2,
        dropout=0.0,
    )
    return RewardModel(config=config, protein_bundle=protein_bundle, molecule_bundle=molecule_bundle)


def test_score_pairs_tokenizes_raw_inputs_and_returns_structured_output():
    model = _build_model()

    outputs = model.score_pairs(
        protein_sequences=["MKT", "GGAA"],
        molecule_sequences=["[C][O]", "[N]"],
        activity_labels=torch.tensor([1.0, 0.0]),
        return_token_embeddings=True,
    )

    assert isinstance(outputs, RewardModelOutput)
    assert outputs.ranking_score.shape == (2,)
    assert outputs.activity_logits.shape == (2,)
    assert outputs.fused_protein_tokens is not None
    assert outputs.fused_molecule_tokens is not None

    assert model.protein_tokenizer.calls[0]["texts"] == ["MKT", "GGAA"]
    assert model.protein_tokenizer.calls[0]["max_length"] == 5
    assert model.molecule_tokenizer.calls[0]["texts"] == ["[C][O]", "[N]"]
    assert model.molecule_tokenizer.calls[0]["max_length"] == 7


def test_score_pairs_rejects_mismatched_raw_batch_lengths():
    model = _build_model()

    try:
        model.score_pairs(
            protein_sequences=["MKT"],
            molecule_sequences=["[C]", "[O]"],
        )
        assert False, "Expected mismatched raw batch lengths to raise"
    except ValueError as exc:
        assert "same length" in str(exc)
