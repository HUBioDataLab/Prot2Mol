import torch

from conftest import DummyEncoder, DummyTokenizer
from reward_model.model import (
    LoadedEncoder,
    RewardModel,
    RewardModelConfig,
    load_encoder_bundle,
    load_reward_model,
    load_reward_model_config,
    save_reward_model,
)


def _dummy_bundles():
    protein_bundle = LoadedEncoder(
        name_or_path="protein/dummy",
        tokenizer=DummyTokenizer(),
        model=DummyEncoder(hidden_size=6),
        hidden_size=6,
    )
    molecule_bundle = LoadedEncoder(
        name_or_path="molecule/dummy",
        tokenizer=DummyTokenizer(),
        model=DummyEncoder(hidden_size=8),
        hidden_size=8,
    )
    return protein_bundle, molecule_bundle


def test_load_encoder_bundle_uses_full_name_and_infers_hidden_size(monkeypatch):
    captured = {}

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(name_or_path, **kwargs):
            captured["tokenizer"] = (name_or_path, kwargs)
            return DummyTokenizer()

    class _AutoModel:
        @staticmethod
        def from_pretrained(name_or_path, **kwargs):
            captured["model"] = (name_or_path, kwargs)
            return DummyEncoder(hidden_size=12)

    monkeypatch.setattr("reward_model.model.encoders.AutoTokenizer", _AutoTokenizer)
    monkeypatch.setattr("reward_model.model.encoders.AutoModel", _AutoModel)

    bundle = load_encoder_bundle(
        name_or_path="/abs/path/to/protein-model",
        tokenizer_name_or_path="/abs/path/to/protein-tokenizer",
        tokenizer_kwargs={"use_fast": False},
        model_kwargs={"trust_remote_code": False},
    )

    assert bundle.hidden_size == 12
    assert captured["tokenizer"] == (
        "/abs/path/to/protein-tokenizer",
        {"clean_up_tokenization_spaces": False, "use_fast": False},
    )
    assert captured["model"] == (
        "/abs/path/to/protein-model",
        {"add_pooling_layer": False, "trust_remote_code": False},
    )


def test_reward_model_save_and_load_round_trip(tmp_path):
    protein_bundle, molecule_bundle = _dummy_bundles()
    config = RewardModelConfig(
        protein_model_name_or_path="protein/dummy",
        molecule_model_name_or_path="molecule/dummy",
        fusion_hidden_dim=10,
        fusion_num_heads=2,
        dropout=0.0,
    )
    model = RewardModel(config=config, protein_bundle=protein_bundle, molecule_bundle=molecule_bundle)

    save_reward_model(model, str(tmp_path))
    loaded_config = load_reward_model_config(str(tmp_path))
    reloaded = load_reward_model(
        str(tmp_path),
        device=torch.device("cpu"),
        strict=True,
        protein_bundle=protein_bundle,
        molecule_bundle=molecule_bundle,
    )

    assert loaded_config.fusion_hidden_dim == 10
    assert reloaded.config.fusion_hidden_dim == 10
    assert reloaded.config.protein_hidden_size == 6

    original_state = model.state_dict()
    reloaded_state = reloaded.state_dict()
    for key in original_state:
        assert torch.equal(original_state[key], reloaded_state[key]), key
