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


def test_reward_model_passes_molformer_loading_contract(monkeypatch):
    protein_bundle, molecule_bundle = _dummy_bundles()
    captured = []

    def _load_bundle(name_or_path, **kwargs):
        captured.append((name_or_path, kwargs))
        return (
            protein_bundle if name_or_path == "protein/dummy" else molecule_bundle
        )

    monkeypatch.setattr("reward_model.model.core.load_encoder_bundle", _load_bundle)

    model = RewardModel(
        RewardModelConfig(
            protein_model_name_or_path="protein/dummy",
            molecule_model_name_or_path="ibm/MoLFormer-XL-both-10pct",
            molecule_input_representation="smiles",
            molecule_trust_remote_code=True,
            molecule_deterministic_eval=True,
            fusion_hidden_dim=10,
            fusion_num_heads=2,
            pair_scoring_mode="cosine",
        )
    )

    assert model.molecule_encoder is molecule_bundle.model
    assert captured[1] == (
        "ibm/MoLFormer-XL-both-10pct",
        {
            "tokenizer_name_or_path": None,
            "tokenizer_kwargs": {"trust_remote_code": True},
            "model_kwargs": {
                "trust_remote_code": True,
                "deterministic_eval": True,
            },
        },
    )


def test_reward_model_passes_encoder_dropout_overrides(monkeypatch):
    protein_bundle, molecule_bundle = _dummy_bundles()
    captured = []

    def _load_bundle(name_or_path, **kwargs):
        captured.append((name_or_path, kwargs))
        return protein_bundle if name_or_path == "protein/dummy" else molecule_bundle

    monkeypatch.setattr("reward_model.model.core.load_encoder_bundle", _load_bundle)

    RewardModel(
        RewardModelConfig(
            protein_model_name_or_path="protein/dummy",
            molecule_model_name_or_path="HUBioDataLab/SELFormer",
            protein_hidden_dropout_prob=0.15,
            protein_attention_probs_dropout_prob=0.15,
            molecule_hidden_dropout_prob=0.15,
            molecule_attention_probs_dropout_prob=0.15,
            fusion_hidden_dim=10,
            fusion_num_heads=2,
            pair_scoring_mode="cosine",
        )
    )

    assert captured[0][1]["model_kwargs"] == {
        "hidden_dropout_prob": 0.15,
        "attention_probs_dropout_prob": 0.15,
    }
    assert captured[1][1]["model_kwargs"] == {
        "trust_remote_code": False,
        "hidden_dropout_prob": 0.15,
        "attention_probs_dropout_prob": 0.15,
    }


def test_reward_model_save_and_load_round_trip(tmp_path):
    protein_bundle, molecule_bundle = _dummy_bundles()
    config = RewardModelConfig(
        protein_model_name_or_path="protein/dummy",
        molecule_model_name_or_path="molecule/dummy",
        fusion_hidden_dim=10,
        fusion_num_heads=2,
        fusion_residual=True,
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
    assert loaded_config.fusion_residual is True
    assert reloaded.config.fusion_hidden_dim == 10
    assert reloaded.config.fusion_residual is True
    assert reloaded.config.protein_hidden_size == 6

    original_state = model.state_dict()
    reloaded_state = reloaded.state_dict()
    for key in original_state:
        assert torch.equal(original_state[key], reloaded_state[key]), key


def test_scaled_cosine_reward_model_save_and_load_round_trip(tmp_path):
    protein_bundle, molecule_bundle = _dummy_bundles()
    config = RewardModelConfig(
        protein_model_name_or_path="protein/dummy",
        molecule_model_name_or_path="molecule/dummy",
        fusion_hidden_dim=10,
        fusion_num_heads=2,
        fusion_residual=True,
        pair_scoring_mode="scaled_cosine",
        cosine_scale_init=13.0,
        cosine_classification_bias_init=-0.25,
        dropout=0.0,
    )
    model = RewardModel(
        config=config,
        protein_bundle=protein_bundle,
        molecule_bundle=molecule_bundle,
    )
    with torch.no_grad():
        model.logit_scale.add_(0.1)
        model.classification_logit_bias.add_(0.2)

    save_reward_model(model, str(tmp_path))
    reloaded = load_reward_model(
        str(tmp_path),
        device=torch.device("cpu"),
        strict=True,
        protein_bundle=protein_bundle,
        molecule_bundle=molecule_bundle,
    )

    assert reloaded.config.pair_scoring_mode == "scaled_cosine"
    assert reloaded.ranking_head is None
    assert reloaded.classification_head is None
    assert torch.equal(model.logit_scale, reloaded.logit_scale)
    assert torch.equal(
        model.classification_logit_bias,
        reloaded.classification_logit_bias,
    )


def test_simple_cosine_reward_model_save_and_load_round_trip(tmp_path):
    protein_bundle, molecule_bundle = _dummy_bundles()
    config = RewardModelConfig(
        protein_model_name_or_path="protein/dummy",
        molecule_model_name_or_path="molecule/dummy",
        fusion_hidden_dim=10,
        fusion_num_heads=2,
        projection_type="nonlinear",
        pair_scoring_mode="cosine",
        classification_loss_weight=0.0,
    )
    model = RewardModel(
        config=config,
        protein_bundle=protein_bundle,
        molecule_bundle=molecule_bundle,
    )

    save_reward_model(model, str(tmp_path))
    reloaded = load_reward_model(
        str(tmp_path),
        device=torch.device("cpu"),
        strict=True,
        protein_bundle=protein_bundle,
        molecule_bundle=molecule_bundle,
    )

    assert reloaded.config.pair_scoring_mode == "cosine"
    assert reloaded.config.projection_type == "nonlinear"
    assert reloaded.fusion is None
    assert reloaded.protein_norm is None
    assert reloaded.molecule_norm is None
    assert reloaded.logit_scale is None
    assert reloaded.classification_logit_bias is None
    for key, value in model.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[key]), key


def test_simple_cosine_mlp_classifier_save_and_load_round_trip(tmp_path):
    protein_bundle, molecule_bundle = _dummy_bundles()
    config = RewardModelConfig(
        protein_model_name_or_path="protein/dummy",
        molecule_model_name_or_path="molecule/dummy",
        fusion_hidden_dim=10,
        fusion_num_heads=2,
        projection_type="nonlinear",
        pair_scoring_mode="cosine",
        cosine_classification_mlp=True,
        classification_loss_weight=0.5,
    )
    model = RewardModel(
        config=config,
        protein_bundle=protein_bundle,
        molecule_bundle=molecule_bundle,
    )

    save_reward_model(model, str(tmp_path))
    reloaded = load_reward_model(
        str(tmp_path),
        device=torch.device("cpu"),
        strict=True,
        protein_bundle=protein_bundle,
        molecule_bundle=molecule_bundle,
    )

    assert reloaded.config.cosine_classification_mlp is True
    assert reloaded.classification_head is not None
    for key, value in model.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[key]), key


def test_fusion_contrastive_reward_model_save_and_load_round_trip(tmp_path):
    protein_bundle, molecule_bundle = _dummy_bundles()
    config = RewardModelConfig(
        protein_model_name_or_path="protein/dummy",
        molecule_model_name_or_path="molecule/dummy",
        fusion_hidden_dim=10,
        fusion_num_heads=2,
        fusion_attention_backend="sdpa",
        fusion_residual=True,
        projection_type="nonlinear",
        pair_scoring_mode="fusion_contrastive",
        ranking_loss_weight=0.0,
        contrastive_loss_weight=0.5,
        classification_loss_weight=0.5,
    )
    model = RewardModel(
        config=config,
        protein_bundle=protein_bundle,
        molecule_bundle=molecule_bundle,
    )

    save_reward_model(model, str(tmp_path))
    reloaded = load_reward_model(
        str(tmp_path),
        device=torch.device("cpu"),
        strict=True,
        protein_bundle=protein_bundle,
        molecule_bundle=molecule_bundle,
    )

    assert reloaded.config.pair_scoring_mode == "fusion_contrastive"
    assert reloaded.fusion is not None
    assert reloaded.classification_head.fc1.out_features == 128
    assert reloaded.contrastive_protein_projection is not None
    assert reloaded.contrastive_molecule_projection is not None
    assert reloaded.molecule_bias_head is None
    assert reloaded.protein_bias_head is None
    for key, value in model.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[key]), key


def test_weight_only_warm_start_unfreezes_encoders_with_fresh_optimizer(tmp_path):
    protein_bundle, molecule_bundle = _dummy_bundles()
    frozen_config = RewardModelConfig(
        protein_model_name_or_path="protein/dummy",
        molecule_model_name_or_path="molecule/dummy",
        fusion_hidden_dim=10,
        fusion_num_heads=2,
        fusion_residual=True,
        pair_scoring_mode="scaled_cosine",
        freeze_protein_encoder=True,
        freeze_molecule_encoder=True,
        dropout=0.0,
    )
    frozen_model = RewardModel(
        config=frozen_config,
        protein_bundle=protein_bundle,
        molecule_bundle=molecule_bundle,
    )
    with torch.no_grad():
        frozen_model.protein_projection.weight.fill_(0.125)
        frozen_model.molecule_projection.weight.fill_(-0.25)
        frozen_model.logit_scale.fill_(1.75)
    expected_state = {
        key: value.detach().clone() for key, value in frozen_model.state_dict().items()
    }
    save_reward_model(frozen_model, str(tmp_path))

    reloaded = load_reward_model(
        str(tmp_path),
        strict=True,
        config_overrides={
            "freeze_protein_encoder": False,
            "freeze_molecule_encoder": False,
        },
        protein_bundle=protein_bundle,
        molecule_bundle=molecule_bundle,
    )

    assert reloaded.config.freeze_protein_encoder is False
    assert reloaded.config.freeze_molecule_encoder is False
    assert all(parameter.requires_grad for parameter in reloaded.protein_encoder.parameters())
    assert all(parameter.requires_grad for parameter in reloaded.molecule_encoder.parameters())
    for key, expected in expected_state.items():
        assert torch.equal(reloaded.state_dict()[key], expected), key

    optimizer = torch.optim.AdamW(
        parameter for parameter in reloaded.parameters() if parameter.requires_grad
    )
    assert optimizer.state == {}
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert {
        id(parameter) for parameter in reloaded.protein_encoder.parameters()
    }.issubset(optimized_ids)
    assert {
        id(parameter) for parameter in reloaded.molecule_encoder.parameters()
    }.issubset(optimized_ids)

    protein_parameter = next(reloaded.protein_encoder.parameters())
    molecule_parameter = next(reloaded.molecule_encoder.parameters())
    protein_before = protein_parameter.detach().clone()
    molecule_before = molecule_parameter.detach().clone()
    outputs = reloaded(
        protein_input_ids=torch.tensor(
            [[1, 2, 0], [2, 3, 0], [3, 4, 0]], dtype=torch.long
        ),
        protein_attention_mask=torch.tensor(
            [[1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=torch.long
        ),
        molecule_input_ids=torch.tensor(
            [[4, 5, 0], [5, 6, 0], [6, 7, 0]], dtype=torch.long
        ),
        molecule_attention_mask=torch.tensor(
            [[1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=torch.long
        ),
        activity_labels=torch.tensor([0.0, 1.0, 0.0]),
        pchembl_values=torch.tensor([4.0, 7.0, 5.0]),
        ranking_group_ids=torch.tensor([0, 0, 0]),
    )
    assert outputs.loss is not None
    outputs.loss.backward()
    assert protein_parameter.grad is not None
    assert molecule_parameter.grad is not None
    assert torch.count_nonzero(protein_parameter.grad).item() > 0
    assert torch.count_nonzero(molecule_parameter.grad).item() > 0
    optimizer.step()
    assert not torch.equal(protein_parameter, protein_before)
    assert not torch.equal(molecule_parameter, molecule_before)
