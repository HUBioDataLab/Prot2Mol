import math

import pytest

from reward_model.model import DEFAULT_RANKING_AFFINITY_MARGIN, RewardModelConfig


def test_reward_model_config_round_trip(tmp_path):
    config = RewardModelConfig(
        protein_model_name_or_path="protein/local",
        molecule_model_name_or_path="molecule/local",
        protein_hidden_dropout_prob=0.15,
        protein_attention_probs_dropout_prob=0.15,
        molecule_input_representation="smiles",
        molecule_trust_remote_code=True,
        molecule_deterministic_eval=True,
        molecule_hidden_dropout_prob=0.0,
        molecule_attention_probs_dropout_prob=0.0,
        fusion_hidden_dim=256,
        fusion_num_heads=8,
        fusion_residual=True,
        pooling_type="mean_all_tok",
        bce_pos_weight=3.0,
    )

    config_path = tmp_path / "config.json"
    config.save_json(str(config_path))
    loaded = RewardModelConfig.load_json(str(config_path))

    assert loaded == config


def test_reward_model_config_rejects_invalid_head_shape():
    try:
        RewardModelConfig(fusion_hidden_dim=250, fusion_num_heads=8)
        assert False, "Expected invalid hidden/head combination to raise"
    except ValueError as exc:
        assert "divisible" in str(exc)


def test_reward_model_config_loads_legacy_payload_with_deduplication_defaults():
    config = RewardModelConfig.from_dict(
        {
            "protein_model_name_or_path": "protein/legacy",
            "molecule_model_name_or_path": "molecule/legacy",
        }
    )

    assert config.deduplicate_protein_inputs is True
    assert config.deduplicate_molecule_inputs is True
    assert config.fusion_attention_backend == "manual"
    assert config.fusion_residual is False
    assert config.pair_scoring_mode == "mlp"
    assert config.ranking_affinity_margin == pytest.approx(math.log10(3.0))
    assert config.ranking_affinity_margin == pytest.approx(
        DEFAULT_RANKING_AFFINITY_MARGIN
    )


def test_reward_model_config_maps_legacy_pair_weight_to_listwise_ranking():
    config = RewardModelConfig.from_dict(
        {
            "protein_model_name_or_path": "protein/legacy",
            "molecule_model_name_or_path": "molecule/legacy",
            "pair_loss_weight": 0.25,
        }
    )

    assert config.ranking_loss_weight == pytest.approx(0.25)
    assert "pair_loss_weight" not in config.to_dict()


def test_reward_model_config_rejects_unknown_fusion_attention_backend():
    try:
        RewardModelConfig(fusion_attention_backend="unknown")
    except ValueError as exc:
        assert "fusion_attention_backend" in str(exc)
    else:
        raise AssertionError("Expected an invalid fusion backend to be rejected")


def test_reward_model_config_rejects_non_boolean_fusion_residual():
    with pytest.raises(ValueError, match="fusion_residual must be a boolean"):
        RewardModelConfig(fusion_residual="true")


def test_reward_model_config_validates_scaled_cosine_settings():
    assert RewardModelConfig(pair_scoring_mode="cosine").pair_scoring_mode == "cosine"
    with pytest.raises(ValueError, match="pair_scoring_mode"):
        RewardModelConfig(pair_scoring_mode="unknown")
    with pytest.raises(ValueError, match="cosine_scale_init"):
        RewardModelConfig(cosine_scale_init=0.0)
    with pytest.raises(ValueError, match="cosine_scale_init must be <="):
        RewardModelConfig(cosine_scale_init=101.0, cosine_scale_max=100.0)
    with pytest.raises(ValueError, match="cosine_classification_bias_init"):
        RewardModelConfig(cosine_classification_bias_init=float("nan"))


def test_reward_model_config_validates_projection_type():
    assert RewardModelConfig(projection_type="nonlinear").projection_type == "nonlinear"
    with pytest.raises(ValueError, match="projection_type"):
        RewardModelConfig(projection_type="unknown")


def test_reward_model_config_validates_ligunity_contrastive_settings():
    config = RewardModelConfig(
        pair_scoring_mode="cosine",
        contrastive_loss_weight=0.5,
        contrastive_active_threshold=5.0,
    )
    assert config.contrastive_loss_weight == pytest.approx(0.5)
    assert config.contrastive_active_threshold == pytest.approx(5.0)
    assert config.contrastive_strict_active_only is False

    with pytest.raises(ValueError, match="contrastive_loss_weight"):
        RewardModelConfig(contrastive_loss_weight=-0.1)
    with pytest.raises(ValueError, match="contrastive_active_threshold"):
        RewardModelConfig(contrastive_active_threshold=float("nan"))
    with pytest.raises(ValueError, match="contrastive_strict_active_only"):
        RewardModelConfig(contrastive_strict_active_only="true")
    with pytest.raises(ValueError, match="pair_scoring_mode='cosine'"):
        RewardModelConfig(
            pair_scoring_mode="mlp",
            contrastive_loss_weight=0.5,
        )
    with pytest.raises(ValueError, match="deduplicate_protein_inputs=true"):
        RewardModelConfig(
            pair_scoring_mode="cosine",
            contrastive_loss_weight=0.5,
            deduplicate_protein_inputs=False,
        )


def test_reward_model_config_validates_cosine_classification_mlp():
    config = RewardModelConfig(
        pair_scoring_mode="cosine",
        cosine_classification_mlp=True,
    )
    assert config.cosine_classification_mlp is True

    with pytest.raises(ValueError, match="cosine_classification_mlp"):
        RewardModelConfig(cosine_classification_mlp="true")
    with pytest.raises(ValueError, match="pair_scoring_mode='cosine'"):
        RewardModelConfig(
            pair_scoring_mode="scaled_cosine",
            cosine_classification_mlp=True,
        )


def test_reward_model_config_validates_molecule_encoder_settings():
    with pytest.raises(ValueError, match="molecule_input_representation"):
        RewardModelConfig(molecule_input_representation="inchi")
    with pytest.raises(ValueError, match="molecule_trust_remote_code"):
        RewardModelConfig(molecule_trust_remote_code="true")
    with pytest.raises(ValueError, match="molecule_deterministic_eval"):
        RewardModelConfig(molecule_deterministic_eval=1)
    with pytest.raises(ValueError, match="molecule_hidden_dropout_prob"):
        RewardModelConfig(molecule_hidden_dropout_prob=1.0)
    with pytest.raises(ValueError, match="molecule_attention_probs_dropout_prob"):
        RewardModelConfig(molecule_attention_probs_dropout_prob=float("nan"))
    with pytest.raises(ValueError, match="protein_hidden_dropout_prob"):
        RewardModelConfig(protein_hidden_dropout_prob=-0.1)
    with pytest.raises(ValueError, match="protein_attention_probs_dropout_prob"):
        RewardModelConfig(protein_attention_probs_dropout_prob=1.0)


@pytest.mark.parametrize(
    "affinity_margin",
    [-0.1, float("nan"), float("inf")],
)
def test_reward_model_config_rejects_invalid_affinity_margin(affinity_margin):
    with pytest.raises(ValueError, match="ranking_affinity_margin"):
        RewardModelConfig(ranking_affinity_margin=affinity_margin)
