import math

import pytest

from reward_model.model import DEFAULT_RANKING_AFFINITY_MARGIN, RewardModelConfig


def test_reward_model_config_round_trip(tmp_path):
    config = RewardModelConfig(
        protein_model_name_or_path="protein/local",
        molecule_model_name_or_path="molecule/local",
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


@pytest.mark.parametrize(
    "affinity_margin",
    [-0.1, float("nan"), float("inf")],
)
def test_reward_model_config_rejects_invalid_affinity_margin(affinity_margin):
    with pytest.raises(ValueError, match="ranking_affinity_margin"):
        RewardModelConfig(ranking_affinity_margin=affinity_margin)
