from reward_model.model import RewardModelConfig


def test_reward_model_config_round_trip(tmp_path):
    config = RewardModelConfig(
        protein_model_name_or_path="protein/local",
        molecule_model_name_or_path="molecule/local",
        fusion_hidden_dim=256,
        fusion_num_heads=8,
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
