from reward_model.data_processing import ChemblPreprocessConfig


def test_chembl_preprocess_config_round_trip(tmp_path):
    config = ChemblPreprocessConfig(
        sqlite_path="/data/chembl_36.sqlite",
        output_dir="/tmp/reward-data",
        activity_threshold=5.5,
        write_parquet=False,
    )

    config_path = tmp_path / "preprocess_config.json"
    config.save_json(str(config_path))
    loaded = ChemblPreprocessConfig.load_json(str(config_path))

    assert loaded == config


def test_chembl_preprocess_config_rejects_invalid_group_size():
    try:
        ChemblPreprocessConfig(output_dir="/tmp/reward-data", min_group_size=1)
        assert False, "Expected invalid min_group_size to raise"
    except ValueError as exc:
        assert "min_group_size" in str(exc)
