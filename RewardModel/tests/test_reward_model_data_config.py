from reward_model.data_processing import ChemblPreprocessConfig


def test_chembl_preprocess_config_round_trip(tmp_path):
    config = ChemblPreprocessConfig(
        sqlite_path="/data/chembl_36.sqlite",
        output_dir="/tmp/reward-data",
        split_seed=123,
        protein_max_length=512,
        molecule_max_length=128,
        tokenization_batch_size=64,
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
