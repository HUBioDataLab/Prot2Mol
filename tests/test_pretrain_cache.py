import json

import pytest

pytest.importorskip("datasets")
pytest.importorskip("rdkit")
pytest.importorskip("selfies")
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("wandb")

from prot2mol.training.pretrain import TrainingScript


def _cache_probe():
    script = TrainingScript.__new__(TrainingScript)
    script.selfies_path = "/data/source.csv"
    script.training_config = {
        "eval_split": "random",
        "eval_split_ratio": 0.1,
        "split_seed": 42,
        "training_stage": "pchembl_only",
        "pchembl_huber_delta": 1.0,
    }
    script.model_config = {"train_pchembl_head": True}
    return script


def test_prepared_split_cache_key_changes_when_processed_dataset_changes(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    dataset_info = processed_dir / "dataset_info.json"
    dataset_info.write_text('{"rows": 10}\n', encoding="utf-8")

    script = _cache_probe()
    first_payload = script._prepared_split_cache_payload(str(processed_dir))
    first_cache_dir = script._prepared_split_cache_dir(str(processed_dir))

    dataset_info.write_text('{"rows": 11}\n', encoding="utf-8")

    second_payload = script._prepared_split_cache_payload(str(processed_dir))
    second_cache_dir = script._prepared_split_cache_dir(str(processed_dir))

    assert first_payload["processed_dataset_fingerprint"] != second_payload["processed_dataset_fingerprint"]
    assert first_cache_dir != second_cache_dir


def test_prepared_split_cache_ready_rejects_mismatched_processed_fingerprint(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    dataset_info = processed_dir / "dataset_info.json"
    dataset_info.write_text('{"rows": 10}\n', encoding="utf-8")

    script = _cache_probe()
    old_payload = script._prepared_split_cache_payload(str(processed_dir))
    cache_dir = tmp_path / "prepared"
    (cache_dir / "train").mkdir(parents=True)
    (cache_dir / "eval").mkdir()
    (cache_dir / "metadata.json").write_text(
        json.dumps({"payload": old_payload}),
        encoding="utf-8",
    )

    dataset_info.write_text('{"rows": 11}\n', encoding="utf-8")

    assert not script._prepared_split_cache_ready(
        str(cache_dir),
        processed_data_path=str(processed_dir),
    )
