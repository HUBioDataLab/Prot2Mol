import pandas as pd
import pytest
from datasets import load_from_disk

from data_processing.preprocess_dataset import (
    DatasetPreprocessor,
    PreprocessingConfig,
)


def _write_raw_splits(path):
    path.mkdir()
    for split, cluster, sequence in (
        ("train", "cluster_train", "MKT"),
        ("validation", "cluster_validation", "GGA"),
    ):
        pd.DataFrame(
            {
                "protein_sequence": [sequence, sequence],
                "compound_selfies": ["[C]", "[O]"],
                "smiles": ["C", "O"],
                "protein_cluster_50": [cluster, cluster],
                "pchembl_value": [7.0, 8.0],
            }
        ).to_parquet(path / f"{split}.parquet", index=False)


def _patch_tokenizers(monkeypatch):
    from conftest import DummyBatchTokenizer

    monkeypatch.setattr(
        "data_processing.preprocess_dataset.load_molgen_tokenizer",
        lambda **kwargs: DummyBatchTokenizer(),
    )
    monkeypatch.setattr(
        "data_processing.preprocess_dataset.get_protein_tokenizer",
        lambda *args, **kwargs: DummyBatchTokenizer(),
    )


def test_preprocessor_preserves_train_validation_and_right_padding(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    output = tmp_path / "tokenized"
    _write_raw_splits(raw)
    _patch_tokenizers(monkeypatch)

    manifest = DatasetPreprocessor(
        PreprocessingConfig(
            input_dir=raw,
            output_dir=output,
            max_mol_len=8,
            prot_max_length=8,
            num_proc=1,
            batch_size=2,
        )
    ).preprocess()
    dataset = load_from_disk(str(output))

    assert set(dataset) == {"train", "validation"}
    assert dataset["train"]["protein_cluster_50"] == ["cluster_train"] * 2
    assert dataset["validation"]["protein_cluster_50"] == ["cluster_validation"] * 2
    assert all(
        label == -100
        for row in dataset["train"]["labels"]
        for index, label in enumerate(row)
        if index >= next((i for i, value in enumerate(row) if value == -100), len(row))
    )
    assert "mol_input_ids" not in dataset["train"].column_names
    assert "mol_attention_mask" not in dataset["train"].column_names
    assert manifest["molecule_padding_side"] == "right"
    assert manifest["protein_model_id_resolved"] == "facebook/esm2_t33_650M_UR50D"
    assert manifest["splits"] == {"train": 2, "validation": 2}
    assert manifest["molecule_tokenizer_coverage"]["splits"]["train"] == {
        "sequences": 2,
        "active_tokens": 6,
        "unknown_tokens": 0,
        "unknown_sequences": 0,
        "max_encoded_length": 3,
        "max_token_id": 4,
    }


def test_preprocessor_requires_explicit_overwrite_and_rejects_test_split(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    output = tmp_path / "tokenized"
    _write_raw_splits(raw)
    output.mkdir()
    _patch_tokenizers(monkeypatch)

    config = PreprocessingConfig(input_dir=raw, output_dir=output, num_proc=1)
    with pytest.raises(FileExistsError, match="--overwrite"):
        DatasetPreprocessor(config).preprocess()

    output.rmdir()
    (raw / "test.parquet").write_bytes((raw / "validation.parquet").read_bytes())
    with pytest.raises(ValueError, match="test.parquet"):
        DatasetPreprocessor(config).preprocess()
