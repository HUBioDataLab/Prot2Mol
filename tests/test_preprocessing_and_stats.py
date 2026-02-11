import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from data_processing.preprocess_dataset import DatasetPreprocessor
from prot2mol.training.normalization_service import NormalizationService


def _build_csv(path):
    df = pd.DataFrame(
        {
            "Target_FASTA": ["MKT", "GGA", "TTT", "PPP"],
            "Compound_SELFIES": ["[C]", "[O]", "[N]", "[C][O]"],
            "pchembl_value_Median": [5.0, 6.0, 7.0, 8.0],
            "AID": ["A1", "A1", "A2", "A2"],
            "Target_ID": ["T1", "T1", "T2", "T2"],
        }
    )
    df.to_csv(path, index=False)


class _FakeHFDataset:
    def __init__(self):
        self._len = 4
        self.map_desc = []
        self.saved_path = None

    def __getitem__(self, key):
        if key != "train":
            raise KeyError(key)
        return [0] * self._len

    def map(self, fn, batched, num_proc, batch_size, desc):
        batch = {
            "Target_FASTA": ["MKT", "GGA"],
            "Compound_SELFIES": ["[C]", "[O]"],
            "pchembl_value_Median": [5.0, 7.0],
            "AID": ["A1", "A2"],
            "Target_ID": ["T1", "T2"],
        }
        fn(batch)
        self.map_desc.append(desc)
        return self

    def save_to_disk(self, path):
        self.saved_path = path


def test_preprocessor_stats_group_ids_and_tokenization(tmp_path, monkeypatch):
    csv_path = tmp_path / "toy.csv"
    _build_csv(csv_path)

    from conftest import DummyBatchTokenizer

    monkeypatch.setattr("data_processing.preprocess_dataset.load_molgen_tokenizer", lambda padding_side="left": DummyBatchTokenizer())
    monkeypatch.setattr("data_processing.preprocess_dataset.get_protein_tokenizer", lambda model_name: DummyBatchTokenizer())

    cfg = SimpleNamespace(
        selfies_path=str(csv_path),
        cache_dir=str(tmp_path / "cache"),
        eval_split="random",
        eval_split_ratio=0.0,
        split_seed=42,
        prot_emb_model="prot_t5",
        max_mol_len=8,
        prot_max_length=16,
        num_proc=1,
        batch_size=2,
    )
    prep = DatasetPreprocessor(cfg)

    stats_path = tmp_path / "cache" / "toy" / "pchembl_stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    assert stats["pchembl_threshold"] == 6.0
    assert "pchembl_mean" in stats and "pchembl_std" in stats

    assert prep.group_id_map is not None
    assert len(prep.group_id_map) == 2

    tok = prep.tokenize_mol_function(
        {
            "Compound_SELFIES": ["[C]", "[O]"],
            "pchembl_value_Median": [5.5, 7.5],
            "AID": ["A1", "A2"],
            "Target_ID": ["T1", "T2"],
        }
    )
    assert "group_id" in tok
    assert torch.equal(tok["train_lm"], torch.tensor([False, True]))


def test_preprocessor_full_preprocess_flow(tmp_path, monkeypatch):
    csv_path = tmp_path / "toy.csv"
    _build_csv(csv_path)

    from conftest import DummyBatchTokenizer

    fake_ds = _FakeHFDataset()
    monkeypatch.setattr("data_processing.preprocess_dataset.load_dataset", lambda *a, **k: fake_ds)
    monkeypatch.setattr("data_processing.preprocess_dataset.load_molgen_tokenizer", lambda padding_side="left": DummyBatchTokenizer())
    monkeypatch.setattr("data_processing.preprocess_dataset.get_protein_tokenizer", lambda model_name: DummyBatchTokenizer())

    cfg = SimpleNamespace(
        selfies_path=str(csv_path),
        cache_dir=str(tmp_path / "cache"),
        eval_split="random",
        eval_split_ratio=0.0,
        split_seed=42,
        prot_emb_model="prot_t5",
        max_mol_len=8,
        prot_max_length=16,
        num_proc=1,
        batch_size=2,
    )
    prep = DatasetPreprocessor(cfg)
    prep.preprocess()

    assert fake_ds.saved_path == str(tmp_path / "cache" / "toy")
    assert fake_ds.map_desc == ["Tokenizing protein sequences", "Tokenizing molecule sequences"]


def test_normalization_service_uses_cached_stats(tmp_path, monkeypatch):
    csv_path = tmp_path / "toy.csv"
    _build_csv(csv_path)

    cache_dir = tmp_path / "cache"
    cached_dir = cache_dir / "toy"
    cached_dir.mkdir(parents=True)
    (cached_dir / "pchembl_stats.json").write_text(
        json.dumps({"pchembl_mean": 9.0, "pchembl_std": 3.0, "pchembl_threshold": 6.5})
    )

    monkeypatch.setenv("DATASETS_CACHE_DIR", str(cache_dir))
    stats = NormalizationService(
        selfies_path=str(csv_path),
        train_pchembl_head=True,
        pchembl_huber_delta_raw=1.5,
    ).load_or_compute()

    assert stats.pchembl_mean == 9.0
    assert stats.pchembl_std == 3.0
    assert stats.pchembl_threshold == 6.5
    assert np.isclose(stats.pchembl_huber_delta_norm, 0.5)
