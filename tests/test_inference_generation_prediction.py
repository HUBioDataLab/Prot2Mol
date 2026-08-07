import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import prot2mol.inference.predict_pchembl as predict_module
from prot2mol.inference.produce_molecules import MoleculeGenerator
from prot2mol.inference.predict_pchembl import PChemblPredictor, parse_args


def _generator_config(**overrides):
    base = dict(
        model_file="/tmp/model",
        prediction_model_file=None,
        prot_emb_model="prot_t5",
        selfies_path="/tmp/data.csv",
        prot_id="CHEMBL1",
        mode="generation",
        input_molecules=None,
        pchembl_mean=5.0,
        pchembl_std=2.0,
        num_samples=5,
        batch_size=2,
        prot_max_length=16,
        max_mol_len=8,
        output_file="/tmp/out.csv",
        attn_output=False,
        n_layer=1,
        n_head=2,
        n_emb=8,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _predictor_config(**overrides):
    base = dict(
        input_file="/tmp/in.csv",
        model_path="/tmp/model",
        output_file="/tmp/out.csv",
        models_base=None,
        data_path=None,
        chembl_uniprot_mapping_path=None,
        protein_targets_path=None,
        prot_emb_model="prot_t5",
        n_layer=1,
        n_head=2,
        n_emb=8,
        prot_max_length=16,
        max_mol_len=8,
        pchembl_tf_hidden_dim=768,
        pchembl_tf_num_heads=8,
        pchembl_tf_group_size=1,
        pchembl_tf_agg_mode="mean",
        pchembl_tf_dropout=0.1,
        batch_size=2,
        pchembl_mean=5.0,
        pchembl_std=2.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _PredictOnlyModel:
    def eval(self):
        return self

    def encode_protein(self, prot_input_ids, prot_attention_mask):
        batch = prot_input_ids.shape[0]
        seq = prot_input_ids.shape[1]
        return torch.ones(batch, seq, 4, dtype=torch.float32)

    def predict_pchembl_from_protein_embeddings(self, mol_input_ids, protein_embeddings, prot_attention_mask):
        return torch.arange(mol_input_ids.shape[0], dtype=torch.float32)


def test_generate_molecules_dataframe_shape(monkeypatch):
    cfg = _generator_config(num_samples=5)

    def _fake_load_components(self):
        from conftest import DummyBatchTokenizer

        self.mol_tokenizer = DummyBatchTokenizer()
        self.prot_tokenizer = DummyBatchTokenizer()
        self.generation_model = SimpleNamespace()
        self.prediction_model = SimpleNamespace()
        self.generation_config = SimpleNamespace(
            max_length=8,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            pad_token_id=1,
            bos_token_id=1,
            eos_token_id=2,
        )

    monkeypatch.setattr(MoleculeGenerator, "_load_components", _fake_load_components)
    monkeypatch.setattr(MoleculeGenerator, "_get_protein_embeddings", lambda self, seq: (torch.ones(1, 4, dtype=torch.long), torch.ones(1, 4, dtype=torch.long)))
    monkeypatch.setattr(MoleculeGenerator, "_encode_protein_for_model", lambda self, model, ids, mask: torch.ones(1, 4, 4))
    monkeypatch.setattr(
        MoleculeGenerator,
        "_generate_molecules_batch_with_tokens",
        lambda self, emb, mask, n: (
            ["[C]", "", "[O]", "[N]", "[C][O]"][:n],
            torch.ones(n, 8, dtype=torch.long),
        ),
    )
    monkeypatch.setattr(
        "prot2mol.inference.produce_molecules.tokenize_selfies_for_inference",
        lambda selfies_list, mol_tokenizer, max_mol_len, device=None: (
            torch.ones(len(selfies_list), max_mol_len, dtype=torch.long),
            torch.ones(len(selfies_list), max_mol_len, dtype=torch.long),
        ),
    )
    monkeypatch.setattr(MoleculeGenerator, "_predict_pchembl_batch", lambda self, pid, pmask, mid, mmask: np.arange(mid.shape[0], dtype=float))

    gen = MoleculeGenerator(cfg)
    out = gen.generate_molecules("MKT", num_samples=5)

    assert len(out) == 5
    assert "Generated_SELFIES" in out.columns
    assert "Predicted_pChEMBL" in out.columns


def test_generator_predict_pchembl_for_all_rows(monkeypatch):
    cfg = _generator_config(mode="prediction")

    def _fake_load_components(self):
        from conftest import DummyBatchTokenizer

        self.mol_tokenizer = DummyBatchTokenizer()
        self.prot_tokenizer = DummyBatchTokenizer()
        self.generation_model = SimpleNamespace()
        self.prediction_model = SimpleNamespace()

    monkeypatch.setattr(MoleculeGenerator, "_load_components", _fake_load_components)
    monkeypatch.setattr(MoleculeGenerator, "_get_protein_embeddings", lambda self, seq: (torch.ones(1, 4, dtype=torch.long), torch.ones(1, 4, dtype=torch.long)))
    monkeypatch.setattr(MoleculeGenerator, "_encode_protein_for_model", lambda self, model, ids, mask: torch.ones(1, 4, 4))
    monkeypatch.setattr(
        "prot2mol.inference.produce_molecules.tokenize_selfies_for_inference",
        lambda selfies_list, mol_tokenizer, max_mol_len, device=None: (
            torch.ones(len(selfies_list), max_mol_len, dtype=torch.long),
            torch.ones(len(selfies_list), max_mol_len, dtype=torch.long),
        ),
    )
    monkeypatch.setattr(MoleculeGenerator, "_predict_pchembl_batch", lambda self, pid, pmask, mid, mmask: np.full(mid.shape[0], 7.2))

    gen = MoleculeGenerator(cfg)
    df = pd.DataFrame({"smiles": ["CCO", "N", "O"]})
    out = gen.predict_pchembl("MKT", df)

    assert len(out) == 3
    assert np.allclose(out["Predicted_pChEMBL"].values, 7.2)


def test_generator_predict_pchembl_preserves_per_row_targets(monkeypatch):
    cfg = _generator_config(mode="prediction", prot_id=None)

    def _fake_load_components(self):
        from conftest import DummyBatchTokenizer

        self.mol_tokenizer = DummyBatchTokenizer()
        self.prot_tokenizer = DummyBatchTokenizer()
        self.generation_model = SimpleNamespace()
        self.prediction_model = SimpleNamespace()

    sequence_values = {"AAA": 1.0, "BBB": 2.0}
    monkeypatch.setattr(MoleculeGenerator, "_load_components", _fake_load_components)
    monkeypatch.setattr(
        MoleculeGenerator,
        "_get_protein_embeddings",
        lambda self, seq: (
            torch.tensor([[sequence_values[seq]]]),
            torch.ones(1, 1, dtype=torch.long),
        ),
    )
    monkeypatch.setattr(
        MoleculeGenerator,
        "_encode_protein_for_model",
        lambda self, model, ids, mask: ids.unsqueeze(-1),
    )
    monkeypatch.setattr(
        "prot2mol.inference.produce_molecules.tokenize_selfies_for_inference",
        lambda selfies_list, mol_tokenizer, max_mol_len, device=None: (
            torch.ones(len(selfies_list), max_mol_len, dtype=torch.long),
            torch.ones(len(selfies_list), max_mol_len, dtype=torch.long),
        ),
    )
    monkeypatch.setattr(
        MoleculeGenerator,
        "_predict_pchembl_batch",
        lambda self, protein_embeddings, prot_mask, mol_ids, mol_mask: np.full(
            len(mol_ids), protein_embeddings[0, 0, 0].item()
        ),
    )

    gen = MoleculeGenerator(cfg)
    df = pd.DataFrame(
        {
            "Target_FASTA": ["AAA", "BBB", "AAA"],
            "smiles": ["CCO", "N", "O"],
        }
    )
    out = gen.predict_pchembl(df)

    assert out["Target_FASTA"].tolist() == ["AAA", "BBB", "AAA"]
    assert out["Predicted_pChEMBL"].tolist() == [1.0, 2.0, 1.0]


def test_generator_normalizes_and_validates_mode(monkeypatch):
    monkeypatch.setattr(MoleculeGenerator, "_load_components", lambda self: None)

    gen = MoleculeGenerator(_generator_config(mode=" Prediction "))
    assert gen.config.mode == "prediction"

    with pytest.raises(ValueError, match="Unsupported mode"):
        MoleculeGenerator(_generator_config(mode="invalid"))


def test_predictor_dataframe_prediction(monkeypatch):
    cfg = _predictor_config(batch_size=2)

    def _fake_load_components(self):
        from conftest import DummyBatchTokenizer

        self.mol_tokenizer = DummyBatchTokenizer()
        self.prot_tokenizer = DummyBatchTokenizer()
        self.model = _PredictOnlyModel()

    monkeypatch.setattr(PChemblPredictor, "_auto_configure_model", lambda self: None)
    monkeypatch.setattr(PChemblPredictor, "_load_components", _fake_load_components)
    monkeypatch.setattr(
        "prot2mol.inference.predict_pchembl.tokenize_selfies_for_inference",
        lambda selfies_list, mol_tokenizer, max_mol_len, device=None: (
            torch.ones(len(selfies_list), max_mol_len, dtype=torch.long),
            torch.ones(len(selfies_list), max_mol_len, dtype=torch.long),
        ),
    )

    pred = PChemblPredictor(cfg)
    df = pd.DataFrame(
        {
            "Target_FASTA": ["MKT", "AAA", "GGG"],
            "smiles": ["CCO", "N", "O"],
        }
    )

    out = pred._predict_dataframe(df)
    assert len(out) == 3
    assert "Predicted_pChEMBL" in out.columns


def test_predictor_requires_target_context(monkeypatch):
    cfg = _predictor_config()

    monkeypatch.setattr(PChemblPredictor, "_auto_configure_model", lambda self: None)
    monkeypatch.setattr(PChemblPredictor, "_load_components", lambda self: None)

    pred = PChemblPredictor(cfg)
    pred.model = SimpleNamespace()
    pred.mol_tokenizer = SimpleNamespace()

    df = pd.DataFrame({"smiles": ["CCO", "N"]})

    try:
        pred._predict_dataframe(df)
        assert False, "Expected ValueError for missing Target_FASTA/Target_CHEMBL_ID"
    except ValueError as exc:
        assert "Target_FASTA" in str(exc) or "Target_CHEMBL_ID" in str(exc)


def test_predictor_load_eval_split_dataset_respects_mode(monkeypatch):
    cfg = _predictor_config()

    monkeypatch.setattr(PChemblPredictor, "_auto_configure_model", lambda self: None)
    monkeypatch.setattr(PChemblPredictor, "_load_components", lambda self: None)

    captured = {}

    def _fake_split(full_data, split_mode, split_ratio, split_seed, logger=None, **kwargs):
        captured["full_data"] = full_data
        captured["split_mode"] = split_mode
        captured["split_ratio"] = split_ratio
        captured["split_seed"] = split_seed
        return "train_split", {"rows": [1, 2]}

    monkeypatch.setattr(
        predict_module,
        "load_processed_dataset",
        lambda input_file, cache_dir=None: ({"train": "FULL_DATA"}, "/tmp/cache/in"),
    )
    monkeypatch.setattr(predict_module, "split_train_eval_dataset", _fake_split)

    pred = PChemblPredictor(cfg)
    eval_ds = pred._load_eval_split_dataset(split_mode="aid", split_ratio=0.25, split_seed=13)

    assert eval_ds == {"rows": [1, 2]}
    assert captured["full_data"] == "FULL_DATA"
    assert captured["split_mode"] == "aid"
    assert captured["split_ratio"] == 0.25
    assert captured["split_seed"] == 13


def test_predictor_uses_matching_precomputed_split(monkeypatch):
    cfg = _predictor_config()
    monkeypatch.setattr(PChemblPredictor, "_auto_configure_model", lambda self: None)
    monkeypatch.setattr(PChemblPredictor, "_load_components", lambda self: None)
    monkeypatch.setattr(
        predict_module,
        "load_processed_dataset",
        lambda input_file, cache_dir=None: (
            {"train": "TRAIN", "test": "EVAL"},
            "/tmp/cache/in",
        ),
    )
    monkeypatch.setattr(
        predict_module,
        "load_processed_stats",
        lambda input_file, cache_dir=None: {
            "eval_split": "aid",
            "eval_split_ratio": 0.25,
            "split_seed": 13,
        },
    )
    monkeypatch.setattr(
        predict_module,
        "split_train_eval_dataset",
        lambda *args, **kwargs: pytest.fail("matching cached split should not be rebuilt"),
    )

    pred = PChemblPredictor(cfg)
    train, eval_data = pred._load_requested_split_dataset(
        dataset_path=cfg.input_file,
        split_mode="aid",
        split_ratio=0.25,
        split_seed=13,
    )

    assert train == "TRAIN"
    assert eval_data == "EVAL"


def test_predictor_resolves_checkpoint_split_settings(monkeypatch):
    cfg = _predictor_config()
    monkeypatch.setattr(PChemblPredictor, "_auto_configure_model", lambda self: None)
    monkeypatch.setattr(PChemblPredictor, "_load_components", lambda self: None)
    pred = PChemblPredictor(cfg)
    pred.saved_model_config = {
        "eval_split": "aid",
        "eval_split_ratio": 0.2,
        "split_seed": 17,
    }

    assert pred._resolve_reproduce_settings("auto") == ("aid", 0.2, 17)
    assert pred._resolve_reproduce_settings("aid") == ("aid", 0.2, 17)
    assert pred._resolve_reproduce_settings("random") == ("random", 0.01, 42)

    pred.config.reproduce_split_ratio = 0.3
    pred.config.reproduce_split_seed = 23
    assert pred._resolve_reproduce_settings("random") == ("random", 0.3, 23)


def test_predictor_auto_reproduce_requires_checkpoint_metadata(monkeypatch):
    cfg = _predictor_config()
    monkeypatch.setattr(PChemblPredictor, "_auto_configure_model", lambda self: None)
    monkeypatch.setattr(PChemblPredictor, "_load_components", lambda self: None)
    pred = PChemblPredictor(cfg)

    with pytest.raises(ValueError, match="does not contain eval split metadata"):
        pred._resolve_reproduce_settings("auto")


def test_predictor_filters_reference_rows_case_insensitively(monkeypatch):
    cfg = _predictor_config()
    monkeypatch.setattr(PChemblPredictor, "_auto_configure_model", lambda self: None)
    monkeypatch.setattr(PChemblPredictor, "_load_components", lambda self: None)
    pred = PChemblPredictor(cfg)
    df = pd.DataFrame(
        {
            "target_chembl_id": ["CHEMBL1", "chembl2", "CHEMBL1"],
            "pchembl_value_Median": [5.0, 6.0, 7.0],
        }
    )

    matched = pred._filter_reference_rows_for_protein(df, "chembl1")
    assert matched["pchembl_value_Median"].tolist() == [5.0, 7.0]


def test_predictor_distribution_plot_handles_finite_values(tmp_path, monkeypatch):
    cfg = _predictor_config(distribution_bins=12)
    monkeypatch.setattr(PChemblPredictor, "_auto_configure_model", lambda self: None)
    monkeypatch.setattr(PChemblPredictor, "_load_components", lambda self: None)
    pred = PChemblPredictor(cfg)
    output = tmp_path / "plots" / "distribution.png"

    saved = pred._save_distribution_plot(
        batch_predictions=[5.0, np.nan, 6.0, 7.0],
        output_path=str(output),
        title="Predictions",
        batch_label="Batch",
        reference_values=[4.0, 5.0, np.inf],
        reference_label="Reference",
    )

    assert saved == str(output)
    assert output.exists() and output.stat().st_size > 0
    assert pred._save_distribution_plot(
        batch_predictions=[np.nan, np.inf],
        output_path=str(tmp_path / "empty.png"),
        title="Empty",
        batch_label="Batch",
    ) is None


def test_predictor_sanitize_output_dataframe_drops_internal_columns(monkeypatch):
    cfg = _predictor_config()
    monkeypatch.setattr(PChemblPredictor, "_auto_configure_model", lambda self: None)
    monkeypatch.setattr(PChemblPredictor, "_load_components", lambda self: None)

    pred = PChemblPredictor(cfg)
    raw = pd.DataFrame(
        {
            "Target_FASTA": ["MKT"],
            "prot_input_ids": [[1, 2]],
            "prot_attention_mask": [[1, 1]],
            "mol_input_ids": [[3, 4]],
            "mol_attention_mask": [[1, 1]],
            "labels": [[5, 6]],
            "train_lm": [True],
            "train_m": [True],
            "Predicted_pChEMBL": [7.1],
            "pchembl_value_Median": [7.0],
        }
    )

    out = pred._sanitize_output_dataframe(raw)
    for col in [
        "Target_FASTA",
        "prot_input_ids",
        "prot_attention_mask",
        "mol_input_ids",
        "mol_attention_mask",
        "labels",
        "train_lm",
        "train_m",
    ]:
        assert col not in out.columns
    assert "Predicted_pChEMBL" in out.columns
    assert "pchembl_value_Median" in out.columns


def test_predictor_resolves_target_fasta_from_chembl_mapping(tmp_path, monkeypatch):
    cfg = _predictor_config(batch_size=2)

    map_file = tmp_path / "chembl_uniprot_mapping.txt"
    map_file.write_text(
        "# header\n"
        "P24941\tCHEMBL301\tCyclin-dependent kinase 2\tSINGLE PROTEIN\n"
    )
    targets_file = tmp_path / "protein_targets.tsv"
    targets_file.write_text(
        "target_id\tHGNC_symbol\tUniProtID\tStatus\tOrganism\tClassification\tLength\tSequence\n"
        "P24941_WT\tCDK2\tCDK2_HUMAN\treviewed\tHomo sapiens\tClass\t4\tMKTK\n"
    )
    cfg.chembl_uniprot_mapping_path = str(map_file)
    cfg.protein_targets_path = str(targets_file)

    def _fake_load_components(self):
        from conftest import DummyBatchTokenizer

        self.mol_tokenizer = DummyBatchTokenizer()
        self.prot_tokenizer = DummyBatchTokenizer()
        self.model = _PredictOnlyModel()

    monkeypatch.setattr(PChemblPredictor, "_auto_configure_model", lambda self: None)
    monkeypatch.setattr(PChemblPredictor, "_load_components", _fake_load_components)
    monkeypatch.setattr(
        "prot2mol.inference.predict_pchembl.tokenize_selfies_for_inference",
        lambda selfies_list, mol_tokenizer, max_mol_len, device=None: (
            torch.ones(len(selfies_list), max_mol_len, dtype=torch.long),
            torch.ones(len(selfies_list), max_mol_len, dtype=torch.long),
        ),
    )

    pred = PChemblPredictor(cfg)
    df = pd.DataFrame(
        {
            "Target_CHEMBL_ID": ["CHEMBL301", "CHEMBL301"],
            "Generated_SELFIES": ["[C]", "[O]"],
        }
    )
    out = pred._predict_dataframe(df)

    assert "Target_FASTA" in out.columns
    assert set(out["Target_FASTA"].tolist()) == {"MKTK"}
    assert "Predicted_pChEMBL" in out.columns


def test_predictor_resolves_target_fasta_from_uniprot_id(tmp_path, monkeypatch):
    cfg = _predictor_config(batch_size=2)

    targets_file = tmp_path / "protein_targets.tsv"
    targets_file.write_text(
        "target_id\tHGNC_symbol\tUniProtID\tStatus\tOrganism\tClassification\tLength\tSequence\n"
        "P31749_WT\tAKT1\tAKT1_HUMAN\treviewed\tHomo sapiens\tClass\t4\tAKTS\n"
    )
    cfg.protein_targets_path = str(targets_file)

    def _fake_load_components(self):
        from conftest import DummyBatchTokenizer

        self.mol_tokenizer = DummyBatchTokenizer()
        self.prot_tokenizer = DummyBatchTokenizer()
        self.model = _PredictOnlyModel()

    monkeypatch.setattr(PChemblPredictor, "_auto_configure_model", lambda self: None)
    monkeypatch.setattr(PChemblPredictor, "_load_components", _fake_load_components)
    monkeypatch.setattr(
        "prot2mol.inference.predict_pchembl.tokenize_selfies_for_inference",
        lambda selfies_list, mol_tokenizer, max_mol_len, device=None: (
            torch.ones(len(selfies_list), max_mol_len, dtype=torch.long),
            torch.ones(len(selfies_list), max_mol_len, dtype=torch.long),
        ),
    )

    pred = PChemblPredictor(cfg)
    df = pd.DataFrame(
        {
            "UniProt_ID": ["P31749", "P31749"],
            "Generated_SELFIES": ["[C]", "[O]"],
        }
    )
    out = pred._predict_dataframe(df)

    assert "Target_FASTA" in out.columns
    assert set(out["Target_FASTA"].tolist()) == {"AKTS"}
    assert "Predicted_pChEMBL" in out.columns


def test_predictor_resolves_target_fasta_from_lowercase_uniprot_id(tmp_path, monkeypatch):
    cfg = _predictor_config(batch_size=2)

    targets_file = tmp_path / "protein_targets.tsv"
    targets_file.write_text(
        "target_id\tHGNC_symbol\tUniProtID\tStatus\tOrganism\tClassification\tLength\tSequence\n"
        "P31749_WT\tAKT1\tAKT1_HUMAN\treviewed\tHomo sapiens\tClass\t4\tAKTS\n"
    )
    cfg.protein_targets_path = str(targets_file)

    def _fake_load_components(self):
        from conftest import DummyBatchTokenizer

        self.mol_tokenizer = DummyBatchTokenizer()
        self.prot_tokenizer = DummyBatchTokenizer()
        self.model = _PredictOnlyModel()

    monkeypatch.setattr(PChemblPredictor, "_auto_configure_model", lambda self: None)
    monkeypatch.setattr(PChemblPredictor, "_load_components", _fake_load_components)
    monkeypatch.setattr(
        "prot2mol.inference.predict_pchembl.tokenize_selfies_for_inference",
        lambda selfies_list, mol_tokenizer, max_mol_len, device=None: (
            torch.ones(len(selfies_list), max_mol_len, dtype=torch.long),
            torch.ones(len(selfies_list), max_mol_len, dtype=torch.long),
        ),
    )

    pred = PChemblPredictor(cfg)
    df = pd.DataFrame(
        {
            "uniprot_id": ["P31749", "P31749"],
            "Generated_SELFIES": ["[C]", "[O]"],
        }
    )
    out = pred._predict_dataframe(df)

    assert "Target_FASTA" in out.columns
    assert set(out["Target_FASTA"].tolist()) == {"AKTS"}
    assert "Predicted_pChEMBL" in out.columns


def test_predictor_auto_configure_uses_parent_checkpoint_config(tmp_path, monkeypatch):
    model_parent = tmp_path / "run_dir"
    ckpt = model_parent / "checkpoint-123"
    ckpt.mkdir(parents=True)
    (model_parent / "config.json").write_text(
        json.dumps(
            {
                "n_layer": 12,
                "n_head": 16,
                "n_emb": 1024,
                "prot_emb_model": "esm2",
                "max_mol_len": 200,
                "prot_max_length": 1000,
                "pchembl_tf_hidden_dim": 640,
                "pchembl_tf_num_heads": 10,
                "pchembl_tf_group_size": 3,
                "pchembl_tf_agg_mode": "cls",
                "pchembl_tf_dropout": 0.25,
            }
        )
    )

    cfg = _predictor_config(model_path=str(ckpt), n_layer=1, n_head=2, n_emb=8, prot_emb_model="prot_t5")
    monkeypatch.setattr(PChemblPredictor, "_load_components", lambda self: None)

    pred = PChemblPredictor(cfg)
    assert pred.config.n_layer == 12
    assert pred.config.n_head == 16
    assert pred.config.n_emb == 1024
    assert pred.config.prot_emb_model == "esm2"
    assert pred.config.pchembl_tf_hidden_dim == 640
    assert pred.config.pchembl_tf_num_heads == 10
    assert pred.config.pchembl_tf_group_size == 3
    assert pred.config.pchembl_tf_agg_mode == "cls"
    assert pred.config.pchembl_tf_dropout == 0.25


def test_predictor_load_components_uses_project_models_fallback(monkeypatch):
    cfg = _predictor_config()
    monkeypatch.setattr(PChemblPredictor, "_auto_configure_model", lambda self: None)
    monkeypatch.delenv("MODELS_BASE_PATH", raising=False)

    captured = {}

    def _fake_load_molgen_tokenizer(models_base=None, fallback_bases=None, padding_side="left"):
        captured["models_base"] = models_base
        captured["fallback_bases"] = fallback_bases
        captured["padding_side"] = padding_side
        return SimpleNamespace()

    class _FakeModel:
        def eval(self):
            return self

    monkeypatch.setattr(predict_module, "load_molgen_tokenizer", _fake_load_molgen_tokenizer)
    monkeypatch.setattr(predict_module, "get_protein_tokenizer", lambda _: SimpleNamespace())
    monkeypatch.setattr(PChemblPredictor, "_load_model", lambda self, _: _FakeModel())

    PChemblPredictor(cfg)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert captured["models_base"] is None
    assert os.path.join(project_root, "models") in captured["fallback_bases"]
    assert os.path.join(os.path.expanduser("~"), "Prot2Mol", "models") in captured["fallback_bases"]
    assert captured["padding_side"] == "left"


def test_parse_args_accepts_reproduce_and_rejects_legacy_eval_flag():
    args = parse_args(
        [
            "--input_file",
            "/tmp/in.csv",
            "--model_path",
            "/tmp/model",
            "--output_file",
            "/tmp/out.csv",
            "--reproduce",
            "auto",
        ]
    )
    assert args.reproduce == "auto"
    assert args.reproduce_split_ratio is None
    assert args.reproduce_split_seed is None

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input_file",
                "/tmp/in.csv",
                "--model_path",
                "/tmp/model",
                "--output_file",
                "/tmp/out.csv",
                "--eval",
            ]
        )
