from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from prot2mol.inference.produce_molecules import MoleculeGenerator
from prot2mol.inference.predict_pchembl import PChemblPredictor


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
        data_path=None,
        prot_emb_model="prot_t5",
        n_layer=1,
        n_head=2,
        n_emb=8,
        prot_max_length=16,
        max_mol_len=8,
        batch_size=2,
        pchembl_mean=5.0,
        pchembl_std=2.0,
        eval=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


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
    monkeypatch.setattr(MoleculeGenerator, "_generate_molecules_batch", lambda self, ids, mask, n: ["[C]", "", "[O]", "[N]", "[C][O]"][:n])
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


def test_predictor_dataframe_prediction(monkeypatch):
    cfg = _predictor_config(batch_size=2)

    def _fake_load_components(self):
        from conftest import DummyBatchTokenizer

        self.mol_tokenizer = DummyBatchTokenizer()
        self.prot_tokenizer = DummyBatchTokenizer()

        class _FakeModel:
            def eval(self):
                return self

            def __call__(self, mol_input_ids, prot_input_ids, prot_attention_mask, train_lm=False):
                b = mol_input_ids.shape[0]
                return {"pchembl_predictions": torch.arange(b, dtype=torch.float32)}

        self.model = _FakeModel()

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
