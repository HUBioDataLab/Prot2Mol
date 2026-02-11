from types import SimpleNamespace

import torch

from prot2mol.core import model as model_module
from prot2mol.io import hf_utils


def _dummy_model_config(mol_tokenizer):
    return {
        "prot_emb_model": "prot_t5",
        "n_layer": 1,
        "n_head": 2,
        "n_emb": 8,
        "max_mol_len": 8,
        "prot_max_length": 16,
        "train_encoder_model": True,
        "train_decoder_model": True,
        "train_pchembl_head": True,
        "stop_pchembl_gradients": True,
        "mol_tokenizer": mol_tokenizer,
    }


def test_model_creation_and_trainability_switch(monkeypatch):
    from conftest import DummyBatchTokenizer, DummyDecoder, DummyEncoderObj

    monkeypatch.setattr(model_module, "GPT2LMHeadModel", DummyDecoder)
    monkeypatch.setattr(model_module, "get_protein_encoder", lambda model_name, max_length, active: DummyEncoderObj(hidden_size=8))
    monkeypatch.setattr(model_module, "get_encoder_size", lambda model_name: 8)

    model = model_module.create_prot2mol_model(_dummy_model_config(DummyBatchTokenizer()))

    model.update_trainable_components(trainable_encoder=False, trainable_decoder=True, trainable_pchembl_head=False)

    assert not any(p.requires_grad for p in model.protein_encoder.parameters())
    assert any(p.requires_grad for p in model.molecule_decoder.parameters())
    assert not any(p.requires_grad for p in model.pchembl_head.parameters())
    assert model.lm_weight.requires_grad and model.pchembl_weight.requires_grad

    model.train(True)
    assert model.protein_encoder.training is False
    assert model.pchembl_head.training is False
    assert model.molecule_decoder.training is True


def test_load_prot2mol_inference_model_with_strict_fallback(tmp_path, monkeypatch):
    ckpt = tmp_path / "model_dir"
    ckpt.mkdir(parents=True)
    (ckpt / "pytorch_model.bin").write_bytes(b"stub")

    class DummyInferenceModel(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.loaded = []
            self.device_seen = None

        def load_state_dict(self, state, strict=True):
            self.loaded.append(strict)
            if strict:
                raise RuntimeError("mismatch")
            return [], []

        def to(self, device):
            self.device_seen = device
            return self

    monkeypatch.setattr(hf_utils, "_resolve_checkpoint_file", lambda model_path: str(ckpt / "pytorch_model.bin"))
    monkeypatch.setattr(hf_utils.torch, "load", lambda *a, **k: {"x": torch.tensor([1.0])})
    monkeypatch.setattr("prot2mol.core.model.Prot2MolModel", DummyInferenceModel)

    loaded = hf_utils.load_prot2mol_inference_model(
        model_path=str(ckpt),
        device=torch.device("cpu"),
        mol_tokenizer=SimpleNamespace(added_tokens_decoder={0: "<pad>"}, pad_token_id=0, bos_token_id=1, eos_token_id=2),
        prot_emb_model="prot_t5",
        n_layer=1,
        n_head=2,
        n_emb=8,
        max_mol_len=8,
        prot_max_length=16,
        strict=True,
        allow_strict_fallback=True,
        logger=SimpleNamespace(warning=lambda *a, **k: None),
    )

    assert loaded.loaded == [True, False]
    assert loaded.device_seen == torch.device("cpu")
