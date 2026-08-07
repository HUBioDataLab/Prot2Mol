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
        "pchembl_tf_hidden_dim": 8,
        "pchembl_tf_num_heads": 2,
        "pchembl_tf_group_size": 1,
        "pchembl_tf_agg_mode": "mean",
        "pchembl_tf_dropout": 0.1,
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


def test_model_forward_uses_token_fusion_head_without_cross_attention(monkeypatch):
    from conftest import DummyBatchTokenizer, DummyDecoder, DummyEncoderObj

    monkeypatch.setattr(model_module, "GPT2LMHeadModel", DummyDecoder)
    monkeypatch.setattr(model_module, "get_protein_encoder", lambda model_name, max_length, active: DummyEncoderObj(hidden_size=8))
    monkeypatch.setattr(model_module, "get_encoder_size", lambda model_name: 8)

    model = model_module.create_prot2mol_model(_dummy_model_config(DummyBatchTokenizer()))
    outputs = model(
        mol_input_ids=torch.tensor([[1, 3, 0, 0], [1, 3, 3, 0]], dtype=torch.long),
        prot_input_ids=torch.tensor([[1, 2, 3, 0], [1, 2, 0, 0]], dtype=torch.long),
        prot_attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long),
        labels=torch.tensor([[1, 3, -100, -100], [1, 3, 3, -100]], dtype=torch.long),
    )

    assert outputs["pchembl_predictions"].shape == (2,)
    assert "cross_attentions" not in outputs


def test_token_fusion_head_respects_masks_and_non_divisible_grouping():
    torch.manual_seed(0)
    head = model_module.FusionDTIPChemblHead(
        d_model=4,
        hidden_dim=8,
        num_heads=2,
        group_size=2,
        agg_mode="mean",
        dropout=0.0,
    )
    head.eval()

    protein = torch.randn(1, 5, 4)
    molecule = torch.randn(1, 5, 4)
    prot_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)
    mol_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long)

    grouped_protein, grouped_mask = head._group_embeddings(head.ln_p(head.proj_p(protein)), prot_mask.bool())
    assert grouped_protein.shape == (1, 3, 8)
    assert grouped_mask.shape == (1, 3)

    altered_protein = protein.clone()
    altered_molecule = molecule.clone()
    altered_protein[:, 3:, :] = altered_protein[:, 3:, :] + 1000.0
    altered_molecule[:, 4:, :] = altered_molecule[:, 4:, :] - 1000.0

    base_pred = head(protein, molecule, prot_mask, mol_mask)
    masked_pred = head(altered_protein, altered_molecule, prot_mask, mol_mask)

    assert base_pred.shape == (1,)
    assert torch.allclose(base_pred, masked_pred, atol=1e-5)


def test_token_fusion_masked_softmax_is_finite_in_float16():
    fusion = model_module.FusionDTITokenFusion(hidden_dim=4, num_heads=2)
    logits = torch.tensor(
        [[[[1.0, -1.0], [2.0, 0.0], [3.0, 1.0]], [[4.0, 2.0], [5.0, 3.0], [6.0, 4.0]]]],
        dtype=torch.float16,
    )
    row_mask = torch.tensor([[True, False]])
    col_mask = torch.tensor([[True, True, False]])

    probabilities = fusion._masked_softmax(logits, row_mask, col_mask)

    assert torch.isfinite(probabilities).all()
    assert torch.all(probabilities[:, 1] == 0)
    assert torch.all(probabilities[:, :, 2] == 0)
    assert torch.allclose(
        probabilities[:, 0, :2].sum(dim=1),
        torch.ones((1, 2), dtype=torch.float16),
    )


def test_load_prot2mol_inference_model_prefers_saved_token_fusion_config(tmp_path, monkeypatch):
    model_parent = tmp_path / "run_dir"
    ckpt = model_parent / "checkpoint-123"
    ckpt.mkdir(parents=True)
    (ckpt / "pytorch_model.bin").write_bytes(b"stub")
    (model_parent / "config.json").write_text(
        '{"n_layer": 12, "n_head": 16, "n_emb": 1024, "prot_emb_model": "esm2", '
        '"max_mol_len": 200, "prot_max_length": 1000, '
        '"pchembl_tf_hidden_dim": 640, "pchembl_tf_num_heads": 10, '
        '"pchembl_tf_group_size": 3, "pchembl_tf_agg_mode": "cls", "pchembl_tf_dropout": 0.25}'
    )

    class DummyInferenceModel(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg

        def load_state_dict(self, state, strict=True):
            return [], []

        def to(self, device):
            return self

    monkeypatch.setattr(hf_utils, "_resolve_checkpoint_file", lambda model_path: str(ckpt / "pytorch_model.bin"))
    monkeypatch.setattr(hf_utils.torch, "load", lambda *a, **k: {})
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
        pchembl_tf_hidden_dim=768,
        pchembl_tf_num_heads=8,
        pchembl_tf_group_size=1,
        pchembl_tf_agg_mode="mean",
        pchembl_tf_dropout=0.1,
        strict=False,
        allow_strict_fallback=False,
    )

    assert loaded.cfg["prot_emb_model"] == "esm2"
    assert loaded.cfg["n_layer"] == 12
    assert loaded.cfg["pchembl_tf_hidden_dim"] == 640
    assert loaded.cfg["pchembl_tf_group_size"] == 3
    assert loaded.cfg["pchembl_tf_agg_mode"] == "cls"


def test_load_prot2mol_inference_model_filters_legacy_pchembl_head(tmp_path, monkeypatch):
    ckpt = tmp_path / "legacy_model"
    ckpt.mkdir(parents=True)
    (ckpt / "pytorch_model.bin").write_bytes(b"stub")

    captured = {}

    class DummyInferenceModel(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg

        def load_state_dict(self, state, strict=True):
            captured["keys"] = sorted(state.keys())
            captured["strict"] = strict
            return [], []

        def to(self, device):
            return self

    monkeypatch.setattr(hf_utils, "_resolve_checkpoint_file", lambda model_path: str(ckpt / "pytorch_model.bin"))
    monkeypatch.setattr(
        hf_utils.torch,
        "load",
        lambda *a, **k: {
            "protein_encoder.encoder_model.weight": torch.tensor([1.0]),
            "pchembl_head.legacy.weight": torch.tensor([2.0]),
        },
    )
    monkeypatch.setattr("prot2mol.core.model.Prot2MolModel", DummyInferenceModel)

    hf_utils.load_prot2mol_inference_model(
        model_path=str(ckpt),
        device=torch.device("cpu"),
        mol_tokenizer=SimpleNamespace(added_tokens_decoder={0: "<pad>"}, pad_token_id=0, bos_token_id=1, eos_token_id=2),
        prot_emb_model="prot_t5",
        n_layer=1,
        n_head=2,
        n_emb=8,
        max_mol_len=8,
        prot_max_length=16,
        strict=False,
        allow_strict_fallback=False,
    )

    assert captured["strict"] is False
    assert captured["keys"] == ["protein_encoder.encoder_model.weight"]
