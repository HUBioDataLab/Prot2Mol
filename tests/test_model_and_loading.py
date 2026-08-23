from types import SimpleNamespace

import pytest
import torch
from transformers import BartConfig, BartForConditionalGeneration

from prot2mol.core import model as model_module
from prot2mol.io import hf_utils


def _tiny_bart():
    return BartForConditionalGeneration(
        BartConfig(
            vocab_size=16,
            d_model=8,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=16,
            decoder_ffn_dim=16,
            max_position_embeddings=16,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            decoder_start_token_id=2,
        )
    )


def _config(tokenizer):
    return {
        "prot_emb_model": "esm2",
        "protein_model_id": "dummy/protein",
        "decoder_type": "molgen",
        "decoder_model_id": "dummy/molgen",
        "conditioning_dropout": 0.0,
        "max_mol_len": 8,
        "prot_max_length": 16,
        "train_encoder_model": True,
        "train_projection_model": True,
        "train_decoder_model": True,
        "mol_tokenizer": tokenizer,
    }


def _patch_tiny_architecture(monkeypatch):
    from conftest import DummyProteinEncoder

    monkeypatch.setattr(
        model_module,
        "get_protein_encoder",
        lambda model_name, model_id, active: DummyProteinEncoder(hidden_size=8),
    )
    monkeypatch.setattr(
        model_module.AutoModelForSeq2SeqLM,
        "from_pretrained",
        lambda path: _tiny_bart(),
    )


def test_model_uses_pretrained_bart_decoder_and_removes_unused_encoder_layers(monkeypatch):
    from conftest import DummyBatchTokenizer

    _patch_tiny_architecture(monkeypatch)
    model = model_module.create_prot2mol_model(_config(DummyBatchTokenizer()))
    assert len(model.molecule_decoder.model.encoder.layers) == 0
    assert model.conditioning_projection[0].in_features == 8
    assert model.conditioning_projection[0].out_features == 8
    assert model.config.vocab_size == len(DummyBatchTokenizer())
    assert model.molecule_decoder.model.shared.weight.requires_grad
    assert not model.molecule_decoder.model.encoder.embed_positions.weight.requires_grad
    assert not model.molecule_decoder.model.encoder.layernorm_embedding.weight.requires_grad


def test_model_rejects_tokenizer_that_does_not_match_pretrained_decoder(monkeypatch):
    from conftest import DummyBatchTokenizer

    _patch_tiny_architecture(monkeypatch)
    with pytest.raises(ValueError, match="special-token ids"):
        model_module.create_prot2mol_model(
            _config(DummyBatchTokenizer(pad_token_id=1, bos_token_id=0, eos_token_id=2))
        )


def test_model_trainability_keeps_frozen_modules_in_eval(monkeypatch):
    from conftest import DummyBatchTokenizer

    _patch_tiny_architecture(monkeypatch)
    model = model_module.create_prot2mol_model(_config(DummyBatchTokenizer()))
    model.update_trainable_components(False, True, True)
    model.train()

    assert not any(parameter.requires_grad for parameter in model.protein_encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.conditioning_projection.parameters())
    assert model.protein_encoder.training is False
    assert model.conditioning_projection.training is True
    assert model.molecule_decoder.training is True
    model.update_trainable_components(False, False, False)
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_forward_delegates_shift_right_to_bart_and_uses_explicit_masks(monkeypatch):
    from conftest import DummyBatchTokenizer

    _patch_tiny_architecture(monkeypatch)
    model = model_module.create_prot2mol_model(_config(DummyBatchTokenizer()))
    labels = torch.tensor([[1, 3, 2, -100], [1, 4, 5, 2]])
    captured = {}
    original = model.molecule_decoder.forward

    def recording_forward(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    model.molecule_decoder.forward = recording_forward
    outputs = model(
        prot_input_ids=torch.tensor([[1, 2, 3, 0], [1, 2, 0, 0]]),
        prot_attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        labels=labels,
    )

    assert captured["decoder_input_ids"] is None
    assert captured["labels"] is labels
    assert torch.equal(captured["attention_mask"], torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]))
    assert set(outputs) == {"logits", "lm_loss", "loss"}
    assert torch.isfinite(outputs["loss"])


def test_generation_uses_external_protein_encoder_outputs(monkeypatch):
    from conftest import DummyBatchTokenizer

    _patch_tiny_architecture(monkeypatch)
    model = model_module.create_prot2mol_model(_config(DummyBatchTokenizer()))
    model.eval()
    generated = model.generate(
        prot_input_ids=torch.tensor([[1, 2, 0]]),
        prot_attention_mask=torch.tensor([[1, 1, 0]]),
        max_length=5,
    )
    assert generated.shape[0] == 1
    assert generated.shape[1] <= 5


def test_molgen_generated_token_log_probs_align_targets_and_stop_at_eos(monkeypatch):
    from conftest import DummyBatchTokenizer

    _patch_tiny_architecture(monkeypatch)
    model = model_module.create_prot2mol_model(_config(DummyBatchTokenizer()))
    generated_ids = torch.tensor(
        [[2, 3, 2, 0, 0], [2, 4, 5, 0, 0]],
        dtype=torch.long,
    )
    log_probs, action_mask = model.generated_token_log_probs(
        generated_ids,
        prot_input_ids=torch.tensor([[1, 2, 0], [3, 4, 0]]),
        prot_attention_mask=torch.tensor([[1, 1, 0], [1, 1, 0]]),
    )

    assert log_probs.shape == (2, 4)
    assert torch.isfinite(log_probs).all()
    assert torch.equal(
        action_mask,
        torch.tensor([[True, True, False, False], [True, True, False, False]]),
    )
    assert torch.equal(log_probs.masked_select(~action_mask), torch.zeros(4))


def test_legacy_left_padded_log_probs_score_pad_actions_until_eos(monkeypatch):
    from conftest import DummyBatchTokenizer

    _patch_tiny_architecture(monkeypatch)
    model = model_module.create_prot2mol_model(_config(DummyBatchTokenizer()))
    generated_ids = torch.tensor(
        [[0, 0, 1, 3, 2, 0], [0, 0, 0, 4, 5, 2]],
        dtype=torch.long,
    )

    log_probs, action_mask = model.generated_token_log_probs(
        generated_ids,
        prot_input_ids=torch.tensor([[1, 2, 0], [3, 4, 0]]),
        prot_attention_mask=torch.tensor([[1, 1, 0], [1, 1, 0]]),
        pad_token_is_termination=False,
    )

    assert log_probs.shape == (2, 5)
    assert torch.equal(
        action_mask,
        torch.tensor(
            [[True, True, True, True, False], [True, True, True, True, True]]
        ),
    )
    assert torch.isfinite(log_probs).all()


def test_loss_backpropagates_through_cross_attention_conditioning(monkeypatch):
    from conftest import DummyBatchTokenizer

    _patch_tiny_architecture(monkeypatch)
    model = model_module.create_prot2mol_model(_config(DummyBatchTokenizer()))
    common = {
        "prot_attention_mask": torch.tensor([[1, 1, 1, 0]]),
        "labels": torch.tensor([[1, 3, 2, -100]]),
    }
    first = model(prot_input_ids=torch.tensor([[1, 2, 3, 0]]), **common)
    second = model(prot_input_ids=torch.tensor([[4, 5, 6, 0]]), **common)
    assert not torch.allclose(first["logits"], second["logits"])

    first["loss"].backward()
    assert model.conditioning_projection[0].weight.grad is not None
    assert model.protein_encoder.model.weight.grad is not None
    cross_attention_grads = [
        parameter.grad
        for name, parameter in model.molecule_decoder.named_parameters()
        if "encoder_attn" in name
    ]
    assert any(gradient is not None for gradient in cross_attention_grads)
    missing_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing_gradients == []


def test_gpt2_decoder_restores_direct_protein_cross_attention(monkeypatch):
    from conftest import DummyBatchTokenizer, DummyProteinEncoder

    monkeypatch.setattr(
        model_module,
        "get_protein_encoder",
        lambda model_name, model_id, active: DummyProteinEncoder(hidden_size=8),
    )
    config = _config(DummyBatchTokenizer())
    config.update(
        {
            "decoder_type": "gpt2",
            "n_layer": 1,
            "n_head": 2,
            "n_emb": 8,
        }
    )
    model = model_module.create_prot2mol_model(config)

    assert model.decoder_type == "gpt2"
    assert model.molecule_decoder.config.add_cross_attention is True
    assert model.molecule_decoder.config.n_layer == 1
    assert model.molecule_decoder.config.n_head == 2
    assert isinstance(model.conditioning_projection, torch.nn.Identity)

    common = {
        "prot_attention_mask": torch.tensor([[1, 1, 1, 0]]),
        "labels": torch.tensor([[1, 3, 2, -100]]),
    }
    first = model(prot_input_ids=torch.tensor([[1, 2, 3, 0]]), **common)
    second = model(prot_input_ids=torch.tensor([[4, 5, 6, 0]]), **common)
    assert not torch.allclose(first["logits"], second["logits"])

    first["loss"].backward()
    cross_attention_grads = [
        parameter.grad
        for name, parameter in model.molecule_decoder.named_parameters()
        if "crossattention" in name
    ]
    assert any(gradient is not None for gradient in cross_attention_grads)


def test_gpt2_generation_uses_protein_encoder_states(monkeypatch):
    from conftest import DummyBatchTokenizer, DummyProteinEncoder

    monkeypatch.setattr(
        model_module,
        "get_protein_encoder",
        lambda model_name, model_id, active: DummyProteinEncoder(hidden_size=8),
    )
    config = _config(DummyBatchTokenizer())
    config.update(
        {
            "decoder_type": "gpt2",
            "n_layer": 1,
            "n_head": 2,
            "n_emb": 8,
        }
    )
    model = model_module.create_prot2mol_model(config)
    generated = model.generate(
        prot_input_ids=torch.tensor([[1, 2, 0], [3, 4, 0]]),
        prot_attention_mask=torch.tensor([[1, 1, 0], [1, 1, 0]]),
        max_length=5,
    )
    assert generated.shape[0] == 2
    assert generated.shape[1] <= 5


def test_legacy_gpt2_state_migration_loads_strictly(monkeypatch):
    from conftest import DummyBatchTokenizer, DummyProteinEncoder

    monkeypatch.setattr(
        model_module,
        "get_protein_encoder",
        lambda model_name, model_id, active: DummyProteinEncoder(hidden_size=8),
    )
    config = _config(DummyBatchTokenizer())
    config.update(
        {
            "decoder_type": "gpt2",
            "n_layer": 1,
            "n_head": 2,
            "n_emb": 8,
        }
    )
    model = model_module.create_prot2mol_model(config)
    legacy_state = {}
    for key, value in model.state_dict().items():
        if key.startswith("protein_encoder.model."):
            key = "protein_encoder.encoder_model." + key.removeprefix(
                "protein_encoder.model."
            )
        legacy_state[key] = value.clone()
    legacy_state["pchembl_head.legacy.weight"] = torch.ones(1)
    legacy_state["lm_weight"] = torch.ones(())
    legacy_state["pchembl_weight"] = torch.ones(())

    migrated = hf_utils.prepare_prot2mol_state_dict(
        legacy_state,
        decoder_type="gpt2",
    )
    result = model.load_state_dict(migrated, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_inference_loader_strict_fallback_is_explicit(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "pytorch_model.bin").write_bytes(b"stub")

    class DummyModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config_seen = config
            self.strict_values = []

        def load_state_dict(self, state, strict=True):
            self.strict_values.append(strict)
            if strict:
                raise RuntimeError("mismatch")
            return [], []

    monkeypatch.setattr(hf_utils.torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr("prot2mol.core.model.Prot2MolModel", DummyModel)
    loaded = hf_utils.load_prot2mol_inference_model(
        model_path=str(checkpoint),
        device=torch.device("cpu"),
        mol_tokenizer=SimpleNamespace(),
        prot_emb_model="esm2",
        max_mol_len=8,
        prot_max_length=16,
        strict=True,
        allow_strict_fallback=True,
    )
    assert loaded.strict_values == [True, False]
    assert loaded.config_seen["initialize_protein_encoder_from_pretrained"] is False
    assert loaded.config_seen["initialize_decoder_from_pretrained"] is False


def test_saved_config_uses_new_architecture_fields_and_ignores_legacy_affinity(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"prot_emb_model":"esm2","decoder_model_id":"zjunlp/MolGen-large",'
        '"conditioning_dropout":0.2,"pchembl_tf_hidden_dim":640}',
        encoding="utf-8",
    )
    loaded = hf_utils.load_saved_model_config(str(tmp_path))
    assert loaded["decoder_type"] == "molgen"
    assert loaded["decoder_model_id"] == "zjunlp/MolGen-large"
    assert loaded["conditioning_dropout"] == 0.2
    assert "pchembl_tf_hidden_dim" not in loaded


def test_legacy_gpt2_config_is_inferred_and_state_is_migrated(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"prot_emb_model":"prot_t5","n_layer":2,"n_head":8,"n_emb":1024}',
        encoding="utf-8",
    )
    loaded = hf_utils.load_saved_model_config(str(tmp_path))
    assert loaded["decoder_type"] == "gpt2"

    state = {
        "protein_encoder.encoder_model.weight": torch.ones(1),
        "protein_encoder.encoder_model.embeddings.position_embeddings.weight": (
            torch.ones(4, 2)
        ),
        "protein_encoder.encoder_model.encoder.layer.0.attention.self."
        "rotary_embeddings.inv_freq": torch.tensor([1.0, 0.5]),
        "protein_encoder.encoder_model.encoder.layer.1.attention.self."
        "rotary_embeddings.inv_freq": torch.tensor([1.0, 0.5]),
        "molecule_decoder.transformer.wte.weight": torch.ones(1),
        "pchembl_head.layer.weight": torch.ones(1),
        "lm_weight": torch.ones(1),
    }
    migrated = hf_utils.prepare_prot2mol_state_dict(state, decoder_type="gpt2")
    assert set(migrated) == {
        "protein_encoder.model.weight",
        "protein_encoder.model.rotary_embeddings.inv_freq",
        "molecule_decoder.transformer.wte.weight",
    }
    assert torch.equal(
        migrated["protein_encoder.model.rotary_embeddings.inv_freq"],
        torch.tensor([1.0, 0.5]),
    )


def test_migrated_esm_rotary_buffer_adapts_to_per_layer_layout():
    shared_key = "protein_encoder.model.rotary_embeddings.inv_freq"
    first_layer = (
        "protein_encoder.model.encoder.layer.0.attention.self."
        "rotary_embeddings.inv_freq"
    )
    second_layer = (
        "protein_encoder.model.encoder.layer.1.attention.self."
        "rotary_embeddings.inv_freq"
    )
    value = torch.tensor([1.0, 0.5])

    aligned = hf_utils.align_prot2mol_state_dict_to_model(
        {shared_key: value, "molecule_decoder.weight": torch.ones(1)},
        [first_layer, second_layer, "molecule_decoder.weight"],
    )

    assert shared_key not in aligned
    assert torch.equal(aligned[first_layer], value)
    assert torch.equal(aligned[second_layer], value)
