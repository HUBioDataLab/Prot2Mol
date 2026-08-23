from types import SimpleNamespace

import pytest
import torch

from prot2mol.training.entry import create_run_name, parse_arguments
from prot2mol.training.trainer import (
    Prot2MolTrainer,
    generation_data_collator,
)
from prot2mol.training.training_runner import TrainingRunner


def _dummy_inputs(batch=3, seq=4):
    return {
        "prot_input_ids": torch.ones(batch, seq, dtype=torch.long),
        "prot_attention_mask": torch.ones(batch, seq, dtype=torch.long),
        "labels": torch.ones(batch, seq, dtype=torch.long),
    }


def _light_trainer():
    trainer = Prot2MolTrainer.__new__(Prot2MolTrainer)
    trainer._lm_loss_sum = 0.0
    trainer._lm_loss_count = 0
    return trainer


class _DummyModel(torch.nn.Module):
    def forward(self, **kwargs):
        batch, seq = kwargs["labels"].shape
        loss = torch.tensor(0.4, requires_grad=True)
        return {"loss": loss, "lm_loss": loss, "logits": torch.zeros(batch, seq, 5)}


def test_generation_collator_ignores_raw_string_metadata():
    features = [
        {
            **{key: value[index] for key, value in _dummy_inputs(batch=2).items()},
            "protein_sequence": sequence,
            "smiles": smiles,
        }
        for index, (sequence, smiles) in enumerate((("MKT", "C"), ("GGA", "O")))
    ]
    batch = generation_data_collator(features)
    assert set(batch) == set(_dummy_inputs())
    assert batch["labels"].shape == (2, 4)


def test_compute_loss_is_language_model_loss_only():
    trainer = _light_trainer()
    model = _DummyModel()
    model.train()
    loss, outputs = trainer.compute_loss(model, _dummy_inputs(), return_outputs=True)
    assert loss.item() == pytest.approx(0.4)
    assert set(outputs) == {"loss", "lm_loss", "logits"}
    assert trainer._lm_loss_count == 1


def test_compute_loss_rejects_model_without_language_model_loss():
    class _NoLossModel(torch.nn.Module):
        def forward(self, **kwargs):
            return {"logits": torch.zeros(1, 1, 5)}

    with pytest.raises(RuntimeError, match="language-modeling loss"):
        _light_trainer().compute_loss(_NoLossModel(), _dummy_inputs())


def test_training_runner_precision_batches_and_clipping():
    runner = TrainingRunner(local_rank=-1, global_rank=0)
    args = runner._create_training_args(
        "run",
        "/tmp/out",
        {
            "resume_from_checkpoint": None,
            "epochs": 2,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "train_batch_size": 3,
            "valid_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "dataloader_num_workers": 0,
            "precision": "fp32",
            "max_grad_norm": 0.5,
            "logging_steps": 20,
        },
    )
    assert args.per_device_train_batch_size == 3
    assert args.per_device_eval_batch_size == 4
    assert args.gradient_accumulation_steps == 2
    assert args.max_grad_norm == 0.5
    assert not args.fp16 and not args.bf16


def test_parse_arguments_defaults_match_generation_contract():
    args = parse_arguments(argv=[])
    assert args.training_mode == "single_gpu"
    assert args.prot_emb_model == "esm2"
    assert args.decoder_type == "gpt2"
    assert args.decoder_model_id == "zjunlp/MolGen-large"
    assert args.n_layer == 1
    assert args.n_head == 16
    assert args.n_emb is None
    assert args.train_projection_model is True
    assert not hasattr(args, "eval_split")


def test_training_arguments_reject_nonpositive_batch_size():
    with pytest.raises(ValueError, match="must be positive"):
        parse_arguments(argv=["--train_batch_size", "0"])


def test_training_arguments_require_a_trainable_component():
    with pytest.raises(ValueError, match="At least one"):
        parse_arguments(
            argv=[
                "--no-train_encoder_model",
                "--no-train_projection_model",
                "--no-train_decoder_model",
            ]
        )


def test_create_run_name_is_compact_stable_and_has_no_affinity_stage():
    config = SimpleNamespace(
        prot_emb_model="esm2",
        decoder_type="gpt2",
        decoder_model_id="zjunlp/MolGen-large",
        n_layer=1,
        n_head=16,
        n_emb=1280,
        train_encoder_model=False,
        train_decoder_model=True,
        max_mol_len=256,
        prot_max_length=1024,
        learning_rate=1e-5,
        train_batch_size=4,
        training_mode="single_gpu",
        run_name_suffix="Phase_IIIa",
    )
    first = create_run_name(config, "generation_dataset")
    assert first == create_run_name(config, "generation_dataset")
    assert len(first) <= 200
    assert "Phase_IIIa" in first
    assert "pchembl" not in first.lower()
