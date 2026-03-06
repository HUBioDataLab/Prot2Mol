from types import SimpleNamespace

import torch

from prot2mol.training.trainer import GPT2_w_crs_attn_Trainer
from prot2mol.training.training_runner import TrainingRunner


def _dummy_inputs(batch=3, seq=4):
    return {
        "mol_input_ids": torch.ones(batch, seq, dtype=torch.long),
        "prot_input_ids": torch.ones(batch, seq, dtype=torch.long),
        "prot_attention_mask": torch.ones(batch, seq, dtype=torch.long),
        "labels": torch.ones(batch, seq, dtype=torch.long),
        "pchembl_values": torch.linspace(0.0, 1.0, steps=batch),
        "group_id": torch.tensor([0, 0, 1], dtype=torch.long),
    }


def _make_light_trainer(pchembl_only=False):
    tr = GPT2_w_crs_attn_Trainer.__new__(GPT2_w_crs_attn_Trainer)
    tr.training_stage = "pchembl_only" if pchembl_only else "multitask"
    tr.pchembl_pair_weight = 1.0
    tr._train_component_sums = {"lm_loss": 0.0, "pchembl_loss": 0.0, "pair_loss": 0.0, "total_loss": 0.0}
    tr._train_component_count = 0
    return tr


class _DummyModel:
    def __init__(self):
        self._config = {"pchembl_huber_delta": 1.0}
        self.lm_weight = torch.tensor(1.0)
        self.pchembl_weight = torch.tensor(1.0)

    def __call__(self, **kwargs):
        batch = kwargs["mol_input_ids"].shape[0]
        seq = kwargs["mol_input_ids"].shape[1]
        return {
            "lm_loss": torch.tensor(0.4),
            "logits": torch.zeros(batch, seq, 5),
            "pchembl_predictions": torch.linspace(0.1, 0.9, steps=batch),
        }


def test_build_model_inputs_defaults_train_lm_tensor():
    trainer = _make_light_trainer(pchembl_only=False)
    inputs = _dummy_inputs(batch=2)
    inputs.pop("labels")

    built = trainer._build_model_inputs(inputs)

    assert built["train_lm"].dtype == torch.bool
    assert built["train_lm"].shape[0] == 2
    assert built["labels"].shape == inputs["mol_input_ids"].shape


def test_compute_loss_full_multitask():
    trainer = _make_light_trainer(pchembl_only=False)
    model = _DummyModel()

    loss, outputs = trainer.compute_loss(model, _dummy_inputs(), return_outputs=True)

    assert torch.is_tensor(loss)
    assert loss.item() > 0
    assert "pchembl_loss" in outputs
    assert "pchembl_pair_loss" in outputs
    assert "lm_loss" in outputs


def test_compute_loss_pchembl_only_mode_ignores_lm_component():
    trainer = _make_light_trainer(pchembl_only=True)
    model = _DummyModel()

    loss, outputs = trainer.compute_loss(model, _dummy_inputs(), return_outputs=True)

    assert torch.is_tensor(loss)
    assert loss.item() > 0
    assert "pchembl_loss" in outputs


def test_compute_loss_raises_when_no_loss_terms():
    trainer = _make_light_trainer(pchembl_only=False)

    class _NoLossModel:
        def __init__(self):
            self._config = {"pchembl_huber_delta": 1.0}
            self.lm_weight = torch.tensor(1.0)
            self.pchembl_weight = torch.tensor(1.0)

        def __call__(self, **kwargs):
            return {"logits": torch.zeros(1, 1, 5)}

    try:
        trainer.compute_loss(_NoLossModel(), _dummy_inputs(), return_outputs=False)
        assert False, "Expected RuntimeError when no loss is available"
    except RuntimeError as exc:
        assert "Loss is None" in str(exc)


def test_training_runner_mode_detection_and_args():
    runner = TrainingRunner(local_rank=-1, global_rank=0)

    args = runner._create_training_args(
        run_name="run",
        output_dir="/tmp/out",
        training_config={
            "resume_from_checkpoint": None,
            "epochs": 2,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "train_batch_size": 3,
            "valid_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "dataloader_num_workers": 0,
        },
    )

    assert args.per_device_train_batch_size == 3
    assert args.per_device_eval_batch_size == 4
    assert args.gradient_accumulation_steps == 2
    assert args.output_dir == "/tmp/out"
    assert args.ddp_backend is None
    eval_field = getattr(args, "evaluation_strategy", getattr(args, "eval_strategy", None))
    assert eval_field is not None
    assert str(eval_field).lower().endswith("epoch")
