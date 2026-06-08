from types import SimpleNamespace

import pytest
import torch

from prot2mol.training.entry import create_run_name, parse_arguments
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


def test_parse_arguments_defaults_to_single_gpu():
    args = parse_arguments(argv=[])

    assert args.training_mode == "single_gpu"


def test_create_run_name_is_compact_and_stable():
    config = SimpleNamespace(
        prot_emb_model="saprot",
        training_stage="pchembl_only",
        train_encoder_model=False,
        train_decoder_model=False,
        train_pchembl_head=True,
        stop_pchembl_gradients=True,
        pchembl_tf_hidden_dim=768,
        pchembl_tf_num_heads=8,
        pchembl_tf_group_size=1,
        pchembl_tf_agg_mode="mean",
        n_layer=1,
        n_head=16,
        n_emb=1024,
        max_mol_len=200,
        prot_max_length=1000,
        learning_rate=1e-5,
        train_batch_size=4,
        training_mode="single_gpu",
        run_name_suffix="Phase_IIIa",
    )

    run_name_a = create_run_name(
        config,
        "prot_comp_set_pchembl_6_protlen_1000_human_False",
    )
    run_name_b = create_run_name(
        config,
        "prot_comp_set_pchembl_6_protlen_1000_human_False",
    )

    assert run_name_a == run_name_b
    assert len(run_name_a) <= 200
    assert "Phase_IIIa" in run_name_a
    assert "stg-pchembl_only" in run_name_a


def test_eval_component_logs_all_reduce_with_zero_local_count(monkeypatch):
    trainer = _make_light_trainer(pchembl_only=True)
    trainer._reset_eval_component_accumulator()
    trainer.args = SimpleNamespace(local_rank=0, device=torch.device("cpu"))
    calls = []

    def fake_all_reduce(tensor, op=None):
        calls.append((tensor.clone(), op))
        tensor[1] = 2.0
        tensor[3] = 2.0
        tensor[4] = 1.0

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    logs = trainer._consume_eval_component_logs("eval")

    assert calls
    assert logs["eval_pchembl_loss"] == pytest.approx(2.0)
    assert logs["eval_total_loss"] == pytest.approx(2.0)


def test_get_pchembl_predictions_gathers_with_zero_local_predictions(monkeypatch):
    trainer = _make_light_trainer(pchembl_only=True)
    trainer.args = SimpleNamespace(local_rank=0, device=torch.device("cpu"))
    trainer.pchembl_predictions_list = []
    trainer.pchembl_targets_list = []
    trainer.pchembl_group_ids_list = []
    calls = {"all_gather": 0, "all_reduce": 0}

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def fake_all_reduce(tensor, op=None):
        calls["all_reduce"] += 1

    def fake_all_gather(outputs, tensor):
        calls["all_gather"] += 1
        if tensor.numel() == 1:
            outputs[0].copy_(torch.tensor([0], dtype=tensor.dtype, device=tensor.device))
            outputs[1].copy_(torch.tensor([2], dtype=tensor.dtype, device=tensor.device))
            return

        outputs[0].zero_()
        outputs[1].copy_(torch.tensor([0.25, 0.75], dtype=tensor.dtype, device=tensor.device))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

    predictions, targets, group_ids = trainer.get_pchembl_predictions()

    assert calls == {"all_gather": 3, "all_reduce": 1}
    assert predictions.tolist() == pytest.approx([0.25, 0.75])
    assert targets.tolist() == pytest.approx([0.25, 0.75])
    assert group_ids is None
