import torch
from datasets import Dataset
from transformers import BartConfig, BartForConditionalGeneration, TrainingArguments

from prot2mol.core import model as model_module
from prot2mol.training.trainer import Prot2MolTrainer


def _tiny_model(monkeypatch):
    from conftest import DummyBatchTokenizer, DummyProteinEncoder

    monkeypatch.setattr(
        model_module,
        "get_protein_encoder",
        lambda model_name, model_id, active: DummyProteinEncoder(hidden_size=8),
    )
    monkeypatch.setattr(
        model_module.AutoModelForSeq2SeqLM,
        "from_pretrained",
        lambda path: BartForConditionalGeneration(
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
        ),
    )
    return model_module.create_prot2mol_model(
        {
            "prot_emb_model": "esm2",
            "protein_model_id": "dummy/protein",
            "decoder_type": "molgen",
            "decoder_model_id": "dummy/molgen",
            "conditioning_dropout": 0.0,
            "max_mol_len": 4,
            "prot_max_length": 4,
            "train_encoder_model": True,
            "train_projection_model": True,
            "train_decoder_model": True,
            "mol_tokenizer": DummyBatchTokenizer(),
        }
    )


def _dataset():
    return Dataset.from_dict(
        {
            "protein_sequence": ["AAAA", "BBBB", "CCCC", "DDDD"],
            "smiles": ["C", "O", "N", "CC"],
            "prot_input_ids": [[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [2, 4, 6, 0]],
            "prot_attention_mask": [[1, 1, 1, 0]] * 4,
            "labels": [[1, 3, 2, -100], [1, 4, 2, -100], [1, 5, 2, -100], [1, 6, 2, -100]],
        }
    )


def _arguments(output_dir, max_steps, overwrite):
    return TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="steps",
        save_steps=1,
        logging_steps=1,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        remove_unused_columns=False,
        use_cpu=True,
    )


def test_train_evaluate_generate_checkpoint_and_resume(tmp_path, monkeypatch):
    data = _dataset()
    output = tmp_path / "run"
    trainer = Prot2MolTrainer(
        model=_tiny_model(monkeypatch),
        args=_arguments(output, max_steps=2, overwrite=True),
        train_dataset=data,
        eval_dataset=data,
        compute_generation_metrics=lambda: {"gen_smoke": 1.0},
    )
    result = trainer.train()
    metrics = trainer.evaluate(run_generation_metrics=True)
    trainer.save_model(str(output / "final"))

    assert result.global_step == 2
    assert torch.isfinite(torch.tensor(metrics["eval_loss"]))
    assert metrics["eval_gen_smoke"] == 1.0
    assert (output / "checkpoint-2" / "pytorch_model.bin").exists()
    assert (output / "checkpoint-2" / "config.json").exists()
    assert (output / "final" / "pytorch_model.bin").exists()

    resumed = Prot2MolTrainer(
        model=_tiny_model(monkeypatch),
        args=_arguments(output, max_steps=3, overwrite=False),
        train_dataset=data,
        eval_dataset=data,
    )
    resumed_result = resumed.train(resume_from_checkpoint=str(output / "checkpoint-2"))
    assert resumed_result.global_step == 3
