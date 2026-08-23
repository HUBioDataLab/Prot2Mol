import json
from types import MethodType, SimpleNamespace

import pandas as pd
import pytest
import torch

from prot2mol.core import model as model_module
from prot2mol.training import grpo_train


class MoleculeTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 16

    def __len__(self):
        return self.vocab_size

    def batch_decode(self, rows, **kwargs):
        del kwargs
        tokens = {3: "[C]", 4: "[O]", 5: "[N]", 6: "[F]"}
        return ["".join(tokens.get(token, "") for token in row) for row in rows]


class ProteinTokenizer:
    padding_side = "right"

    def __call__(self, sequences, *, max_length, **kwargs):
        del kwargs
        rows = []
        masks = []
        for sequence in sequences:
            tokens = [3 + (ord(value) % 4) for value in sequence][:max_length]
            mask = [1] * len(tokens)
            padding = max_length - len(tokens)
            rows.append(tokens + [0] * padding)
            masks.append(mask + [0] * padding)
        return {
            "input_ids": torch.tensor(rows),
            "attention_mask": torch.tensor(masks),
        }


class FakeFusionDTI(torch.nn.Module):
    protein_representation = "structure_aware"
    molecule_representation = "selfies"

    def __init__(self):
        super().__init__()
        self.register_buffer("anchor", torch.zeros(()))

    def forward(self, proteins, molecules):
        assert all(value in {"AaCa", "GaTa"} for value in proteins)
        values = [0.15 + 0.1 * (len(value) % 7) for value in molecules]
        return torch.tensor(values).clamp(0.0, 1.0)


class FakeWandbRun:
    id = "test-run"

    def __init__(self):
        self.logs = []
        self.finished = False

    def log(self, values, step=None):
        self.logs.append((step, values))

    def finish(self):
        self.finished = True


def _write_random_split(tmp_path):
    train = tmp_path / "train.parquet"
    validation = tmp_path / "val.parquet"
    structures = tmp_path / "structures.parquet"
    pd.DataFrame(
        {
            "protein_accession": ["P1", "P1", "P2"],
            "protein_sequence": ["AC", "AC", "GT"],
            "smiles": ["C", "CC", "O"],
            "binary_label": [1, 0, 1],
        }
    ).to_parquet(train, index=False)
    pd.DataFrame(
        {
            "protein_accession": ["P1", "P1", "P2", "P2"],
            "protein_sequence": ["AC", "AC", "GT", "GT"],
            "smiles": ["C", "CC", "O", "CO"],
            "binary_label": [1, 1, 1, 1],
        }
    ).to_parquet(validation, index=False)
    pd.DataFrame(
        {
            "protein_accession": ["P1", "P2"],
            "structure_aware_sequence": ["AaCa", "GaTa"],
        }
    ).to_parquet(structures, index=False)
    return train, validation, structures


def _tiny_policy(monkeypatch):
    from conftest import DummyProteinEncoder

    monkeypatch.setattr(
        model_module,
        "get_protein_encoder",
        lambda model_name, model_id, active: DummyProteinEncoder(hidden_size=8),
    )
    tokenizer = MoleculeTokenizer()
    policy = model_module.create_prot2mol_model(
        {
            "prot_emb_model": "esm2",
            "protein_model_id": "dummy/protein",
            "decoder_type": "gpt2",
            "decoder_model_id": "dummy/molecule",
            "n_layer": 1,
            "n_head": 2,
            "n_emb": 8,
            "max_mol_len": 5,
            "conditioning_dropout": 0.0,
            "train_encoder_model": False,
            "train_projection_model": False,
            "train_decoder_model": False,
            "mol_tokenizer": tokenizer,
        }
    )
    patterns = torch.tensor(
        [
            [1, 3, 2, 0, 0],
            [1, 4, 2, 0, 0],
            [1, 5, 2, 0, 0],
            [1, 6, 2, 0, 0],
            [1, 3, 4, 2, 0],
            [1, 4, 4, 2, 0],
            [1, 3, 3, 2, 0],
            [1, 5, 3, 2, 0],
        ]
    )

    def deterministic_generate(self, protein_embeddings, prot_attention_mask, **kwargs):
        del prot_attention_mask, kwargs
        count = protein_embeddings.size(0)
        repeats = (count + len(patterns) - 1) // len(patterns)
        return patterns.repeat(repeats, 1)[:count].to(protein_embeddings.device)

    policy.generate_from_protein_embeddings = MethodType(deterministic_generate, policy)
    return policy, tokenizer


def test_unique_proteins_require_and_join_structure_aware_sequences(tmp_path):
    train, _, structures = _write_random_split(tmp_path)

    proteins = grpo_train.load_unique_training_proteins(
        train,
        structure_aware_path=structures,
    )

    assert proteins == [
        grpo_train.ProteinRecord("P1", "AC", "AaCa"),
        grpo_train.ProteinRecord("P2", "GT", "GaTa"),
    ]
    with pytest.raises(ValueError, match="requires a separate SaProt/Foldseek"):
        grpo_train.load_unique_training_proteins(train)


def test_unique_proteins_can_validate_only_fixed_training_panel(tmp_path):
    train, _, structures = _write_random_split(tmp_path)
    structure_frame = pd.read_parquet(structures)
    structure_frame.loc[structure_frame["protein_accession"] == "P2"].to_parquet(
        structures, index=False
    )

    proteins = grpo_train.load_unique_training_proteins(
        train,
        structure_aware_path=structures,
        required_protein_ids=["P2"],
    )

    assert proteins == [grpo_train.ProteinRecord("P2", "GT", "GaTa")]

    with pytest.raises(ValueError, match="Required protein IDs"):
        grpo_train.load_unique_training_proteins(
            train,
            structure_aware_path=structures,
            required_protein_ids=["P3"],
        )


def test_structure_mapping_rejects_plain_amino_acid_values(tmp_path):
    train, _, structures = _write_random_split(tmp_path)
    frame = pd.read_parquet(structures)
    frame["structure_aware_sequence"] = ["AC", "GT"]
    frame.to_parquet(structures, index=False)

    with pytest.raises(ValueError, match="refusing to pass a plain sequence"):
        grpo_train.load_unique_training_proteins(
            train,
            structure_aware_path=structures,
        )


def test_full_grpo_runner_logs_train_eval_chemistry_fcd_and_saves_resume_state(
    tmp_path,
    monkeypatch,
):
    train, validation, structures = _write_random_split(tmp_path)
    generator = tmp_path / "generator"
    generator.mkdir()
    output = tmp_path / "output"
    policy, molecule_tokenizer = _tiny_policy(monkeypatch)
    resumed_policy, _ = _tiny_policy(monkeypatch)
    policies = iter((policy, resumed_policy))
    fake_runs = []
    wandb_init_calls = []

    monkeypatch.setattr(
        grpo_train,
        "load_molgen_tokenizer",
        lambda **kwargs: molecule_tokenizer,
    )
    monkeypatch.setattr(
        grpo_train,
        "get_protein_tokenizer",
        lambda *args, **kwargs: ProteinTokenizer(),
    )
    monkeypatch.setattr(
        grpo_train,
        "load_prot2mol_inference_model",
        lambda **kwargs: next(policies),
    )
    monkeypatch.setattr(
        grpo_train.FusionDTIActivityScorer,
        "from_pretrained",
        lambda **kwargs: FakeFusionDTI(),
    )
    def fake_wandb_init(**kwargs):
        wandb_init_calls.append(kwargs)
        run = FakeWandbRun()
        fake_runs.append(run)
        return run

    monkeypatch.setattr(grpo_train.wandb, "init", fake_wandb_init)
    monkeypatch.setattr(
        grpo_train.wandb,
        "Table",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    arguments = [
            "--train_parquet_path",
            str(train),
            "--validation_parquet_path",
            str(validation),
            "--structure_aware_path",
            str(structures),
            "--generator_checkpoint",
            str(generator),
            "--output_dir",
            str(output),
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--max_mol_len",
            "5",
            "--prot_max_length",
            "4",
            "--n_layer",
            "1",
            "--n_head",
            "2",
            "--n_emb",
            "8",
            "--max_steps",
            "2",
            "--no-train_on_evaluation_panel_only",
            "--eval_proteins",
            "1",
            "--wandb_target_metric_ids",
            "P1",
            "--eval_samples_per_protein",
            "8",
            "--min_fcd_reference_actives",
            "2",
            "--eval_steps",
            "1",
            "--save_steps",
            "1",
            "--wandb_mode",
            "disabled",
        ]
    config = grpo_train.parse_arguments(arguments)
    runner = grpo_train.GRPOTrainingRun(config)
    runner._fcd = lambda ref, gen: float(len(ref) + len(gen))

    summary = runner.run()

    assert summary["global_step"] == 2
    assert summary["proteins_seen"] == 2
    assert summary["unique_training_proteins"] == 2
    assert (output / "checkpoint-2" / "trainer_state.pt").exists()
    assert (output / "final" / "pytorch_model.bin").exists()
    assert (output / "cohort.parquet").exists()
    assert (output / "evaluation" / "start" / "generated_molecules.parquet").exists()
    assert (output / "evaluation" / "start" / "per_protein_metrics.parquet").exists()
    assert (output / "evaluation" / "end" / "generated_molecules.parquet").exists()
    assert (output / "evaluation" / "end" / "per_protein_metrics.parquet").exists()
    endpoint_rows = pd.read_parquet(
        output / "evaluation" / "end" / "generated_molecules.parquet"
    )
    assert len(endpoint_rows) == 8
    assert {
        "generated_selfies",
        "generated_smiles",
        "terminated",
        "chemically_valid",
        "reward_eligible",
        "activity_reward",
        "qed",
        "sas",
        "logp",
    }.issubset(endpoint_rows.columns)
    assert json.loads((output / "training_summary.json").read_text())["global_step"] == 2
    logged = [values for _, values in fake_runs[0].logs]
    assert any("grpo/reward_mean" in values for values in logged)
    assert any("grpo/qed_mean" in values for values in logged)
    assert any("grpo/sas_mean" in values for values in logged)
    assert any("grpo/valid_unique_fraction" in values for values in logged)
    assert any("eval/fcd_macro" in values for values in logged)
    assert any("eval/per_protein" in values for values in logged)
    assert any("eval/targets/P1/fcd" in values for values in logged)
    assert fake_runs[0].finished is True

    resume_config = grpo_train.parse_arguments(
        [
            *arguments,
            "--resume_from_checkpoint",
            str(output / "checkpoint-1"),
        ]
    )
    resumed_runner = grpo_train.GRPOTrainingRun(resume_config)
    resumed_runner._fcd = lambda ref, gen: float(len(ref) + len(gen))

    resumed_summary = resumed_runner.run()

    assert resumed_summary["global_step"] == 2
    assert resumed_summary["proteins_seen"] == 2
    assert wandb_init_calls[1]["id"] == "test-run"
    assert wandb_init_calls[1]["resume"] == "must"
    assert fake_runs[1].finished is True
