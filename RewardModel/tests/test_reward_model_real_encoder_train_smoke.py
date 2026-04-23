from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_model.model import RewardModel, RewardModelConfig


RUN_REAL_ENCODER_TRAIN_SMOKE_ENV = "RUN_REAL_ENCODER_TRAIN_SMOKE"

DEFAULT_PROTEIN_MODEL = "facebook/esm2_t12_35M_UR50D"
DEFAULT_MOLECULE_MODEL = "HUBioDataLab/SELFormer"
CURATED_ROWS_PATH = ROOT / "reward_model" / "data" / "curated" / "chembl_assay_rows.csv"


def _print_header(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def _shape_of(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(tensor.shape)


def _mask_lengths(mask: torch.Tensor) -> list[int]:
    return [int(value) for value in mask.sum(dim=1).detach().cpu().tolist()]


def _first_grad_sum(module: torch.nn.Module) -> float:
    for parameter in module.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            return float(parameter.grad.detach().abs().sum().item())
    return 0.0


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        total += float(torch.sum(grad * grad).item())
    return total ** 0.5


def _parameter_delta(before: torch.Tensor, after: torch.Tensor) -> float:
    return float((after.detach().cpu() - before.detach().cpu()).abs().sum().item())


def _peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device=device) / (1024 ** 2))


def _load_balanced_rows_from_curated_csv(
    curated_csv_path: Path = CURATED_ROWS_PATH,
    min_positive: int = 2,
    min_negative: int = 2,
    max_scan_rows: int = 5000,
) -> tuple[list[dict[str, str]], str, int]:
    if not curated_csv_path.exists():
        raise FileNotFoundError(f"Curated CSV not found at {curated_csv_path}")

    grouped_rows: dict[str, dict[str, list[dict[str, str]]]] = {}
    with curated_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=1):
            target_id = row["target_chembl_id"]
            label_value = int(float(row["activity_label"]))
            group = grouped_rows.setdefault(target_id, {"positive": [], "negative": []})
            bucket = "positive" if label_value > 0 else "negative"
            group[bucket].append(row)

            if len(group["positive"]) >= min_positive and len(group["negative"]) >= min_negative:
                selected_rows = group["negative"][:min_negative] + group["positive"][:min_positive]
                return selected_rows, target_id, line_number

            if line_number >= max_scan_rows:
                break

    raise ValueError(
        f"Could not find a target with at least {min_negative} negative and {min_positive} positive rows "
        f"within the first {max_scan_rows} rows of {curated_csv_path}"
    )


def _build_train_batch_from_curated_csv(
    curated_csv_path: Path = CURATED_ROWS_PATH,
) -> tuple[list[dict[str, str]], str, int, list[str], list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, target_id, scanned_rows = _load_balanced_rows_from_curated_csv(curated_csv_path=curated_csv_path)

    protein_sequences = [row["protein_sequence"] for row in rows]
    molecule_sequences = [row["compound_selfies"] for row in rows]
    activity_labels = torch.tensor([float(row["activity_label"]) for row in rows], dtype=torch.float32)

    num_negative = sum(float(row["activity_label"]) <= 0.0 for row in rows)
    num_positive = len(rows) - num_negative
    pair_count = min(num_negative, num_positive)

    positive_indices = torch.arange(num_negative, num_negative + pair_count, dtype=torch.long)
    negative_indices = torch.arange(0, pair_count, dtype=torch.long)
    return (
        rows,
        target_id,
        scanned_rows,
        protein_sequences,
        molecule_sequences,
        activity_labels,
        positive_indices,
        negative_indices,
    )


def run_real_encoder_train_smoke(
    curated_csv_path: Path = CURATED_ROWS_PATH,
    protein_model_name_or_path: str = DEFAULT_PROTEIN_MODEL,
    molecule_model_name_or_path: str = DEFAULT_MOLECULE_MODEL,
    protein_max_length: int = 512,
    molecule_max_length: int = 256,
    fusion_hidden_dim: int = 256,
    fusion_num_heads: int = 8,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    num_steps: int = 5,
) -> dict[str, object]:
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    (
        rows,
        target_id,
        scanned_rows,
        protein_sequences,
        molecule_sequences,
        activity_labels,
        positive_indices,
        negative_indices,
    ) = _build_train_batch_from_curated_csv(curated_csv_path=curated_csv_path)

    _print_header("Reward Model Real Encoder Train Smoke")
    print(f"device: {device}")
    print(f"curated_csv_path: {curated_csv_path}")
    print(f"lines_scanned_until_batch_found: {scanned_rows}")
    print(f"selected_target_chembl_id: {target_id}")
    print(f"protein_model: {protein_model_name_or_path}")
    print(f"molecule_model: {molecule_model_name_or_path}")
    print(f"num_examples: {len(rows)}")
    print(f"num_steps: {num_steps}")
    print(f"learning_rate: {learning_rate}")
    print(f"weight_decay: {weight_decay}")
    for index, row in enumerate(rows):
        print(
            "row"
            f" {index}: label={row['activity_label']}"
            f" pchembl={row['pchembl_value']}"
            f" assay={row['assay_chembl_id']}"
            f" molecule={row['molecule_chembl_id']}"
            f" protein_len={len(row['protein_sequence'])}"
            f" selfies_len={len(row['compound_selfies'])}"
        )
    print(f"positive_indices: {positive_indices.tolist()}")
    print(f"negative_indices: {negative_indices.tolist()}")

    _print_header("Building Model")
    config = RewardModelConfig(
        protein_model_name_or_path=protein_model_name_or_path,
        molecule_model_name_or_path=molecule_model_name_or_path,
        protein_max_length=protein_max_length,
        molecule_max_length=molecule_max_length,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_num_heads=fusion_num_heads,
        dropout=0.0,
        pooling_type="mean",
    )
    model = RewardModel(config=config)
    model.to(device)
    model.train()

    print(f"inferred_protein_hidden_size: {model.config.protein_hidden_size}")
    print(f"inferred_molecule_hidden_size: {model.config.molecule_hidden_size}")
    print(f"fusion_hidden_dim: {model.config.fusion_hidden_dim}")
    print(f"num_parameters: {model.num_parameters():,}")
    print(f"num_trainable_parameters: {model.num_trainable_parameters():,}")

    _print_header("Tokenization")
    protein_batch = model.tokenize_proteins(protein_sequences, device=device)
    molecule_batch = model.tokenize_molecules(molecule_sequences, device=device)

    print(f"protein_input_ids_shape: {_shape_of(protein_batch['input_ids'])}")
    print(f"protein_attention_mask_shape: {_shape_of(protein_batch['attention_mask'])}")
    print(f"protein_token_counts: {_mask_lengths(protein_batch['attention_mask'])}")
    print(f"molecule_input_ids_shape: {_shape_of(molecule_batch['input_ids'])}")
    print(f"molecule_attention_mask_shape: {_shape_of(molecule_batch['attention_mask'])}")
    print(f"molecule_token_counts: {_mask_lengths(molecule_batch['attention_mask'])}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    activity_labels = activity_labels.to(device)
    positive_indices = positive_indices.to(device)
    negative_indices = negative_indices.to(device)

    protein_projection_before = model.protein_projection.weight.detach().cpu().clone()
    molecule_projection_before = model.molecule_projection.weight.detach().cpu().clone()
    ranking_head_before = model.ranking_head.fc1.weight.detach().cpu().clone()
    classification_head_before = model.classification_head.fc1.weight.detach().cpu().clone()

    loss_history: list[float] = []
    pair_loss_history: list[float] = []
    classification_loss_history: list[float] = []
    grad_norm_history: list[float] = []

    _print_header("Training Loop")
    for step in range(1, num_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            protein_input_ids=protein_batch["input_ids"],
            protein_attention_mask=protein_batch["attention_mask"],
            molecule_input_ids=molecule_batch["input_ids"],
            molecule_attention_mask=molecule_batch["attention_mask"],
            activity_labels=activity_labels,
            positive_indices=positive_indices,
            negative_indices=negative_indices,
            return_token_embeddings=False,
        )

        assert outputs.loss is not None
        loss_value = float(outputs.loss.detach().cpu().item())
        pair_loss_value = float(outputs.pair_loss.detach().cpu().item())
        classification_loss_value = float(outputs.classification_loss.detach().cpu().item())

        outputs.loss.backward()

        grad_norm_value = _grad_norm(model)
        protein_encoder_grad = _first_grad_sum(model.protein_encoder)
        molecule_encoder_grad = _first_grad_sum(model.molecule_encoder)
        protein_projection_grad = float(model.protein_projection.weight.grad.detach().abs().sum().item())
        molecule_projection_grad = float(model.molecule_projection.weight.grad.detach().abs().sum().item())
        ranking_head_grad = float(model.ranking_head.fc1.weight.grad.detach().abs().sum().item())
        classification_head_grad = float(model.classification_head.fc1.weight.grad.detach().abs().sum().item())

        optimizer.step()

        positive_mean_rank = float(outputs.ranking_score[positive_indices].detach().mean().cpu().item())
        negative_mean_rank = float(outputs.ranking_score[negative_indices].detach().mean().cpu().item())
        positive_mean_prob = float(outputs.activity_probability[positive_indices].detach().mean().cpu().item())
        negative_mean_prob = float(outputs.activity_probability[negative_indices].detach().mean().cpu().item())
        peak_memory_value = _peak_memory_mb(device)

        loss_history.append(loss_value)
        pair_loss_history.append(pair_loss_value)
        classification_loss_history.append(classification_loss_value)
        grad_norm_history.append(grad_norm_value)

        print(
            f"step {step:02d}"
            f" | loss={loss_value:.6f}"
            f" pair={pair_loss_value:.6f}"
            f" cls={classification_loss_value:.6f}"
            f" grad_norm={grad_norm_value:.6f}"
            f" enc_p_grad={protein_encoder_grad:.6f}"
            f" enc_m_grad={molecule_encoder_grad:.6f}"
            f" proj_p_grad={protein_projection_grad:.6f}"
            f" proj_m_grad={molecule_projection_grad:.6f}"
            f" head_r_grad={ranking_head_grad:.6f}"
            f" head_c_grad={classification_head_grad:.6f}"
            f" pos_rank={positive_mean_rank:.6f}"
            f" neg_rank={negative_mean_rank:.6f}"
            f" pos_prob={positive_mean_prob:.6f}"
            f" neg_prob={negative_mean_prob:.6f}"
            f" peak_mem_mb={peak_memory_value:.2f}"
        )

    _print_header("Final Parameter Deltas")
    parameter_deltas = {
        "protein_projection_delta": _parameter_delta(protein_projection_before, model.protein_projection.weight),
        "molecule_projection_delta": _parameter_delta(molecule_projection_before, model.molecule_projection.weight),
        "ranking_head_delta": _parameter_delta(ranking_head_before, model.ranking_head.fc1.weight),
        "classification_head_delta": _parameter_delta(
            classification_head_before,
            model.classification_head.fc1.weight,
        ),
    }
    for name, value in parameter_deltas.items():
        print(f"{name}: {value:.6f}")

    _print_header("Training Smoke Finished")
    print(f"loss_history: {[round(value, 6) for value in loss_history]}")
    print(f"pair_loss_history: {[round(value, 6) for value in pair_loss_history]}")
    print(f"classification_loss_history: {[round(value, 6) for value in classification_loss_history]}")
    print(f"grad_norm_history: {[round(value, 6) for value in grad_norm_history]}")
    print("multi-step optimization completed successfully")

    summary = {
        "target_id": target_id,
        "scanned_rows": scanned_rows,
        "num_examples": len(rows),
        "loss_history": loss_history,
        "pair_loss_history": pair_loss_history,
        "classification_loss_history": classification_loss_history,
        "grad_norm_history": grad_norm_history,
        "parameter_deltas": parameter_deltas,
        "optimizer_state_size": len(optimizer.state),
    }

    assert len(loss_history) == num_steps
    assert bool(torch.isfinite(torch.tensor(loss_history)).all().item())
    assert bool(torch.isfinite(torch.tensor(pair_loss_history)).all().item())
    assert bool(torch.isfinite(torch.tensor(classification_loss_history)).all().item())
    assert all(value > 0.0 for value in grad_norm_history)
    assert parameter_deltas["protein_projection_delta"] > 0.0
    assert parameter_deltas["molecule_projection_delta"] > 0.0
    assert parameter_deltas["ranking_head_delta"] > 0.0
    assert parameter_deltas["classification_head_delta"] > 0.0
    assert summary["optimizer_state_size"] > 0
    return summary


def test_build_train_batch_from_curated_csv_finds_mixed_label_target():
    if not CURATED_ROWS_PATH.exists():
        pytest.skip(f"Curated CSV not found at {CURATED_ROWS_PATH}")

    rows, target_id, scanned_rows, _, _, activity_labels, positive_indices, negative_indices = (
        _build_train_batch_from_curated_csv()
    )

    assert target_id
    assert scanned_rows >= len(rows)
    assert len(rows) == 4
    assert len({row["target_chembl_id"] for row in rows}) == 1
    assert activity_labels.tolist() == [0.0, 0.0, 1.0, 1.0]
    assert positive_indices.tolist() == [2, 3]
    assert negative_indices.tolist() == [0, 1]


def test_reward_model_real_encoder_multi_step_train_smoke():
    if os.environ.get(RUN_REAL_ENCODER_TRAIN_SMOKE_ENV) != "1":
        pytest.skip(f"Set {RUN_REAL_ENCODER_TRAIN_SMOKE_ENV}=1 to run the real encoder train smoke test.")

    run_real_encoder_train_smoke()


def main() -> None:
    run_real_encoder_train_smoke()


if __name__ == "__main__":
    main()
