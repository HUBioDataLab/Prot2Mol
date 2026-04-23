from __future__ import annotations

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


RUN_REAL_ENCODER_SMOKE_ENV = "RUN_REAL_ENCODER_SMOKE"

DEFAULT_PROTEIN_MODEL = "facebook/esm2_t12_35M_UR50D"
DEFAULT_MOLECULE_MODEL = "HUBioDataLab/SELFormer"


# These examples are copied from the curated reward-model rows the user provided.
# They are only for a forward/backward smoke pass, not for scientific evaluation.
SMOKE_ROWS = [
    {
        "protein_sequence": (
            "MAHVRHFRTLVSGFYFWEAALLLSLVATKETDSARSRSAPMSPSDFLDKLMGRTSGYDARIRPNFKGPPVNVTCNIFINSFGSIAETT"
            "MDYRVNIFLRQKWNDPRLAYSEYPDDSLDLDPSMLDSIWKPDLFFANEKGANFHEVTTDNKLLRIFKNGNVLYSIRLTLTLSCPMDLKN"
            "FPMDVQTCIMQLESFGYTMNDLIFEWQDEAPVQVAEGLTLPQFLLKEEKDLRYCTKHYNTGKFTCIEVRFHLERQMGYYLIQMYIPSLL"
            "IVILSWVSFWINMDAAPARVALGITTVLTMTTQSSGSRASLPKVSYVKAIDIWMAVCLLFVFSALLEYAAVNFVSRQHKELLRFRRKRK"
            "NKTEAFALEKFYRFSDMDDEVRESRFSFTAYGMGPCLQAKDGMTPKGPNHPVQVMPKSPDEMRKVFIDRAKKIDTISRACFPLAFLIFN"
            "IFYWVIYKILRHEDIHQQQD"
        ),
        "compound_selfies": "[C][N][C][=Branch1][C][=O][/C][=Branch1][#C][=C][\\C][=C][NH1][C][=C][C][=C][C][=C][Ring1][=Branch2][Ring1][=Branch1][N][Branch1][C][C][C][Ring1][P][=O]",
        "pchembl_value": 4.17,
        "activity_label": 0.0,
    },
    {
        "protein_sequence": (
            "MAHVRHFRTLVSGFYFWEAALLLSLVATKETDSARSRSAPMSPSDFLDKLMGRTSGYDARIRPNFKGPPVNVTCNIFINSFGSIAETT"
            "MDYRVNIFLRQKWNDPRLAYSEYPDDSLDLDPSMLDSIWKPDLFFANEKGANFHEVTTDNKLLRIFKNGNVLYSIRLTLTLSCPMDLKN"
            "FPMDVQTCIMQLESFGYTMNDLIFEWQDEAPVQVAEGLTLPQFLLKEEKDLRYCTKHYNTGKFTCIEVRFHLERQMGYYLIQMYIPSLL"
            "IVILSWVSFWINMDAAPARVALGITTVLTMTTQSSGSRASLPKVSYVKAIDIWMAVCLLFVFSALLEYAAVNFVSRQHKELLRFRRKRK"
            "NKTEAFALEKFYRFSDMDDEVRESRFSFTAYGMGPCLQAKDGMTPKGPNHPVQVMPKSPDEMRKVFIDRAKKIDTISRACFPLAFLIFN"
            "IFYWVIYKILRHEDIHQQQD"
        ),
        "compound_selfies": "[C][N][C][=Branch1][C][=O][N][C][=Branch1][C][=O][/C][Ring1][#Branch1][=C][\\C][=C][NH1][C][=C][C][=C][C][=C][Ring1][=Branch2][Ring1][=Branch1]",
        "pchembl_value": 4.47,
        "activity_label": 0.0,
    },
    {
        "protein_sequence": (
            "MAHVRHFRTLVSGFYFWEAALLLSLVATKETDSARSRSAPMSPSDFLDKLMGRTSGYDARIRPNFKGPPVNVTCNIFINSFGSIAETT"
            "MDYRVNIFLRQKWNDPRLAYSEYPDDSLDLDPSMLDSIWKPDLFFANEKGANFHEVTTDNKLLRIFKNGNVLYSIRLTLTLSCPMDLKN"
            "FPMDVQTCIMQLESFGYTMNDLIFEWQDEAPVQVAEGLTLPQFLLKEEKDLRYCTKHYNTGKFTCIEVRFHLERQMGYYLIQMYIPSLL"
            "IVILSWVSFWINMDAAPARVALGITTVLTMTTQSSGSRASLPKVSYVKAIDIWMAVCLLFVFSALLEYAAVNFVSRQHKELLRFRRKRK"
            "NKTEAFALEKFYRFSDMDDEVRESRFSFTAYGMGPCLQAKDGMTPKGPNHPVQVMPKSPDEMRKVFIDRAKKIDTISRACFPLAFLIFN"
            "IFYWVIYKILRHEDIHQQQD"
        ),
        "compound_selfies": "[C][C@H1][C@H1][C][=Branch1][C][=O][N][Branch1][C][C][C][=C][C][=N][C][=C][Ring1][=Branch1][C@H1][Ring1][N][C][N][Ring1][#C][S][=Branch1][C][=O][=Branch1][C][=O][C][=C][C][=C][C][=Branch1][Ring2][=C][Ring1][=Branch1][O][C][O][Ring1][=Branch1]",
        "pchembl_value": 7.30,
        "activity_label": 1.0,
    },
    {
        "protein_sequence": (
            "MAHVRHFRTLVSGFYFWEAALLLSLVATKETDSARSRSAPMSPSDFLDKLMGRTSGYDARIRPNFKGPPVNVTCNIFINSFGSIAETT"
            "MDYRVNIFLRQKWNDPRLAYSEYPDDSLDLDPSMLDSIWKPDLFFANEKGANFHEVTTDNKLLRIFKNGNVLYSIRLTLTLSCPMDLKN"
            "FPMDVQTCIMQLESFGYTMNDLIFEWQDEAPVQVAEGLTLPQFLLKEEKDLRYCTKHYNTGKFTCIEVRFHLERQMGYYLIQMYIPSLL"
            "IVILSWVSFWINMDAAPARVALGITTVLTMTTQSSGSRASLPKVSYVKAIDIWMAVCLLFVFSALLEYAAVNFVSRQHKELLRFRRKRK"
            "NKTEAFALEKFYRFSDMDDEVRESRFSFTAYGMGPCLQAKDGMTPKGPNHPVQVMPKSPDEMRKVFIDRAKKIDTISRACFPLAFLIFN"
            "IFYWVIYKILRHEDIHQQQD"
        ),
        "compound_selfies": "[C][N][C][=Branch1][C][=O][C@@H1][C][N][Branch2][Ring1][Branch1][S][=Branch1][C][=O][=Branch1][C][=O][C][=C][C][=C][O][C][=C][C][Ring1][Branch1][=C][Ring1][=Branch2][C][C@@H1][Ring1][P][C][=C][N][=C][C][=C][Ring1][=Branch1][Ring2][Ring1][#Branch2]",
        "pchembl_value": 6.35,
        "activity_label": 1.0,
    },
]


def _print_header(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def _shape_of(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(tensor.shape)


def _mask_lengths(mask: torch.Tensor) -> list[int]:
    return mask.sum(dim=1).detach().cpu().tolist()


def _first_grad_sum(module: torch.nn.Module) -> float:
    for parameter in module.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            return float(parameter.grad.detach().abs().sum().item())
    return 0.0


def _build_smoke_batch() -> tuple[list[str], list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    protein_sequences = [row["protein_sequence"] for row in SMOKE_ROWS]
    molecule_sequences = [row["compound_selfies"] for row in SMOKE_ROWS]
    activity_labels = torch.tensor([row["activity_label"] for row in SMOKE_ROWS], dtype=torch.float32)

    # Smoke-only ranking pairs:
    # lower-activity rows at indices 0,1
    # higher-activity rows at indices 2,3
    positive_indices = torch.tensor([2, 3], dtype=torch.long)
    negative_indices = torch.tensor([0, 1], dtype=torch.long)
    return protein_sequences, molecule_sequences, activity_labels, positive_indices, negative_indices


def run_real_encoder_smoke(
    protein_model_name_or_path: str = DEFAULT_PROTEIN_MODEL,
    molecule_model_name_or_path: str = DEFAULT_MOLECULE_MODEL,
    protein_max_length: int = 1024,
    molecule_max_length: int = 256,
    fusion_hidden_dim: int = 256,
    fusion_num_heads: int = 8,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protein_sequences, molecule_sequences, activity_labels, positive_indices, negative_indices = _build_smoke_batch()

    _print_header("Reward Model Real Encoder Smoke Run")
    print(f"device: {device}")
    print(f"protein_model: {protein_model_name_or_path}")
    print(f"molecule_model: {molecule_model_name_or_path}")
    print(f"num_examples: {len(protein_sequences)}")
    print(f"protein_lengths: {[len(seq) for seq in protein_sequences]}")
    print(f"selfies_lengths: {[len(seq) for seq in molecule_sequences]}")
    print(f"activity_labels: {activity_labels.tolist()}")
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

    _print_header("Forward Pass")
    outputs = model(
        protein_input_ids=protein_batch["input_ids"],
        protein_attention_mask=protein_batch["attention_mask"],
        molecule_input_ids=molecule_batch["input_ids"],
        molecule_attention_mask=molecule_batch["attention_mask"],
        activity_labels=activity_labels.to(device),
        positive_indices=positive_indices.to(device),
        negative_indices=negative_indices.to(device),
        return_token_embeddings=True,
    )

    print(f"ranking_score_shape: {_shape_of(outputs.ranking_score)}")
    print(f"activity_logits_shape: {_shape_of(outputs.activity_logits)}")
    print(f"activity_probability_shape: {_shape_of(outputs.activity_probability)}")
    print(f"joint_embedding_shape: {_shape_of(outputs.joint_embedding)}")
    print(f"projected_protein_tokens_shape: {_shape_of(outputs.protein_token_embeddings)}")
    print(f"projected_molecule_tokens_shape: {_shape_of(outputs.molecule_token_embeddings)}")
    print(f"fused_protein_tokens_shape: {_shape_of(outputs.fused_protein_tokens)}")
    print(f"fused_molecule_tokens_shape: {_shape_of(outputs.fused_molecule_tokens)}")
    print(f"ranking_scores: {outputs.ranking_score.detach().cpu().tolist()}")
    print(f"activity_probabilities: {outputs.activity_probability.detach().cpu().tolist()}")
    print(f"pair_loss: {float(outputs.pair_loss.detach().cpu().item()):.6f}")
    print(f"classification_loss: {float(outputs.classification_loss.detach().cpu().item()):.6f}")
    print(f"total_loss: {float(outputs.loss.detach().cpu().item()):.6f}")

    _print_header("Backward Pass")
    outputs.loss.backward()

    grad_summaries = {
        "protein_encoder_first_grad_sum": _first_grad_sum(model.protein_encoder),
        "molecule_encoder_first_grad_sum": _first_grad_sum(model.molecule_encoder),
        "protein_projection_grad_sum": float(model.protein_projection.weight.grad.detach().abs().sum().item()),
        "molecule_projection_grad_sum": float(model.molecule_projection.weight.grad.detach().abs().sum().item()),
        "fusion_query_p_grad_sum": float(model.fusion.query_p.weight.grad.detach().abs().sum().item()),
        "fusion_query_m_grad_sum": float(model.fusion.query_m.weight.grad.detach().abs().sum().item()),
        "ranking_head_grad_sum": float(model.ranking_head.fc1.weight.grad.detach().abs().sum().item()),
        "classification_head_grad_sum": float(model.classification_head.fc1.weight.grad.detach().abs().sum().item()),
    }
    for name, value in grad_summaries.items():
        print(f"{name}: {value:.6f}")

    _print_header("Smoke Run Finished")
    print("forward/backward completed successfully")

    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss).all()
    assert outputs.ranking_score.shape[0] == len(protein_sequences)
    assert outputs.activity_logits.shape[0] == len(protein_sequences)
    assert outputs.joint_embedding.shape[1] == model.config.fusion_hidden_dim * 2
    for name, value in grad_summaries.items():
        assert value > 0.0, f"expected non-zero gradient summary for {name}"


def test_reward_model_real_encoder_forward_backward_smoke():
    if os.environ.get(RUN_REAL_ENCODER_SMOKE_ENV) != "1":
        pytest.skip(f"Set {RUN_REAL_ENCODER_SMOKE_ENV}=1 to run the real encoder smoke test.")

    run_real_encoder_smoke()


def main() -> None:
    run_real_encoder_smoke()


if __name__ == "__main__":
    main()
