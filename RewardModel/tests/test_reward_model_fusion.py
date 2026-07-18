import copy

import torch
import torch.nn.functional as F

from reward_model.model.fusion import TokenFusion


def test_sdpa_fusion_matches_manual_forward_and_backward():
    torch.manual_seed(42)
    manual = TokenFusion(hidden_dim=8, num_heads=2, attention_backend="manual")
    sdpa = copy.deepcopy(manual)
    sdpa.attention_backend = "sdpa"
    manual_protein = torch.randn(2, 5, 8, requires_grad=True)
    manual_molecule = torch.randn(2, 4, 8, requires_grad=True)
    sdpa_protein = manual_protein.detach().clone().requires_grad_(True)
    sdpa_molecule = manual_molecule.detach().clone().requires_grad_(True)
    protein_mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]],
        dtype=torch.long,
    )
    molecule_mask = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 1, 0]],
        dtype=torch.long,
    )

    manual_outputs = manual(
        manual_protein,
        manual_molecule,
        protein_mask,
        molecule_mask,
    )
    sdpa_outputs = sdpa(
        sdpa_protein,
        sdpa_molecule,
        protein_mask,
        molecule_mask,
    )

    for manual_output, sdpa_output in zip(manual_outputs, sdpa_outputs):
        assert torch.allclose(manual_output, sdpa_output, atol=1e-6, rtol=1e-5)

    sum(output.square().sum() for output in manual_outputs).backward()
    sum(output.square().sum() for output in sdpa_outputs).backward()
    assert torch.allclose(manual_protein.grad, sdpa_protein.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(manual_molecule.grad, sdpa_molecule.grad, atol=1e-6, rtol=1e-5)
    sdpa_parameters = dict(sdpa.named_parameters())
    for name, manual_parameter in manual.named_parameters():
        assert manual_parameter.grad is not None, name
        assert sdpa_parameters[name].grad is not None, name
        assert torch.allclose(
            manual_parameter.grad,
            sdpa_parameters[name].grad,
            atol=1e-6,
            rtol=1e-5,
        ), name


def test_sdpa_fusion_zeroes_padded_query_rows():
    fusion = TokenFusion(hidden_dim=8, num_heads=2, attention_backend="sdpa")
    protein_tokens = torch.randn(1, 3, 8)
    molecule_tokens = torch.randn(1, 2, 8)
    protein_mask = torch.tensor([[1, 0, 0]], dtype=torch.long)
    molecule_mask = torch.tensor([[1, 1]], dtype=torch.long)

    fused_protein, fused_molecule = fusion(
        protein_tokens,
        molecule_tokens,
        protein_mask,
        molecule_mask,
    )

    assert torch.count_nonzero(fused_protein[:, 1:]) == 0
    assert torch.isfinite(fused_protein).all()
    assert torch.isfinite(fused_molecule).all()


def test_sdpa_fusion_never_sends_a_fully_masked_query(monkeypatch):
    original_sdpa = F.scaled_dot_product_attention

    def _checked_sdpa(query, key, value, *, attn_mask, **kwargs):
        assert attn_mask.any(dim=-1).all()
        return original_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            **kwargs,
        )

    monkeypatch.setattr(F, "scaled_dot_product_attention", _checked_sdpa)
    fusion = TokenFusion(hidden_dim=8, num_heads=2, attention_backend="sdpa")
    fusion(
        torch.randn(1, 3, 8),
        torch.randn(1, 2, 8),
        torch.tensor([[1, 0, 0]], dtype=torch.long),
        torch.tensor([[1, 1]], dtype=torch.long),
    )


def test_fusion_rejects_unknown_attention_backend():
    try:
        TokenFusion(hidden_dim=8, num_heads=2, attention_backend="unknown")
    except ValueError as exc:
        assert "attention_backend" in str(exc)
    else:
        raise AssertionError("Expected an invalid attention backend to be rejected")
