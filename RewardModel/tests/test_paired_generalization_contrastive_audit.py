import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from experiments.paired_generalization.contrastive_audit import candidate_pool, directional_losses
from reward_model.training.evaluation import compute_contrastive_evaluation_loss


def test_fixed_pool_directions_reproduce_original_full_coverage_evaluation():
    frame = pd.DataFrame({"group_id": ["a"] * 2 + ["b"] * 19 + ["c"] * 5,
                          "target_chembl_id": ["x"] * 21 + ["y"] * 5,
                          "compound_id": [str(i % 17) for i in range(26)],
                          "pchembl_value": np.linspace(4.5, 8., 26)})
    generator = torch.Generator().manual_seed(71)
    p = F.normalize(torch.randn(26, 8, generator=generator), dim=1)
    m = F.normalize(torch.randn(26, 8, generator=generator), dim=1)
    pool, fingerprint = candidate_pool(frame, batch_size=2)
    result = directional_losses(p, m, pool, {"learned": 17., "fixed": 13.})
    original = compute_contrastive_evaluation_loss(
        normalized_protein_embeddings=p, normalized_molecule_embeddings=m,
        pchembl_values=torch.tensor(frame.pchembl_value.to_numpy(), dtype=torch.float32),
        contrastive_group_ids=torch.tensor(pd.factorize(frame.group_id, sort=True)[0]),
        target_identity_ids=torch.tensor(pd.factorize(frame.target_chembl_id, sort=True)[0]),
        molecule_identity_ids=torch.tensor(pd.factorize(frame.compound_id, sort=True)[0]),
        temperature=1 / 17., active_threshold=6., assay_batch_size=2, strict_active_only=True)
    assert result["learned"]["total"] == pytest.approx(float(original), rel=1e-6)
    for value in result.values():
        assert value["total"] == pytest.approx(value["protein_to_molecule"] + value["molecule_to_protein"], rel=1e-6)
    assert candidate_pool(frame, batch_size=2)[1] == fingerprint
    assert candidate_pool(frame, batch_size=2, seed=43)[1] != fingerprint
