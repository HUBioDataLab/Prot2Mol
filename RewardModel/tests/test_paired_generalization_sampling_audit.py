import numpy as np
import pytest
import torch

from experiments.paired_generalization.sampling_audit import false_negative_counts
from reward_model.model.losses import ligunity_bidirectional_contrastive_loss


def test_known_positive_audit_matches_actual_contrastive_gradient_mask():
    targets = np.array([10, 20, 10])
    molecules = np.arange(6)
    owners = np.repeat(np.arange(3), 2)
    known = {10: {0, 1, 2, 4, 5}, 20: {2, 3}}
    counts = false_negative_counts(targets, molecules, owners, np.ones(6, bool), known)
    assert counts['protein_to_molecule_known_false_negatives'] == 4
    assert counts['molecule_to_protein_known_false_negatives'] == 2
    assert counts['unmasked_same_target_negatives'] == 0
    assert counts['unmasked_duplicate_molecule_negatives'] == 0
    scores = torch.zeros((3, 6), dtype=torch.float64, requires_grad=True)
    _, p2m, m2p = ligunity_bidirectional_contrastive_loss(scores, torch.full((6,), 8.),
        torch.tensor(owners), torch.tensor(targets), torch.tensor(molecules),
        active_threshold=6., strict_active_only=True)
    for loss in (p2m, m2p):
        gradient, = torch.autograd.grad(loss, scores, retain_graph=True)
        assert gradient[0, 4] == 0 and gradient[0, 5] == 0
        assert gradient[2, 0] == 0 and gradient[2, 1] == 0
        # Known training positives outside the current query assay still
        # receive the positive gradient that pushes negative-candidate scores down.
        assert gradient[0, 2] > 0 and gradient[2, 2] > 0


def test_duplicate_candidate_and_inactive_anchor_do_not_inflate_false_negative_count():
    counts = false_negative_counts([10, 20], [0, 1, 0, 3], [0, 0, 1, 1],
                                   [True, False, False, True], {10: {0, 1}, 20: {0, 3}})
    assert counts['duplicate_molecule_cross_list_cells_masked'] == 2
    assert counts['protein_to_molecule_known_false_negatives'] == 0
    assert counts['molecule_to_protein_known_false_negatives'] == 0
    assert counts['active_anchors'] == 2
