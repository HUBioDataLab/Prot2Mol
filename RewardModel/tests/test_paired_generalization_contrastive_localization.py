from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.paired_generalization import contrastive_localization as audit
from experiments.paired_generalization.contrastive_audit import directional_losses
from experiments.paired_generalization.similarity_composition import descriptors
from reward_model.model.losses import ligunity_bidirectional_contrastive_loss


@pytest.mark.parametrize('labels', [[6., 6.001, 7., 5., 8., 6.5], [5.]*6, [7.]*6])
def test_query_contributions_reproduce_native_masks_weights_and_strict_cutoff(labels):
    batch = SimpleNamespace(owners=torch.tensor([0, 0, 1, 2, 2, 2]),
        target_ids=torch.tensor([10, 10, 20]), molecule_ids=torch.tensor([1, 2, 3, 2, 4, 5]),
        labels=torch.tensor(labels))
    scores = torch.tensor([[2., 1., -1., .5, -.3, 1.], [1., 0., 3., 2., 1., -1.], [1., -1., 0., 2., 1., 3.]])
    result = audit.batch_contributions(scores, batch)
    expected = ligunity_bidirectional_contrastive_loss(scores, batch.labels, batch.owners,
        batch.target_ids, batch.molecule_ids, active_threshold=6., strict_active_only=True)
    for name, value in zip(('total', 'protein_to_molecule', 'molecule_to_protein'), expected):
        torch.testing.assert_close(result[name].mean(), value, atol=1e-6, rtol=0)
    torch.testing.assert_close(result['total'], result['protein_to_molecule']+result['molecule_to_protein'])
    if max(labels) <= 6.:
        assert torch.equal(result['total'], torch.zeros(3))
    # Removing the complete mask bundle changes the same observed batch.
    if min(labels) > 6.:
        unmasked = ligunity_bidirectional_contrastive_loss(scores, batch.labels, batch.owners,
            batch.target_ids, batch.molecule_ids, active_threshold=6., strict_active_only=True, mask_policy='none')
        assert float(unmasked[0]) != pytest.approx(float(expected[0]))


def fixture():
    lengths = [1, 2, 3, 4, 18]
    groups = np.repeat(np.arange(5), lengths)
    targets = groups % 3
    frame = pd.DataFrame(dict(group_id=[f'p{p}__a{g}' for p, g in zip(targets, groups)],
        assay_id=[f'a{g}' for g in groups], target_chembl_id=[f'p{p}' for p in targets],
        compound_id=[f'm{i}' for i in range(len(groups))], pchembl_value=7., binary_label=1))
    frame.loc[frame.assay_id == 'a3', ['pchembl_value', 'binary_label']] = [5., 0]
    scaffold = pd.Series(['S']*6+['U']*(len(frame)-6), index=frame.compound_id)
    rows, desc, _ = descriptors(frame, ['m0'], scaffold, pd.Series({f'a{i}': i//2 for i in range(5)}))
    rng = np.random.default_rng(42)
    pv = rng.normal(size=(3, 8)); mv = rng.normal(size=(len(frame), 8))
    pv /= np.linalg.norm(pv, axis=1, keepdims=True); mv /= np.linalg.norm(mv, axis=1, keepdims=True)
    return frame, desc, torch.tensor(pv[targets], dtype=torch.float32), torch.tensor(mv, dtype=torch.float32)


def test_full_pool_attribution_reconstructs_both_directions_and_keeps_zero_active_lists():
    frame, desc, protein, molecule = fixture()
    pools, _ = audit.candidate_pool(frame)
    scales = dict(learned_scale=17., fixed_initial_scale13=13.)
    summary, tables, checked = audit.localize(frame, protein, molecule, pools, scales, desc, check_every=1)
    expected = directional_losses(protein, molecule, pools, scales)
    assert checked['native_batches_checked'] == 6
    for scale, table in tables.items():
        for direction in audit.DIRECTIONS:
            assert summary[scale]['all'][direction] == pytest.approx(expected[scale][direction], abs=1e-6)
            numerator = sum(summary[scale][f'scaffold/{key}']['list_instances'] *
                (summary[scale][f'scaffold/{key}'][direction] or 0.) for key in ('all_shared', 'mixed', 'all_unseen'))
            assert numerator/len(table) == pytest.approx(expected[scale][direction], abs=1e-6)
        assert len(table[table.group_id == 'p0__a3']) == 3
        assert table.loc[table.group_id == 'p0__a3', 'total'].eq(0.).all()
        assert summary[scale]['all']['list_instances'] == 18
        assert summary[scale]['scaffold/mixed']['total'] is None
    # Rebuilding candidates using only the singleton stratum is a different task.
    subset = frame.iloc[[0]].reset_index(drop=True)
    subset_pool, _ = audit.candidate_pool(subset)
    isolated = directional_losses(protein[:1], molecule[:1], subset_pool, scales)
    assert isolated['learned_scale']['total'] == 0.
    assert summary['learned_scale']['size/1']['total'] > 0.


def test_invalid_embeddings_missing_descriptors_and_wrong_policy_are_refused():
    frame, desc, protein, molecule = fixture(); pools, _ = audit.candidate_pool(frame)
    with pytest.raises(ValueError, match='aligned and finite'):
        audit.localize(frame, protein[:-1], molecule, pools, {'learned': 13.}, desc)
    with pytest.raises(ValueError, match='positive finite'):
        audit.localize(frame, protein, molecule, pools, {'learned': -1.}, desc)
    with pytest.raises(ValueError, match='lack fixed'):
        audit.localize(frame, protein, molecule, pools, {'learned': 13.}, desc.iloc[:-1])
    policy = dict(pair_scoring_mode='mlp_contrastive', contrastive_active_threshold=6., contrastive_strict_active_only=True)
    audit.check_policy(policy)
    for change in [dict(contrastive_active_threshold=5.), dict(contrastive_strict_active_only=False), dict(contrastive_mask_policy='training_known')]:
        with pytest.raises(ValueError, match='fixed reference'):
            audit.check_policy({**policy, **change})


def test_incomplete_trajectory_is_not_reported_as_complete():
    with pytest.raises(ValueError, match='fifteen'):
        audit.trajectory_summary([dict(epoch=1)])


def test_endpoint_anchor_uses_this_runs_native_validation_peak():
    records = [dict(epoch=i, native_macro_spearman=-(i-9)**2,
        strata={'learned_scale': {'all': dict(total=float(i), protein_to_molecule=.6*i, molecule_to_protein=.4*i)}})
        for i in range(1, 16)]
    result = audit.trajectory_summary(records)
    assert result['native_macro_peak_epoch'] == 9
    assert set(result['endpoint_point_changes']['learned_scale']['total']) == {'1_to_15', '9_to_15'}
