import itertools

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.paired_generalization import representation_geometry as geometry
from experiments.paired_generalization.qualify_controls import make_model
from test_paired_generalization_checkpoint_evaluation import examples_fixture
from reward_model.training.data import RewardPairCollator


def test_rank_uses_singular_entropy_not_covariance_entropy_or_centering():
    values = np.diag([3., 1.])
    expected = np.exp(-(.75*np.log(.75)+.25*np.log(.25)))
    result = geometry.effective_rank(values)
    assert result['value'] == pytest.approx(expected)
    covariance_entropy = np.exp(-(.9*np.log(.9)+.1*np.log(.1)))
    assert abs(result['value']-covariance_entropy) > .2
    assert geometry.effective_rank(np.ones((5, 3)))['value'] == pytest.approx(1.)
    assert geometry.effective_rank(np.zeros((5, 3)))['value'] == 0.
    assert geometry.effective_rank(np.eye(4))['value'] == pytest.approx(4.)
    assert geometry.effective_rank(values*1e150)['value'] == pytest.approx(expected)


def test_uniformity_matches_explicit_pairs_and_is_scale_rotation_invariant():
    values = np.array([[1., 0., 0.], [0., 2., 0.], [-3., 0., 0.], [1., 1., 1.]])
    norm = values/np.linalg.norm(values, axis=1, keepdims=True)
    expected = np.log(np.mean([np.exp(-2*np.sum((norm[i]-norm[j])**2))
                              for i, j in itertools.combinations(range(4), 2)]))
    result = geometry.uniformity(values, block_size=2)
    assert result['value'] == pytest.approx(expected)
    assert result['valid_pairs'] == 6 and result['pair_coverage'] == 1.
    rotation, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(3, 3)))
    assert geometry.uniformity(values@rotation*8)['value'] == pytest.approx(expected)
    assert geometry.uniformity(np.ones((4, 3)))['value'] == pytest.approx(0.)


def test_pair_sampling_is_fixed_uniform_without_self_pairs_and_keeps_zero_coverage():
    pairs, metadata = geometry.pair_indices(7, budget=10, seed=4)
    repeated, same = geometry.pair_indices(7, budget=10, seed=4)
    np.testing.assert_array_equal(pairs, repeated)
    assert metadata == same and (pairs[:, 0] < pairs[:, 1]).all()
    # Sampling one ordered distinct pair and canonicalizing is uniform over
    # unordered pairs; a reproducible larger draw checks endpoint balance.
    pairs, _ = geometry.pair_indices(1000, budget=200000, seed=4)
    occurrences = np.bincount(pairs.ravel(), minlength=1000)
    assert occurrences.min() > 300 and occurrences.max() < 500
    values = np.eye(4); values[0] = 0
    result = geometry.uniformity(values)
    assert result['valid_rows'] == 3 and result['valid_pairs'] == 3
    assert result['pair_coverage'] == .5 and result['value'] == pytest.approx(-4.)
    assert geometry.uniformity(np.zeros((4, 3)))['value'] is None
    assert geometry.uniformity(np.ones((1, 3)))['value'] is None


def test_alignment_has_explicit_shared_space_positive_selection_and_zero_exclusion():
    protein = np.array([[1., 0.], [0., 1.], [0., 0.]])
    molecule = np.array([[2., 0.], [0., -1.], [1., 0.]])
    result = geometry.alignment(protein, molecule)
    assert result['value'] == pytest.approx(2.) and result['valid_pairs'] == 2
    assert geometry.alignment(protein, molecule, selected=np.array([False, True, True]))['value'] == 4.
    with pytest.raises(ValueError, match='alignment space'):
        geometry.alignment(protein, molecule, stage='encoder_pooled')
    with pytest.raises(ValueError, match='identical shapes'):
        geometry.alignment(protein, np.ones((3, 4)))
    with pytest.raises(ValueError, match='boolean mask'):
        geometry.alignment(protein, molecule, selected=[1, 0, 1])


def test_memberships_ignore_labels_and_row_order_and_separate_entity_weighting():
    frame = pd.DataFrame([dict(group_id=f'g{i}', target_chembl_id=f'p{i%3}',
                              compound_id=f'm{i%5}', pchembl_value=float(i)) for i in range(20)])
    first, counts = geometry.sampling_design(frame, 'B', 'val', cap=7)
    altered = frame.iloc[::-1].assign(pchembl_value=-100.)
    second, counts2 = geometry.sampling_design(altered, 'B', 'val', cap=7)
    pd.testing.assert_frame_equal(first.drop(columns='source_row_index'), second.drop(columns='source_row_index'))
    assert counts == counts2
    assert [counts[k]['sampled'] for k in ['observations','unique_proteins','unique_molecules']] == [7,3,5]
    for _, row in first.iterrows():
        assert frame.iloc[row.source_row_index][geometry.IDENTITY].tolist() == row[geometry.IDENTITY].tolist()
    with pytest.raises(ValueError, match='unique observation'):
        geometry.sampling_design(pd.concat([frame, frame.iloc[:1]]), 'A', 'train')


def test_real_model_capture_uses_pooled_contrastive_branch_and_preserves_weights():
    model = make_model(None, {})
    model.eval()
    examples = examples_fixture(model)
    rows = [examples[i] for i in range(3)]
    collator = RewardPairCollator(dynamic_padding=True, protein_pad_token_id=1, molecule_pad_token_id=1)
    inputs = collator.collate_example_tokens(rows)
    weights = {k:v.clone() for k,v in model.state_dict().items()}
    with torch.inference_mode(), geometry.capture_projection_stages(model) as captured:
        output = model(**inputs, return_dict=True)
    assert len(captured) == 4
    for side in ('protein', 'molecule'):
        raw = captured[side+'_encoder_pooled']
        projected = captured[side+'_contrastive_projected']
        layer = getattr(model, 'contrastive_'+side+'_projection')
        with torch.inference_mode():
            expected = layer(torch.from_numpy(raw)).numpy()
        np.testing.assert_allclose(projected, expected, atol=1e-6)
        normalized, _ = geometry.normalized(projected)
        np.testing.assert_allclose(normalized, getattr(output,'normalized_'+side+'_embedding').numpy(), atol=1e-6)
        assert not layer._forward_hooks
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    model.train()
    with pytest.raises(ValueError, match='model.eval'):
        with geometry.capture_projection_stages(model):
            pass
    model.eval()
    with pytest.raises(RuntimeError, match='inference failed'):
        with geometry.capture_projection_stages(model):
            raise RuntimeError('inference failed')
    assert not model.contrastive_protein_projection._forward_hooks
    assert not model.contrastive_molecule_projection._forward_hooks


@pytest.mark.parametrize('values', [np.ones(3), np.empty((0, 3)), np.array([[np.nan, 1.]])])
def test_invalid_representations_cannot_be_silently_scored(values):
    with pytest.raises(ValueError, match='finite nonempty'):
        geometry.effective_rank(values)
    with pytest.raises(ValueError, match='finite nonempty'):
        geometry.uniformity(values)
