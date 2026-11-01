import math

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset

from reward_model.model import RewardModelConfig
from reward_model.model.losses import ligunity_listwise_loss, pairwise_logistic_ranking_loss
from test_paired_generalization_adaptation import model, batch
from reward_model.training.evaluation import compute_joint_evaluation_metrics, assay_balanced_partition_weights
from reward_model.training.data import RewardAssayListDataset, RewardAssayListCollator


def test_listwise_normalization_is_weighted_mean_of_retained_selection_terms():
    scores = torch.tensor([.8, -.1, .3], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([9., 7., 4.], dtype=torch.float64)
    groups = torch.zeros(3, dtype=torch.long)
    terms = torch.stack((torch.logsumexp(scores, 0) - scores[0],
                         torch.logsumexp(scores[1:], 0) - scores[1]))
    weights = torch.tensor([1/(math.sqrt(3)*math.log(2)), 1/(math.sqrt(3)*math.log(3))])
    unnormalized = ligunity_listwise_loss(scores, labels, groups)
    normalized = ligunity_listwise_loss(scores, labels, groups, normalize_per_group=True)
    uniform = ligunity_listwise_loss(scores, labels, groups, normalize_per_group=True, position_weighting='uniform')
    torch.testing.assert_close(unnormalized, (terms * weights).sum(), rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(normalized, (terms * weights).sum()/weights.sum(), rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(uniform, terms.mean())


@pytest.mark.parametrize('weighting', ['top', 'uniform'])
def test_pairwise_two_items_are_logistic_and_soft_target_gradient_is_exact(weighting):
    scores = torch.tensor([.7, -.2], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([7., 6.8], dtype=torch.float64)
    groups = torch.zeros(2, dtype=torch.long)
    common = dict(min_list_size=2, min_pchembl_span=0, affinity_margin=0,
                  normalize_per_group=True, position_weighting=weighting)
    hard = pairwise_logistic_ranking_loss(scores, labels, groups, **common)
    torch.testing.assert_close(hard, F.softplus(-(scores[0]-scores[1])))
    soft = pairwise_logistic_ranking_loss(scores, labels, groups, pair_target_temperature=.6, **common)
    expected_target = torch.sigmoid((labels[0]-labels[1])/.6)
    expected_gradient = torch.sigmoid(scores[0]-scores[1]) - expected_target
    soft.backward()
    torch.testing.assert_close(scores.grad, torch.stack((expected_gradient, -expected_gradient)))


def test_uniform_pairs_are_averaged_within_groups_then_groups_are_equal_weighted():
    scores = torch.tensor([2., -1., .3, .1, .7, -.4], dtype=torch.float64)
    labels = torch.tensor([9., 5., 8., 7., 5., 2.], dtype=torch.float64)
    groups = torch.tensor([0, 0, 1, 1, 1, 1])
    common = dict(min_list_size=2, affinity_margin=0, position_weighting='uniform')
    loss = pairwise_logistic_ranking_loss(scores, labels, groups, **common)
    second = torch.stack([F.softplus(-(scores[i]-scores[j])) for i in range(2, 6) for j in range(i+1, 6)]).mean()
    expected = (F.softplus(-(scores[0]-scores[1])) + second) / 2
    torch.testing.assert_close(loss, expected)
    # Pair-count-weighting would give the second group six times the first.
    assert abs(float(loss - (F.softplus(-(scores[0]-scores[1])) + 6*second)/7)) > .01


@pytest.mark.parametrize('objective,extra', [
    (ligunity_listwise_loss, {}),
    (pairwise_logistic_ranking_loss, {}),
    (pairwise_logistic_ranking_loss, {'pair_target_temperature': .6, 'affinity_margin': 0}),
])
def test_tie_permutation_preserves_loss_and_gradient(objective, extra):
    scores = torch.tensor([.7, -.1, .4, -.3], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([8., 8., 6., 4.], dtype=torch.float64)
    groups = torch.zeros(4, dtype=torch.long)
    options = dict(normalize_per_group=True, position_weighting='top', **extra)
    loss = objective(scores, labels, groups, **options)
    gradient, = torch.autograd.grad(loss, scores)
    permutation = torch.tensor([1, 0, 3, 2])
    permuted = scores.detach()[permutation].requires_grad_(True)
    other = objective(permuted, labels[permutation], groups, **options)
    other_gradient, = torch.autograd.grad(other, permuted)
    torch.testing.assert_close(loss, other)
    torch.testing.assert_close(gradient[permutation], other_gradient)


def test_hard_margin_drops_near_ties_while_soft_targets_retain_them():
    scores = torch.tensor([1., -1.], requires_grad=True)
    labels = torch.tensor([7., 7.1])
    groups = torch.zeros(2, dtype=torch.long)
    common = dict(min_list_size=2, min_pchembl_span=0)
    hard = pairwise_logistic_ranking_loss(scores, labels, groups, affinity_margin=.5, **common)
    assert hard == 0
    hard.backward()
    assert torch.count_nonzero(scores.grad) == 0
    soft = pairwise_logistic_ranking_loss(scores, labels, groups, affinity_margin=0,
                                        pair_target_temperature=.5, **common)
    assert soft > 0
    with pytest.raises(ValueError, match='explicit'):
        pairwise_logistic_ranking_loss(scores, labels, groups, pair_target_temperature=.5, **common)


@pytest.mark.parametrize('objective,extra', [
    ('ligunity_listwise', {}),
    ('pairwise_logistic', {}),
    ('pairwise_logistic', {'ranking_pair_target_temperature': .6, 'ranking_affinity_margin': 0}),
])
def test_configured_objective_is_used_by_the_model(objective, extra):
    fitted = model(ranking_objective=objective, ranking_position_weighting='uniform',
                   ranking_normalize_per_group=True, **extra).eval()
    data = batch()
    output = fitted(**data)
    function = ligunity_listwise_loss if objective == 'ligunity_listwise' else pairwise_logistic_ranking_loss
    options = {} if objective == 'ligunity_listwise' else {
        'pair_target_temperature': fitted.config.ranking_pair_target_temperature}
    expected = function(output.ranking_score, data['pchembl_values'], data['ranking_group_ids'],
                        min_pchembl_span=0, affinity_margin=fitted.config.ranking_affinity_margin,
                        position_weighting='uniform', normalize_per_group=True, **options)
    torch.testing.assert_close(output.ranking_loss, expected)
    output.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fitted.ranking_head.parameters())


def test_soft_temperature_configuration_is_explicit_and_validated():
    with pytest.raises(ValueError, match='Soft targets'):
        RewardModelConfig(ranking_pair_target_temperature=.6)
    with pytest.raises(ValueError, match='finite'):
        RewardModelConfig(ranking_objective='pairwise_logistic', ranking_affinity_margin=0,
                          ranking_pair_target_temperature=float('nan'))


@pytest.mark.parametrize('objective,normalization,weighting,soft_temperature', [
    ('ligunity_listwise', False, 'top', None),
    ('ligunity_listwise', True, 'top', None),
    ('pairwise_logistic', True, 'top', None),
    ('pairwise_logistic', True, 'uniform', None),
    ('pairwise_logistic', True, 'uniform', .6),
])
def test_evaluation_recomputes_the_configured_training_objective(objective, normalization, weighting, soft_temperature):
    fitted = model(ranking_objective=objective, ranking_position_weighting=weighting,
                   ranking_normalize_per_group=normalization,
                   ranking_pair_target_temperature=soft_temperature,
                   ranking_affinity_margin=0 if soft_temperature else .47712125472).eval()
    data = batch()
    output = fitted(**data)
    metrics, _ = compute_joint_evaluation_metrics(
        activity_logits=output.activity_logits, ranking_scores=output.ranking_score,
        activity_labels=data['activity_labels'], pchembl_values=data['pchembl_values'],
        ranking_group_ids=data['ranking_group_ids'], group_id_names=['p1__a1', 'p2__a2', 'p3__a3'],
        classification_loss_weight=.25, ranking_loss_weight=.25, bce_pos_weight=1.,
        ranking_temperature=1., ranking_affinity_margin=fitted.config.ranking_affinity_margin,
        ranking_min_pchembl_span=.5, ranking_objective=objective,
        ranking_normalize_per_group=normalization, ranking_position_weighting=weighting,
        ranking_pair_target_temperature=soft_temperature,
        contrastive_loss=output.contrastive_loss, contrastive_loss_weight=1.)
    assert metrics['eval_ranking_loss'] == pytest.approx(float(output.ranking_loss.detach()), abs=1e-6)
    assert metrics['eval_loss'] == pytest.approx(float(output.loss.detach()), abs=1e-6)


@pytest.mark.parametrize('objective', [ligunity_listwise_loss, pairwise_logistic_ranking_loss])
def test_assay_balancing_corrects_sampling_multiplicity_and_survives_batch_partitioning(objective):
    rows = []
    for assay, count in [('large', 40), ('small', 3), ('singleton', 1)]:
        for index in range(count):
            value = [8., 6., 4.][index % 3]
            rows.append(dict(group_id='p__'+assay, pchembl_value=value, binary_label=int(value > 6),
                             protein_input_ids=[3, 4], protein_attention_mask=[1, 1],
                             molecule_input_ids=[3+index % 10, 4], molecule_attention_mask=[1, 1],
                             target_chembl_id='p', compound_id=f'{assay}_{index}',
                             score=(-value if assay == 'large' else value)))
    examples = Dataset.from_list(rows)
    settings = dict(include_all_assays_for_contrastive=True, max_classification_only_per_item=16)
    reference = RewardAssayListDataset(examples, **settings)
    balanced = RewardAssayListDataset(examples, balance_ranking_assays=True, **settings)
    assert balanced.epoch_example_indices() == reference.epoch_example_indices()
    features = [balanced[i] for i in range(len(balanced))]
    weights_by_assay = {}
    losses_by_assay = {}
    for feature in features:
        offset = 0
        for assay, size, flag in zip(feature['contrastive_assay_ids'], feature['contrastive_group_sizes'], feature['ranking_group_flags']):
            part = feature['rows'][offset:offset+size]
            offset += size
            if flag:
                ys = torch.tensor([row['pchembl_value'] for row in part])
                scores = torch.tensor([row['score'] for row in part])
                loss = objective(scores, ys, torch.zeros(size, dtype=torch.long), normalize_per_group=True)
                losses_by_assay.setdefault(assay, []).append(loss)
        for assay, weight in zip(feature['ranking_assay_ids'], feature['ranking_assay_weights']):
            weights_by_assay[assay] = weights_by_assay.get(assay, 0) + weight / len(balanced)
    assert weights_by_assay == pytest.approx({'p__large': .5, 'p__small': .5})
    expected = torch.stack([torch.stack(values).mean() for values in losses_by_assay.values()]).mean()
    collator = RewardAssayListCollator()
    actual = torch.tensor(0.)
    for start in range(0, len(features), 2):
        micro = features[start:start+2]
        batch_data = collator(micro)
        scores = torch.tensor([row['score'] for feature in micro for row in feature['rows']])
        loss = objective(scores, batch_data['pchembl_values'], batch_data['ranking_group_ids'],
                         normalize_per_group=True, group_weights=batch_data['ranking_group_weights'],
                         reduction_denominator=batch_data['ranking_batch_item_count'])
        actual += loss * len(micro) / len(features)
    torch.testing.assert_close(actual, expected)


def test_balanced_evaluation_averages_lists_within_original_assays():
    # Original assay zero supplies two lists; assay one supplies one.
    lists = torch.tensor([0]*3 + [1]*3 + [2]*3 + [-1])
    assays = torch.tensor([0]*6 + [1]*3 + [2])
    scores = torch.tensor([0., 1., 2., 2., 1., 0., 1., 0., 2., 3.])
    labels = torch.tensor([8., 6., 4.]*3 + [8.])
    weights = assay_balanced_partition_weights(lists, assays)
    expected_weights = torch.tensor([.25]*6 + [.5]*3 + [0.])
    torch.testing.assert_close(weights, expected_weights)
    by_list = [ligunity_listwise_loss(scores[lists == i], labels[lists == i], torch.zeros(3, dtype=torch.long),
                                     normalize_per_group=True) for i in range(3)]
    actual = ligunity_listwise_loss(scores, labels, lists, normalize_per_group=True,
                                    group_weights=weights, reduction_denominator=1.)
    torch.testing.assert_close(actual, .25*by_list[0] + .25*by_list[1] + .5*by_list[2])


def test_balanced_model_requires_the_sampling_weights_and_uses_them():
    fitted = model(ranking_balance_assays=True, ranking_normalize_per_group=True).eval()
    data = batch()
    with pytest.raises(ValueError, match='source-assay'):
        fitted(**data)
    data.update(ranking_group_weights=torch.tensor([.5]*6 + [2.]*3), ranking_batch_item_count=3)
    output = fitted(**data)
    expected = ligunity_listwise_loss(output.ranking_score, data['pchembl_values'], data['ranking_group_ids'],
                                      normalize_per_group=True, group_weights=data['ranking_group_weights'],
                                      reduction_denominator=3)
    torch.testing.assert_close(output.ranking_loss, expected)
