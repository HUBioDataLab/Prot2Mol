import numpy as np
import pytest

from experiments.paired_generalization import ranking_localization as ranking
from experiments.paired_generalization import stratified_trajectory as scalar
from experiments.paired_generalization import verify_localization as verify
from test_paired_generalization_ranking_localization import fixture


def data():
    frame, groups = fixture()
    frame['activity_logits'] = np.round(np.random.default_rng(7).normal(size=len(frame))*4)
    frame.loc[0, 'activity_logits'] = 1000.
    frame.loc[1, 'activity_logits'] = -1000.
    # Keep a rank-eligible assay with constant predictions explicitly undefined.
    frame.loc[frame.group_id == 'p__a2', 'ranking_score'] = 1.
    return frame, groups


def test_independent_scalar_checks_ties_extreme_logits_and_undefined_correlations():
    frame, groups = data()
    summaries, table = scalar.score(frame, frame, groups)
    result = verify.verify_scalar(frame, groups, table, summaries)
    assert result['groups_checked'] == 7 and result['strata_checked'] == 10
    assert table.loc[table.group_id == 'p__a2', 'spearman'].isna().all()
    assert verify.rank_auc([1, 0], [2., 2.]) == .5
    assert verify.rank_auc([1, 1], [1., 2.]) is None


def test_float32_exported_logits_are_accumulated_without_float32_rounding():
    frame, groups = data()
    frame['activity_logits'] = np.random.default_rng(91).normal(size=len(frame)).astype(np.float32)
    summaries, table = scalar.score(frame, frame, groups)
    original = frame.activity_logits.to_numpy().copy()
    assert frame.activity_logits.dtype == np.float32
    verify.verify_scalar(frame, groups, table, summaries)
    np.testing.assert_array_equal(frame.activity_logits.to_numpy(), original)


@pytest.mark.parametrize('field', ['bce_sum', 'spearman', 'size', 'roc_auc'])
def test_scalar_table_or_summary_corruption_is_rejected(field):
    frame, groups = data()
    summaries, table = scalar.score(frame, frame, groups)
    if field == 'roc_auc':
        summaries['all']['roc_auc'] += .01
    elif field == 'spearman':
        index = table.index[table.spearman.notna()][0]
        table.loc[index, field] += .01
    else:
        table.loc[0, field] += 1
    with pytest.raises((ValueError, AssertionError)):
        verify.verify_scalar(frame, groups, table, summaries)


def test_native_descriptor_and_sampled_loss_checks_cover_three_partitions():
    frame, groups = data()
    summaries, table = ranking.evaluate(frame, ranking.make_pool(frame), groups)
    result = verify.verify_ranking(frame, groups, table, summaries)
    assert result['list_descriptors_checked'] == len(table)
    assert result['native_list_checks'] >= 3
    assert result['maximum_sampled_list_difference'] < 2e-6
    changed = table.copy()
    changed.loc[0, 'list_size'] += 1
    with pytest.raises(AssertionError):
        verify.verify_ranking(frame, groups, changed, summaries)


def test_consistently_reduced_but_wrong_list_loss_fails_against_production_operator():
    frame, groups = data()
    summaries, table = ranking.evaluate(frame, ranking.make_pool(frame), groups)
    table.loc[0, 'loss'] += .25
    for name, members in verify.stratum_members(groups).items():
        selected = table[table.group_id.isin(members)]
        summaries[name]['loss'] = float(selected.loss.mean()) if len(selected) else None
    with pytest.raises(AssertionError):
        verify.verify_ranking(frame, groups, table, summaries)


def test_native_row_order_cannot_be_replaced_by_a_different_partitioning():
    frame, groups = data()
    summaries, table = ranking.evaluate(frame, ranking.make_pool(frame), groups)
    with pytest.raises(AssertionError):
        verify.verify_ranking(frame.iloc[::-1].reset_index(drop=True), groups, table, summaries)


def test_hash_check_rejects_changed_bytes(tmp_path):
    path = tmp_path/'artifact'
    path.write_bytes(b'original')
    digest = verify.sha256(path)
    assert verify.checked(path, digest) == path
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Artifact changed'):
        verify.checked(path, digest)
