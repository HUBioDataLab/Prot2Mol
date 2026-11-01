import numpy as np
import pandas as pd
import pytest
import torch

from experiments.paired_generalization import ranking_localization as audit
from experiments.paired_generalization.similarity_composition import descriptors
from reward_model.model.losses import ligunity_listwise_loss
from reward_model.training.evaluation import build_complete_coverage_ranking_partitions
from test_paired_generalization_native_analysis import snapshot


def fixture():
    rng = np.random.default_rng(91)
    lengths = [1, 2, 3, 16, 17, 65, 5]
    names = np.repeat([f'p__a{i}' for i in range(len(lengths))], lengths)
    labels = np.round(rng.uniform(5, 8, sum(lengths))*4)/4
    labels[-5:] = 6.
    frame = pd.DataFrame(dict(group_id=names, assay_id=[s.split('__')[1] for s in names],
        target_chembl_id='p', compound_id=[f'm{i}' for i in range(len(names))],
        pchembl_value=labels, binary_label=(labels>=6).astype(int), ranking_score=rng.normal(size=len(names))))
    scaffolds = pd.Series(['s']*len(frame), index=frame.compound_id)
    units = pd.Series({f'a{i}': i//2 for i in range(len(lengths))})
    _, groups, _ = descriptors(frame, ['m0'], scaffolds, units)
    return frame, groups


def test_vectorized_lists_match_native_ties_near_ties_and_zero_loss():
    rng = np.random.default_rng(42)
    counts = np.array([3, 4, 7, 15, 16, 5])
    values = np.round(rng.uniform(5, 7, counts.sum())*4)/4
    values[-5:] = 6.
    scores = rng.normal(0, 3, len(values)).astype(np.float32)
    index = np.zeros((len(counts), 16), dtype=int)
    offset = 0; expected = []
    for row, n in enumerate(counts):
        index[row, :n] = np.arange(offset, offset+n)
        expected.append(float(ligunity_listwise_loss(torch.tensor(scores[offset:offset+n]),
            torch.tensor(values[offset:offset+n], dtype=torch.float32), torch.zeros(n, dtype=torch.long),
            affinity_margin=audit.POLICY['ranking_affinity_margin'], min_pchembl_span=0.)))
        offset += n
    actual = audit.list_losses(scores, values, index, counts)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
    assert actual[-1] == 0.
    with pytest.raises(ValueError, match='Nonfinite'):
        audit.list_losses(scores*np.nan, values, index, counts)


def test_native_complete_coverage_reduction_and_original_assay_strata():
    frame, groups = fixture(); pool = audit.make_pool(frame)
    result, table = audit.evaluate(frame, pool, groups)
    ids = torch.tensor(pd.factorize(frame.group_id, sort=True)[0])
    labels = torch.tensor(frame.pchembl_value, dtype=torch.float32)
    scores = torch.tensor(frame.ranking_score, dtype=torch.float32)
    parts = build_complete_coverage_ranking_partitions(pchembl_values=labels, ranking_group_ids=ids)
    expected = [ligunity_listwise_loss(scores, labels, part, min_pchembl_span=0.,
        affinity_margin=audit.POLICY['ranking_affinity_margin']) for part in parts]
    assert result['all']['loss'] == pytest.approx(float(torch.stack(expected).mean()), abs=2e-6)
    assert result['size/>25']['groups'] == 1
    assert result['size/>25']['lists_per_partition'] == 5
    assert result['size/1']['loss'] is None
    assert result['size/2']['loss'] is None
    assert 'p__a6' not in set(table.group_id)  # Flat complete assay is ineligible.
    # Full eligible assays are covered once in each partition, without duplicate rows.
    for indices, counts in pool[0]:
        members = np.concatenate([indices[i, :n] for i, n in enumerate(counts)])
        assert len(members) == len(np.unique(members)) == 3+16+17+65
    same = audit.make_pool(frame)
    assert same[2] == pool[2]
    assert audit.make_pool(frame.iloc[::-1].reset_index(drop=True))[2] != pool[2]


def test_actual_trainer_ranking_loss_parity(tmp_path):
    path, expected, examples, _ = snapshot(tmp_path)
    frame, native = audit.native_frame(path, expected, examples._fingerprint, 'eval')
    scaffolds = pd.Series(['s']*len(frame), index=frame.compound_id)
    units = pd.Series({name: i for i, name in enumerate(expected.assay_id.unique())})
    _, groups, _ = descriptors(audit.trainer_label_frame(expected), ['m0'], scaffolds, units)
    result, _ = audit.evaluate(frame, audit.make_pool(frame), groups)
    assert result['all']['loss'] == pytest.approx(native['metrics']['eval_ranking_loss'], abs=2e-6)


def test_rejects_different_objective_normalization_and_native_group_indices():
    policy = {k: v for k, v in audit.POLICY.items() if k.startswith('ranking_')}
    audit.check_model_policy(policy)
    for key, value in [('ranking_temperature', 2.), ('ranking_balance_assays', True),
                       ('ranking_normalize_per_group', True), ('ranking_position_weighting', 'uniform')]:
        with pytest.raises(ValueError, match='Unsupported'):
            audit.check_model_policy({**policy, key: value})
    frame, _ = fixture();frame['evaluation_group_index'] = 99
    with pytest.raises(ValueError, match='assay index'):
        audit.make_pool(frame)


def test_bootstrap_keeps_lists_from_whole_units_and_includes_ineligible_units():
    frame, groups = fixture(); _, first = audit.evaluate(frame, audit.make_pool(frame), groups)
    last = first.copy();last['loss'] += last.unit_id.to_numpy()+.25
    # Add a wholly ineligible unit: it remains part of the resampling universe.
    units = np.r_[groups.unit_id.unique(), 99]
    result = audit.changes(first, last, units, repetitions=500)
    totals = first.assign(delta=last.loss-first.loss).groupby('unit_id').delta.agg(['sum', 'count']).reindex(sorted(units), fill_value=0)
    draws = np.random.default_rng(42).integers(len(units), size=(500, len(units)))
    count = totals['count'].to_numpy()[draws].sum(axis=1)
    total = totals['sum'].to_numpy()[draws].sum(axis=1)
    expected = total[count>0]/count[count>0]
    assert result['all']['interval'] == pytest.approx(np.quantile(expected, [.025, .975]))
    assert result['all']['all_partition_units'] == len(units)
    assert result['size/1']['change'] is None
    changed = last.copy();changed.loc[0, 'group_id'] = 'other'
    with pytest.raises(AssertionError):
        audit.changes(first, changed, units)
