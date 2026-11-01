import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import log_loss

from experiments.paired_generalization import stratified_trajectory as audit
from experiments.paired_generalization.similarity_composition import descriptors
from test_paired_generalization_native_analysis import snapshot


def fixture():
    frame = pd.DataFrame(dict(group_id=['p__a']*3+['p__b']*4+['p__c'],
        target_chembl_id=['p']*8, assay_id=['a']*3+['b']*4+['c'],
        compound_id=[f'm{i}' for i in range(8)],
        pchembl_value=[5., 6., 6., 5., 5.5, 6.5, 7., 6.],
        binary_label=[0, 1, 1, 0, 0, 1, 1, 1],
        ranking_score=[0., 2., 1., 2., 1., 0., 0., 1.],
        activity_logits=[-.5, .3, 1., -.3, -.1, .2, 2., .1]))
    scaffolds = pd.Series(['shared']*3+['new']*5, index=frame.compound_id)
    rows, groups, _ = descriptors(frame, ['m0'], scaffolds, pd.Series({'a': 0, 'b': 0, 'c': 1}))
    return frame, rows, groups


def test_ties_row_vs_assay_weighting_empty_strata_and_permuted_rows():
    frame, rows, groups = fixture()
    result, table = audit.score(frame.sample(frac=1, random_state=9), rows, groups)
    expected = [spearmanr(g.pchembl_value, g.ranking_score).statistic
                for _, g in frame.groupby('group_id') if len(g) >= 3]
    assert result['all']['macro_spearman'] == pytest.approx(np.mean(expected))
    assert result['all']['row_bce'] == pytest.approx(log_loss(frame.binary_label, expit(frame.activity_logits)))
    assay_losses = [np.mean(np.logaddexp(0, g.activity_logits)-g.binary_label*g.activity_logits)
                    for _, g in frame.groupby('group_id')]
    assert result['all']['assay_macro_bce'] == pytest.approx(np.mean(assay_losses))
    assert result['all']['assay_macro_bce'] != pytest.approx(result['all']['row_bce'])
    assert result['size/1']['macro_spearman'] is None
    assert result['size/2']['rows'] == 0
    assert result['scaffold/mixed']['roc_auc'] is None
    assert table.spearman.notna().sum() == 2


def test_changed_identity_label_eligibility_and_nonfinite_scores_fail():
    frame, rows, groups = fixture()
    with pytest.raises(ValueError, match='complete permutation'):
        audit.score(frame.iloc[:-1], rows, groups)
    changed = frame.copy(); changed.loc[0, 'binary_label'] = 1
    with pytest.raises(ValueError, match='labels disagree'):
        audit.score(changed, rows, groups)
    changed = groups.copy(); changed.loc[0, 'eligible'] = False
    with pytest.raises(ValueError, match='eligibility'):
        audit.score(frame, rows, changed)
    changed = frame.copy(); changed.loc[0, 'activity_logits'] = np.inf
    with pytest.raises(ValueError, match='finite scores'):
        audit.score(changed, rows, groups)


def test_bootstrap_resamples_whole_units_and_excludes_unmatched_undefined_ranks():
    frame, rows, groups = fixture()
    _, first = audit.score(frame, rows, groups)
    changed = frame.copy(); changed.activity_logits *= -2
    changed.loc[changed.assay_id == 'a', 'ranking_score'] = 0.
    _, last = audit.score(changed, rows, groups)
    result = audit.paired_changes(first, last, repetitions=500)
    assert result['all']['matched_groups'] == 1
    assert result['all']['all_partition_units'] == 2
    assert result['scaffold/all_shared']['matched_macro_spearman']['change'] is None
    assert result['scaffold/all_shared']['stratum_units'] == 1
    # Independent bootstrap of the two blocks; assays a/b must travel together.
    delta = (last.bce_sum-first.bce_sum).to_numpy()
    totals = np.array([delta[:2].sum(), delta[2]])
    counts = np.array([7, 1])
    draw = np.random.default_rng(42).integers(2, size=(500, 2))
    samples = totals[draw].sum(axis=1)/counts[draw].sum(axis=1)
    assert result['all']['row_bce']['interval'] == pytest.approx(np.quantile(samples, [.025, .975]))
    assert result['size/2']['row_bce']['interval'] is None


def test_native_trainer_prediction_bce_and_headline_match(tmp_path):
    manifest, expected, examples, _ = snapshot(tmp_path)
    frame, record = audit.native_frame(manifest, expected, examples._fingerprint, 'eval')
    scaffolds = pd.Series(['s']*len(frame), index=frame.compound_id)
    units = pd.Series({name: i for i, name in enumerate(expected.assay_id.unique())})
    rows, groups, _ = descriptors(audit.trainer_label_frame(expected), ['m0'], scaffolds, units)
    result, _ = audit.score(frame, rows, groups)
    assert result['all']['macro_spearman'] == pytest.approx(record['metrics']['eval_macro_spearman'], abs=1e-14)
    assert result['all']['row_bce'] == pytest.approx(record['metrics']['eval_classification_loss'], abs=1e-5)


def test_prefix_cannot_become_complete_and_changed_group_membership_fails():
    frame, rows, groups = fixture()
    stats, table = audit.score(frame, rows, groups)
    result = audit.summarize([dict(epoch=1, strata=stats)], {1: table})
    assert result['complete_horizon'] is False
    assert result['epochs'] == [1]
    assert result['paired_endpoint_changes']['1_to_1']['all']['row_bce']['change'] == 0.
    with pytest.raises(ValueError, match='contiguous'):
        audit.summarize([dict(epoch=2, strata=stats)], {2: table})
    changed = table.copy(); changed.loc[0, 'unit_id'] = 99
    with pytest.raises(AssertionError):
        audit.paired_changes(table, changed)
