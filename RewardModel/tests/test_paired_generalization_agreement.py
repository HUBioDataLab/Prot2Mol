import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from experiments.paired_generalization.scorer_agreement import (
    SCORERS, PAIRS, group_agreement, matched_endpoint_changes, summarize,
)


def fixture():
    return pd.DataFrame(dict(group_id=['T__A']*4+['T__B']*3+['T__C'],
        ranking_score=[2, 2, 7, 1, 1, 2, 3, 1],
        activity_logits=[-2, -2, -7, -1, 3, 2, 1, 1],
        cosine_similarity=[0, .2, .1, .3, 0, 0, 0, 0],
        pchembl_value=[6, 7, 5, 8, 4, 5, 8, 4]))


def test_agreement_matches_scipy_for_ties_and_reports_constant_coverage():
    frame = fixture()
    table = group_agreement(frame, dict(A=0, B=1, C=2))
    for name, (i, j) in PAIRS.items():
        expected = spearmanr(frame.iloc[:4][SCORERS[i]], frame.iloc[:4][SCORERS[j]]).statistic
        assert table.iloc[0][name] == pytest.approx(expected, abs=1e-15)
    summary = summarize(table)['original_ranking_eligible']
    assert summary['ranking_score__activity_logits']['macro_spearman'] == pytest.approx(-1.)
    assert summary['ranking_score__cosine_similarity']['defined_fraction'] == .5
    assert table.iloc[2][list(PAIRS)].isna().all()
    shifted = frame.copy()
    shifted['ranking_score'] = frame.ranking_score * 2 + frame.group_id.map({'T__A': 20, 'T__B': -200, 'T__C': 3})
    pd.testing.assert_frame_equal(table, group_agreement(shifted, dict(A=0, B=1, C=2)))


def test_endpoint_change_uses_mutually_defined_same_groups():
    first = group_agreement(fixture(), dict(A=0, B=1, C=2))
    last = first.copy()
    column = 'ranking_score__cosine_similarity'
    last.loc[0, column] = first.loc[0, column] + .1
    last.loc[1, column] = .9  # newly defined; cannot change matched endpoint mean
    change = matched_endpoint_changes(first, last)[column]
    assert change['matched_groups'] == 1
    assert change['mean_change'] == pytest.approx(.1)
    last.loc[0, 'component'] = 3
    with pytest.raises(ValueError, match='changed'):
        matched_endpoint_changes(first, last)
