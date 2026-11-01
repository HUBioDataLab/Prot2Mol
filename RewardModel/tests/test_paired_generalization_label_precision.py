import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization.analyze_predictions import rescore
from experiments.paired_generalization.label_precision import POLICY, read_predictions, trainer_label_frame
from experiments.paired_generalization.noise_band_audit import score_bands
from experiments.paired_generalization.scorer_agreement import group_agreement
from reward_model.training.evaluation import _compute_groupwise_spearman_with_records


def boundary_frame():
    return pd.DataFrame(dict(
        group_id=['x__a']*3+['x__b']*3+['x__c']*3,
        pchembl_value=[7.9, 8.15, 8.4, 3.6, 3.8, 4.1, 5., 5.+1e-8, 6.],
        ranking_score=[3., 1., 2.]*3, activity_logits=[1., 2., 3.]*3,
        cosine_similarity=[2., 1., 3.]*3, binary_label=[0, 0, 1]*3))


def test_rounding_changes_boundary_membership_and_ties_but_preserves_saved_outputs(tmp_path):
    frame = boundary_frame()
    source = tmp_path/'predictions.parquet'
    frame.to_parquet(source, index=False)
    corrected, provenance = read_predictions(source, True)
    pd.testing.assert_frame_equal(frame.drop(columns='pchembl_value'), corrected.drop(columns='pchembl_value'))
    pd.testing.assert_frame_equal(frame, pd.read_parquet(source))
    assert provenance['label_precision'] == POLICY
    assert corrected.pchembl_value.dtype == np.float64
    assert corrected.pchembl_value.iloc[6] == corrected.pchembl_value.iloc[7]
    raw = group_agreement(frame, dict(a=0, b=1, c=2))
    updated = group_agreement(corrected, dict(a=0, b=1, c=2))
    assert raw.original_eligible.tolist() == [True, False, True]
    assert updated.original_eligible.tolist() == [False, True, True]


def test_all_three_rescored_pathways_match_native_trainer_label_arithmetic():
    frame = trainer_label_frame(boundary_frame())
    summaries, records = rescore(frame, dict(a=0, b=1, c=2))
    for scorer in ['ranking_score', 'activity_logits', 'cosine_similarity']:
        native, groups = _compute_groupwise_spearman_with_records(
            group_ids=frame.group_id, ranking_scores=frame[scorer], pchembl_values=frame.pchembl_value,
            min_group_size=3, min_pchembl_span=.5)
        assert summaries[scorer]['original_eligible_macro_spearman'] == pytest.approx(native['eval_macro_spearman'], abs=1e-14)
        assert records[scorer].query('original_ranking_eligible').group_id.tolist() == [r['group_id'] for r in groups]


def test_noise_band_recount_uses_same_rounded_labels_without_changing_threshold():
    frame = boundary_frame().iloc[:3].assign(pchembl_value=[1., 1.1, 1.2])
    raw, _ = score_bands(frame, {'fixed': .2}, ['ranking_score'])
    corrected, _ = score_bands(trainer_label_frame(frame), {'fixed': .2}, ['ranking_score'])
    assert raw['fixed/ranking_score']['comparable_pairs'] == 0
    assert corrected['fixed/ranking_score']['comparable_pairs'] == 1
    assert np.ptp(trainer_label_frame(frame).pchembl_value) > .2
