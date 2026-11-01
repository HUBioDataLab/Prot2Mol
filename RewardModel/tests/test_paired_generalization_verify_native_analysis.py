import copy
import json

import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization import native_analysis as audit
from experiments.paired_generalization import verify_native_analysis as verify
from experiments.paired_generalization.run import sha256
from experiments.paired_generalization.similarity_composition import descriptors, composition_weights
from test_paired_generalization_native_analysis import snapshot


def test_independent_rank_and_noise_arithmetic_handles_ties_constants_and_band_boundary():
    assert np.isnan(verify.correlation([1], [2]))
    assert np.isnan(verify.correlation([1, 2, 3], [0, 0, 0]))
    assert verify.correlation([1, 1, 2], [3, 3, 2]) == pytest.approx(-1.)
    assert verify.correlation([1, 2, 3], [1, 3, 2]) == pytest.approx(.5)
    assert verify.enumerated_pairs([0, .5, 1], [1, 1, 0], .5) == dict(
        all_pairs=3, comparable_pairs=1, concordant=0, discordant=1,
        prediction_ties=0, label_band_pairs=2)
    assert verify.enumerated_pairs([0, 1, 2], [1, 1, 2], 0) == dict(
        all_pairs=3, comparable_pairs=3, concordant=2, discordant=0,
        prediction_ties=1, label_band_pairs=0)


@pytest.fixture
def native_audit(tmp_path):
    path, reference, examples, _ = snapshot(tmp_path)
    scaffolds = pd.Series(['S']*3+['U']*6, index=reference.compound_id)
    unit_map = pd.Series(dict(a0=0, a1=1, a2=2))
    rows, groups, _ = descriptors(audit.trainer_label_frame(reference), ['m0'], scaffolds, unit_map)
    cells = composition_weights(groups, groups)
    output = tmp_path/'audit'
    result = audit.audit_snapshot(path, reference, examples._fingerprint, unit_map.to_dict(), output,
        prefix='eval', noise_rms=.7, provenance={}, regime='A', similarity_design=(rows, groups, cells))
    return result, output, reference, dict(fingerprint=examples._fingerprint), groups, cells


def test_actual_trainer_predictions_pass_independent_correlations_pairs_and_weighted_means(native_audit):
    result, output, reference, info, groups, cells = native_audit
    value = verify.verify_snapshot(result, output, reference, info, .7, groups, cells)
    assert value['scorer_group_correlations_checked'] == 9
    assert value['brute_force_band_group_checks'] == 6
    assert value['composition_standardized'] == pytest.approx(value['macros']['ranking_score'])
    assert value['rows'] == 9 and value['eligible_groups'] == 3


def test_numerical_corruption_is_detected_even_when_artifact_hash_is_updated(native_audit):
    result, output, reference, info, groups, cells = native_audit
    file = output/'groups_ranking_score.parquet'
    table = pd.read_parquet(file)
    table.loc[0, 'spearman'] = 0.12345
    table.to_parquet(file, index=False)
    result['artifacts_sha256'][file.name] = sha256(file)
    with pytest.raises(ValueError, match='Independent group correlations'):
        verify.verify_snapshot(result, output, reference, info, .7, groups, cells)


def test_row_order_labels_and_fingerprint_are_checked_against_frozen_reference(native_audit):
    result, _, reference, info, _, _ = native_audit
    with pytest.raises(ValueError, match='identity/order'):
        verify.snapshot_frame(result, reference.iloc[::-1], info)
    changed = reference.copy()
    changed.loc[0, 'pchembl_value'] += 1
    with pytest.raises(ValueError, match='float32 labels'):
        verify.snapshot_frame(result, changed, info)
    with pytest.raises(ValueError, match='metadata differs'):
        verify.snapshot_frame(result, reference, dict(fingerprint='other'))


def test_sampled_pair_and_composition_corruption_are_detected(native_audit):
    result, output, reference, info, groups, cells = native_audit
    changed = copy.deepcopy(result)
    changed['composition_standardization']['scores']['standardized']['score'] += .2
    with pytest.raises(ValueError, match='composition standardized'):
        verify.verify_snapshot(changed, output, reference, info, .7, groups, cells)
    file = output/'noise_reference_margin_ranking_score.parquet'
    table = pd.read_parquet(file)
    table.loc[0, 'comparable_pairs'] += 1
    table.to_parquet(file, index=False)
    result['artifacts_sha256'][file.name] = sha256(file)
    with pytest.raises(ValueError, match='enumerated noise-band'):
        verify.verify_snapshot(result, output, reference, info, .7, groups, cells)


def test_test_predictions_are_refused_before_any_file_read():
    with pytest.raises(ValueError, match='Only validation'):
        verify.snapshot_frame(dict(prefix='test'), None, None)


def test_earlier_checks_require_identical_prefix_and_complete_epoch_partition_coverage(tmp_path):
    old = dict(analysis_plan_sha256='plan', trajectory=[dict(epoch=1., eval={'score': .2}, train_eval={'score': .3})])
    data = dict(analysis_plan_sha256='plan', trajectory=old['trajectory']+[dict(epoch=2.)])
    old_path = tmp_path/'trajectory_through_epoch01.json'
    old_path.write_text(json.dumps(old))
    record = dict(analysis_plan_sha256='plan', trajectory_sha256=sha256(old_path),
        source_and_artifact_hashes_verified=True, test_predictions_read=False,
        checks=[dict(epoch=1., prefix=p) for p in ('eval', 'train_eval')])
    path = tmp_path/'verification_through_epoch01.json'
    path.write_text(json.dumps(record))
    previous, digest, checks = verify.inherit_previous(tmp_path, data, 1)
    assert previous == old['trajectory'] and digest == sha256(path) and checks == record['checks']
    changed = copy.deepcopy(data)
    changed['trajectory'][0]['eval']['score'] += .1
    with pytest.raises(ValueError, match='prefix differs'):
        verify.inherit_previous(tmp_path, changed, 1)
    record['checks'].pop()
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='both snapshots'):
        verify.inherit_previous(tmp_path, data, 1)
    with pytest.raises(ValueError, match='earlier verified prefix'):
        verify.inherit_previous(tmp_path, data, 2)
