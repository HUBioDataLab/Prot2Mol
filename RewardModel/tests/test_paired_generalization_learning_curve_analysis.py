"""E41 analysis qualification: native tiny predictions, cohort identity and slopes.

Synthetic curve scores qualify reporting only; they are not experiment results.
"""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr, kendalltau

from experiments.paired_generalization import learning_curve_analysis as module
from experiments.paired_generalization.run import write_json, sha256
from test_paired_generalization_selected_analysis import native_pair
from test_paired_generalization_controlled_summary import records_fixture


def test_actual_native_core_matches_independent_metrics_and_ignores_full_support(native_pair):
    evaluations, refs = native_pair
    result, tables = module.core_diagnostics(evaluations, refs, 'A')
    for split in ('val', 'test'):
        frame, _ = module.selected_analysis.read_native_predictions(evaluations[split]['prediction_manifest'])
        for scorer in module.selected_analysis.SCORERS:
            rhos, taus = [], []
            for _, group in frame.groupby('group_id'):
                rhos.append(spearmanr(group.pchembl_value, group[scorer]).statistic)
                taus.append(kendalltau(group.pchembl_value, group[scorer]).statistic)
            values = result['partitions'][split]['metrics'][scorer]
            assert values[module.MACRO] == pytest.approx(np.nanmean(rhos))
            assert values['macro_kendall_tau_b'] == pytest.approx(np.nanmean(taus))
            assert values['unit_bootstrap_original_eligible']['units'] == 3
            assert len(tables[f'{split}/groups_{scorer}']) == 3
        assert 'similarity' not in result['partitions'][split]
        # These fields describe full training support; curve metrics must not use them.
        refs[split]['groups'] = pd.DataFrame({'wrong_full_support': [999]})
        refs[split]['cells'] = None
    repeat, _ = module.core_diagnostics(evaluations, refs, 'A')
    assert repeat == result
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('damage', ['fingerprint', 'labels', 'units', 'readout', 'native_metric'])
def test_changed_native_inputs_cannot_enter_curve_metrics(native_pair, damage):
    evaluations, refs = native_pair
    if damage == 'fingerprint':
        refs['val']['fingerprint'] = 'wrong'
    elif damage == 'labels':
        refs['val']['rows'].loc[0, 'pchembl_value'] += .1
    elif damage == 'units':
        refs['val']['rows']['unit_id'] = refs['test']['rows'].unit_id.iloc[0]
    else:
        path = Path(evaluations['val']['prediction_manifest']); record = module.read(path)
        if damage == 'readout':
            record['ranking_readout'] = 'cosine'
        else:
            record['metrics']['eval_macro_spearman'] += .1
        write_json(path, record)
    with pytest.raises((ValueError, AssertionError)):
        module.core_diagnostics(evaluations, refs, 'A')


def synthetic_curves():
    records, decision = records_fixture()
    base = module.controlled_summary.summarize_records(records, decision, 'selection')
    summaries, stats = {}, {'A': {}, 'B': {}}
    for f, effect, lr, a_rows, b_rows, units in zip(module.FRACTIONS, (0., .2, .1, .4),
            (.3, 1., 3., 1.), (1000, 2500, 5000, 10000), (9900, 9910, 9950, 10000), (10, 25, 50, 100)):
        s = copy.deepcopy(base); s['selected_learning_rate_multiplier'] = lr
        for rule in module.controlled_summary.RULES:
            for split in ('val', 'test'):
                for scorer in module.selected_analysis.SCORERS:
                    for regime in ('A', 'B'):
                        for point in s['selected'][rule][split][scorer][regime]['points'].values():
                            point['metrics'] = {k: v+effect for k, v in point['metrics'].items()}
        summaries[f] = s
        for regime, rows in (('A', a_rows), ('B', b_rows)):
            stats[regime][module.curve_plan.fraction_name(f)] = dict(rows=rows, units=units,
                row_fraction=rows/10000, largest_training_unit_included=regime == 'B')
    return summaries, stats


def test_slopes_use_actual_sizes_keep_dips_and_do_not_extrapolate():
    first = dict(score=.2, undefined=None); second = dict(score=.1, undefined=.5)
    result = module.segment(first, second, dict(rows=9900, units=10), dict(rows=10000, units=100))
    assert result['score_change']['score'] == pytest.approx(-.1)
    assert result['per_doubling']['rows']['score'] == pytest.approx(-.1/math.log2(10000/9900))
    assert result['per_doubling']['units']['score'] == pytest.approx(-.1/math.log2(10))
    assert result['per_doubling']['rows']['undefined'] is None
    assert abs(result['per_doubling']['rows']['score']) > 100*abs(result['per_doubling']['units']['score'])
    tied = module.segment(first, second, dict(rows=10, units=2), dict(rows=10, units=2))
    assert tied['score_change']['score'] == pytest.approx(-.1)
    assert tied['per_doubling']['rows']['score'] is None
    assert tied['per_doubling']['units']['score'] is None


@pytest.mark.parametrize('sizes', [(0, 1), (10, 9), (-1, 2), (1.5, 2)])
def test_invalid_training_sizes_cannot_define_slopes(sizes):
    with pytest.raises(ValueError, match='sizes'):
        module.segment(dict(x=.1), dict(x=.2), dict(rows=sizes[0], units=1), dict(rows=sizes[1], units=2))


def test_nonfinite_scores_and_missing_fractions_are_not_silently_filled():
    with pytest.raises(ValueError, match='nonfinite'):
        module.segment(dict(x=np.nan), dict(x=.2), dict(rows=1, units=1), dict(rows=2, units=2))
    with pytest.raises(ValueError, match='all four'):
        module.series([])
    summaries, stats = synthetic_curves(); summaries.pop(.25)
    with pytest.raises(ValueError, match='four summaries'):
        module.scalar_curves(summaries, stats)


def test_seed_roles_lr_choices_rules_and_all_metrics_remain_separate():
    summaries, stats = synthetic_curves()
    result = module.scalar_curves(summaries, stats)
    view = result['selected']['best_macro']['test']['ranking_score']['B']
    assert set(view['individual']) == {'42', '43', '44'}
    assert [p['learning_rate_multiplier'] for p in view['individual']['42']['points']] == [.3, 1., 3., 1.]
    assert view['individual']['42']['adjacent'][1]['score_change'][module.MACRO] == pytest.approx(-.1)
    for mean, a, b in zip(view['confirmation_mean']['points'],
            view['individual']['43']['points'], view['individual']['44']['points']):
        assert mean['metrics'][module.MACRO] == pytest.approx((a['metrics'][module.MACRO]+b['metrics'][module.MACRO])/2)
        assert set(mean['coverage_by_seed']) == {'43', '44'}
    assert set(view['individual']['42']['points'][0]['metrics']) == set(module.METRICS)
    assert len(view['slope_seed_spread']['rows']) == 3
    assert set(result['selected']) == set(module.controlled_summary.RULES)
    for lr, data in result['same_lr_seed42'].items():
        assert {p['learning_rate_multiplier'] for p in data['val']['ranking_score']['A']['points']} == {float(lr)}
    json.dumps(result, allow_nan=False)


def group_tables():
    base = pd.DataFrame(dict(group_id=['a', 'b', 'c', 'd'], component=[1, 2, 3, 4],
        size=[3, 3, 3, 1], original_ranking_eligible=[True, True, True, False]))
    scores = [[.1, .2, .3, np.nan], [.3, np.nan, .5, np.nan],
              [.4, .6, np.nan, np.nan], [.7, .8, .9, np.nan]]
    return {f: base.assign(spearman=v).iloc[::-1] for f, v in zip(module.FRACTIONS, scores)}


def test_matched_curve_uses_four_way_cohort_and_preserves_zero_eligible_units():
    tables = group_tables(); original = {f: frame.copy() for f, frame in tables.items()}
    sizes = {f: dict(rows=int(f*1000), units=int(f*100)) for f in module.FRACTIONS}
    result = module.matched_curve(tables, sizes)
    assert result['common_cohort']['eligible_groups'] == 3
    assert result['common_cohort']['common_defined_eligible_groups'] == 1
    assert result['common_cohort']['units'] == 4
    assert result['common_cohort']['means'] == dict(zip(module.FRACTIONS, [.1, .3, .4, .7]))
    for i, change in enumerate((.2, .1, .3)):
        item = result['adjacent'][i]
        assert item['paired_change']['mean_change'] == pytest.approx(change)
        assert item['paired_change']['unit_bootstrap']['units'] == 4
        assert item['paired_change']['mutually_defined_groups'] == 1
        assert item['conditional_unit_slope_interval']['rows'] == pytest.approx(
            [change/item['log2_size_change']['rows']]*2)
    for f in tables:
        pd.testing.assert_frame_equal(tables[f], original[f])


@pytest.mark.parametrize('field', ['group_id', 'component', 'size', 'original_ranking_eligible'])
def test_changed_groups_units_sizes_or_eligibility_reject_pairing(field):
    tables = group_tables()
    tables[.5].loc[0, field] = {'group_id': 'different', 'component': 9, 'size': 7,
        'original_ranking_eligible': False}[field]
    sizes = {f: dict(rows=int(f*1000), units=int(f*100)) for f in module.FRACTIONS}
    with pytest.raises(ValueError, match='differ'):
        module.matched_curve(tables, sizes)


def test_empty_common_cohort_produces_null_slopes_not_zero():
    tables = group_tables(); tables[1.]['spearman'] = np.nan
    sizes = {f: dict(rows=int(f*1000), units=int(f*100)) for f in module.FRACTIONS}
    result = module.matched_curve(tables, sizes)
    assert all(x is None for x in result['common_cohort']['means'].values())
    assert result['adjacent'][0]['score_change'][module.MACRO] is None
    assert result['adjacent'][0]['conditional_unit_slope_interval']['rows'] is None


def test_job_analysis_persists_real_tables_and_rejects_corruption(native_pair, tmp_path, monkeypatch):
    evaluations, refs = native_pair
    monkeypatch.setattr(module, 'WORK', tmp_path)
    plan = dict(job=dict(id='fixture', regime='A', fraction=.1), rules={'best_macro': dict(checkpoint='checkpoint-1')},
        seed_role='tuning_exposed', selected_learning_rate=False, selection_sha256='selection',
        training_view={'provenance': 'synthetic'},
        checkpoints=[dict(id='checkpoint-1', weights_sha256='fixture')])
    source = dict(metric_fixture='actual tiny model; synthetic plan, not GPU evidence')
    monkeypatch.setattr(module, 'analysis_inputs', lambda _: (plan, {'checkpoint-1': evaluations}, refs, source))
    path = module.analyze_job('fixture')
    result = module.verify_record(path, plan, source)
    assert len(result['artifacts_sha256']) == 7
    assert module.analyze_job('fixture') == path
    table = path.parent/'checkpoint-1/test/groups_ranking_score.parquet'
    table.write_bytes(table.read_bytes()+b'corruption')
    with pytest.raises(ValueError, match='artifact changed'):
        module.analyze_job('fixture')


def test_native_completion_is_required_before_any_prediction_or_reference_read(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'WORK', tmp_path)
    def incomplete(*args, **kwargs):
        raise ValueError('native evaluation incomplete')
    def forbidden(*args, **kwargs):
        raise AssertionError('must not read held-out references')
    monkeypatch.setattr(module.evaluation, 'verify_complete', incomplete)
    monkeypatch.setattr(module.selected_analysis, 'references', forbidden)
    with pytest.raises(ValueError, match='native evaluation incomplete'):
        module.analysis_inputs('valid-name')


def test_report_rejects_missing_fits_before_consuming_curve_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'WORK', tmp_path)
    write_json(tmp_path/'controlled/plan.json', {})
    monkeypatch.setattr(module.evaluation, 'prepared', lambda: ({}, {}))
    monkeypatch.setattr(module.evaluation, 'shared_endpoint', lambda _: dict(job_ids=['full']))
    monkeypatch.setattr(module.variant_relation, 'verify_summary', lambda *args: (
        dict(summary=dict(ranking_readout='ranking_head'), source=dict(primary_plan_sha256=sha256(tmp_path/'controlled/plan.json'))),
        {'full': 'not read'}))
    with pytest.raises(ValueError, match='thirty distinct'):
        module.analyze([], tmp_path/'endpoint.json')


def test_plots_use_actual_rows_and_units_without_generating_new_fits(tmp_path):
    summaries, stats = synthetic_curves()
    curves = module.scalar_curves(summaries, stats)
    module.draw(curves, tmp_path)
    assert (tmp_path/'learning_curves.png').stat().st_size > 1000
    assert (tmp_path/'learning_curves.svg').stat().st_size > 1000

def test_complete_report_assembly_requires_all_fractions_and_preserves_artifacts(tmp_path, monkeypatch):
    """Synthetic records exercise assembly; native provenance gates have separate tests."""
    monkeypatch.setattr(module, 'WORK', tmp_path)
    for name in ('controlled', 'learning_curves'):
        write_json(tmp_path/name/'plan.json', {})
    summaries, statistics = synthetic_curves()
    all_locations = {}; curve_paths = []
    full_records, full_decision = records_fixture()
    for fraction in module.FRACTIONS:
        records, decision = records_fixture()
        arm = 'full' if fraction == 1. else 'unit_curve_'+module.curve_plan.fraction_name(fraction)
        decision['arm'] = arm
        decision['plan_sha256'] = 'primary' if fraction == 1. else 'curve'
        if fraction < 1.:
            selection = {k: v for k, v in decision.items() if k != 'arm'}
            path = tmp_path/'learning_curves/selections'/(module.curve_plan.fraction_name(fraction)+'.json')
            write_json(path, selection)
            selection_hash = sha256(path)
        else:
            selection_hash = 'selection'
        for record in records:
            job = record['job']
            job.update(arm=arm, fraction=fraction, plan_sha256=decision['plan_sha256'])
            job['id'] = module.control_plan.job_id(arm, job['regime'], job['seed'], job['learning_rate_multiplier'])
            record['selection_sha256'] = selection_hash
            path = tmp_path/'analyses'/job['id']/'complete.json'
            write_json(path, record)
            for checkpoint in record['checkpoints']:
                for split in ('val', 'test'):
                    target = path.parent/checkpoint/split/'groups_ranking_score.parquet'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    group_tables()[fraction].to_parquet(target, index=False)
            all_locations[job['id']] = (path, record)
            if fraction < 1.:
                curve_paths.append(path)
        if fraction == 1.:
            full_records = records
            full_decision = decision
    full_summary = module.controlled_summary.summarize_records(full_records, full_decision, 'selection')
    endpoint_path = tmp_path/'endpoint.json'
    full = dict(summary=full_summary, source=dict(primary_plan_sha256=sha256(tmp_path/'controlled/plan.json')))
    write_json(endpoint_path, full)
    endpoint_locations = {k: v for k, v in all_locations.items() if v[1]['job']['fraction'] == 1.}
    monkeypatch.setattr(module.evaluation, 'prepared', lambda: ({}, {}))
    monkeypatch.setattr(module.evaluation, 'shared_endpoint', lambda _: dict(job_ids=list(endpoint_locations)))
    monkeypatch.setattr(module.variant_relation, 'verify_summary', lambda *args: (full, dict(endpoint_locations)))
    monkeypatch.setattr(module, 'analysis_inputs', lambda identity: (
        {}, {}, {}, {'core_metric_software': module.selected_analysis.software()}))
    monkeypatch.setattr(module, 'verify_record', lambda path, *args: module.read(path))
    monkeypatch.setattr(module, 'size_contract', lambda curve, primary, regime, fraction: dict(
        provenance=dict(statistics=statistics[regime][module.curve_plan.fraction_name(fraction)]), input_sha256={}))
    path = module.analyze(curve_paths, endpoint_path)
    report = module.read(path)
    assert report['payload']['accounting']['total_fits'] == 40
    assert report['payload']['accounting']['new_fits_created'] == 0
    assert len(report['source']['analysis_sha256']) == 30
    assert set(report['payload']['fraction_summaries']) == {'0.1', '0.25', '0.5', '1.0'}
    assert report['payload']['common_cohort_ranking']['best_macro']['test']['A']['42']['common_cohort']['common_defined_eligible_groups'] == 1
    assert module.analyze(curve_paths, endpoint_path) == path
    # A missing fraction cannot be concealed by duplicating a different analysis.
    wrong = [*curve_paths[:-1], curve_paths[0]]
    with pytest.raises(ValueError, match='thirty distinct'):
        module.analyze(wrong, endpoint_path)
    (path.parent/'learning_curves.svg').write_text('corrupt')
    with pytest.raises(ValueError, match='artifact changed'):
        module.analyze(curve_paths, endpoint_path)


@pytest.mark.parametrize('damage', [None, 'rows', 'units', 'row_fraction', 'unit_fraction'])
def test_reported_axes_must_match_selected_rows_and_whole_unit_counts(tmp_path, monkeypatch, damage):
    metadata = tmp_path/'metadata.parquet'; indices = tmp_path/'indices.npy'
    pd.DataFrame(dict(unit=['giant', 'giant', 'giant', 'small', 'other'])).to_parquet(metadata)
    np.save(indices, np.array([0, 1, 2, 3]))
    manifest = tmp_path/'view.json'
    write_json(manifest, dict(source_metadata=dict(path=str(metadata)), indices=dict(path=str(indices))))
    statistics = dict(rows=4, units=2, row_fraction=.8, unit_fraction=2/3)
    if damage:
        statistics[damage] += .1
    view = dict(provenance=dict(manifest_path=str(manifest), statistics=statistics))
    monkeypatch.setattr(module.evaluation, 'view_contract', lambda *args: view)
    if damage:
        with pytest.raises(ValueError, match='actual selected'):
            module.size_contract({}, {}, 'B', .1)
    else:
        assert module.size_contract({}, {}, 'B', .1) == view
