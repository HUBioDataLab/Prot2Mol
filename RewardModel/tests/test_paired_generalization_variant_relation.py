"""Numerical relations, common-cohort sensitivity and scientific provenance gates."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization import variant_relation as module
from experiments.paired_generalization.run import sha256, write_json
from test_paired_generalization_control_plan import ROOT, noise_fixture


def points(x, y):
    return [dict(arm=f'arm{i}', A=a, B=b, coverage={}) for i, (a, b) in enumerate(zip(x, y))]


def test_linear_inverse_and_missing_relations():
    rows = points([0., 1., 2., None], [1., 3., 5., .8])
    result = module.association(rows)
    assert result['points'] == rows
    assert result['statistics'] == dict(variants=4, defined_pairs=3, excluded_variants=['arm3'],
        pearson=1., spearman=1., slope_B_on_A=2., intercept_B_on_A=1., in_sample_r_squared=1.)
    stats = module.association(points([0., 1., 2.], [5., 3., 1.]))['statistics']
    assert stats['pearson'] == -1 and stats['spearman'] == -1
    assert stats['slope_B_on_A'] == -2 and stats['intercept_B_on_A'] == 5


def test_tied_ranks_constants_and_insufficient_pairs():
    stats = module.association(points([0., 0., 1., 2.], [0., 1., 1., 2.]))['statistics']
    # Average ranks centered cross product 3.75; both squared norms are 4.5.
    assert stats['spearman'] == pytest.approx(5/6)
    for x, y in [([1., 1.], [2., 3.]), ([1., 2.], [3., 3.]), ([None, 1.], [2., 3.])]:
        result = module.association(points(x, y))['statistics']
        assert result['pearson'] is None and result['spearman'] is None and result['in_sample_r_squared'] is None
    assert module.association(points([1., 1.], [2., 3.]))['statistics']['slope_B_on_A'] is None
    assert module.association(points([1., 2.], [3., 3.]))['statistics']['slope_B_on_A'] == 0
    with pytest.raises(ValueError, match='twice'):
        module.association([points([1], [2])[0]]*2)


def test_confirmation_means_exclude_tuning_and_require_both_confirmations():
    data = {'42': points([999., 999.], [-999., -999.]), '43': points([.1, None], [.4, .6]),
        '44': points([.3, .7], [.8, .8])}
    means = module.seed_views(data)['confirmation_mean']['points']
    assert means[0]['A'] == pytest.approx(.2) and means[0]['B'] == pytest.approx(.6)
    assert means[1]['A'] is None and means[1]['B'] == pytest.approx(.7)
    data.pop('44')
    with pytest.raises(ValueError, match='three seed roles'):
        module.seed_views(data)


def frame(values):
    return pd.DataFrame(dict(group_id=['a', 'b', 'c'], component=[1, 1, 2], size=[3, 4, 1],
        original_ranking_eligible=[True, True, False], spearman=values))


def test_common_cohort_excludes_undefined_assays_but_keeps_all_units_in_coverage():
    tables = {'one': frame([.2, .8, .9]), 'two': frame([.4, np.nan, np.nan]).iloc[::-1]}
    result = module.common_cohort(tables)
    assert result['means'] == {'one': .2, 'two': .4}
    assert result['groups'] == 3 and result['units'] == 2
    assert result['eligible_groups'] == 2 and result['common_defined_eligible_groups'] == 1
    assert result['own_defined_eligible_groups'] == {'one': 2, 'two': 1}
    tables['two'] = frame([np.nan, np.nan, .8])
    result = module.common_cohort(tables)
    assert result['means'] == {'one': None, 'two': None} and result['common_defined_eligible_groups'] == 0


@pytest.mark.parametrize('column,value', [('group_id', 'other'), ('component', 3), ('size', 6), ('original_ranking_eligible', False)])
def test_common_cohort_rejects_changed_assays_units_sizes_or_eligibility(column, value):
    second = frame([.3, .5, np.nan]); second.loc[0, column] = value
    with pytest.raises(ValueError, match='cohorts, units or eligibility'):
        module.common_cohort({'one': frame([.2, .4, np.nan]), 'two': second})


@pytest.fixture
def reports(tmp_path, monkeypatch):
    """Synthetic assembly fixture; canonical native verification is exercised separately."""
    root = tmp_path/'project'; work = root/'outputs'; root.mkdir()
    monkeypatch.setattr(module, 'ROOT', root); monkeypatch.setattr(module, 'WORK', work)
    monkeypatch.setattr(module, 'software', lambda: dict(synthetic=True, variant_plot_matplotlib='fixture'))
    primary = module.control_plan.build_plan(ROOT, work, noise_fixture()); primary_path = work/'controlled/plan.json'
    write_json(primary_path, primary)
    code = root/'qualified.py'; code.write_text('qualified fixture')
    sources = {'qualified.py': sha256(code)}
    runtime = dict(world_size=4, sources=sources,
        signature=module.control_plan.digest_json(dict(sources=sources, plan_sha256=sha256(primary_path))))
    paths = {}; records = {}; calls = []
    for i, definition in enumerate(primary['arms']):
        arm = definition['name']; lr = (.3, 1., 3.)[i % 3]; evidence = {}; index = {}
        for identity, (regime, seed, candidate) in module.controlled_summary.expected_jobs(arm, lr).items():
            output = work/'controlled/runs'/identity; write_json(output/'runtime_qualification.json', runtime)
            path = work/'postselection_analysis'/identity/'fixture/complete.json'
            record = dict(job=dict(id=identity, output_dir=str(output)),
                rules={r: dict(checkpoint='checkpoint') for r in module.controlled_summary.RULES},
                source=dict(software=dict(native=True), design_sha256='design', noise_summary_sha256='noise'))
            write_json(path, record); evidence[str(path)] = sha256(path); records[identity] = path
            value = (.1+.01*i if regime == 'A' else .5-.008*i)+.001*(seed-42)+.002*candidate
            point = dict(job_id=identity, checkpoint='checkpoint', metrics={m: value for m in module.METRICS},
                coverage=dict(groups=3, eligible_groups=2, eligible_defined_fraction=1.))
            index[regime, seed, candidate] = point
            for split in ('val', 'test'):
                table = path.parent/'checkpoint'/split/'groups_ranking_score.parquet'; table.parent.mkdir(parents=True)
                frame([value-.1, value+.1, np.nan]).to_parquet(table, index=False)
        selected = {rule: {split: dict(ranking_score={r: dict(points={str(s): index[r, s, lr] for s in (42, 43, 44)})
            for r in ('A', 'B')}) for split in ('val', 'test')} for rule in module.controlled_summary.RULES}
        candidates = {str(c): {split: dict(ranking_score={r: index[r, 42, c] for r in ('A', 'B')})
            for split in ('val', 'test')} for c in module.control_plan.LEARNING_RATE_MULTIPLIERS}
        path = work/'controlled_summaries'/arm/'fixture/complete.json'
        write_json(path, dict(source=dict(analysis_sha256=evidence, primary_plan_sha256=sha256(primary_path),
            software=module.controlled_summary.software()), summary=dict(arm=arm,
            ranking_readout=definition['model'].get('ranking_readout', 'ranking_head'),
            selected_learning_rate_multiplier=lr, selected=selected, tuning_candidates=candidates)))
        paths[arm] = path
    def canonical(arm, analyses):
        calls.append(arm)
        assert set(analyses) == set(module.read(paths[arm])['source']['analysis_sha256'])
        return paths[arm]
    monkeypatch.setattr(module.controlled_summary, 'summarize', canonical)
    def synthetic_plot(relations, definitions, destination, title):
        for ext in ('.png', '.svg'):
            destination.with_suffix(ext).write_text('SYNTHETIC assembly fixture '+title)
    monkeypatch.setattr(module, 'draw', synthetic_plot)
    return dict(root=root, work=work, primary=primary, paths=paths, records=records, calls=calls, code=code)


def test_complete_primary_report_keeps_all_metrics_rules_lrs_and_scopes(reports):
    path = module.analyze(list(reports['paths'].values())); saved = module.read(path); result = saved['result']
    assert len(reports['calls']) == 28 and result['accounting']['existing_primary_fits'] == 280
    assert len(result['selected']) == 5 and set(result['selected']['best_macro']['test']) == set(module.METRICS)
    assert set(result['same_lr_seed42']) == {'0.3', '1.0', '3.0'}
    for cell in result['same_lr_seed42'].values():
        assert cell['test'][module.MACRO]['statistics']['defined_pairs'] == 28
        assert cell['test'][module.MACRO]['statistics']['pearson'] == pytest.approx(-1)
    assert result['common_cohort_sensitivity']['best_macro']['test']['coverage']['44']['A']['units'] == 2
    assert result['common_cohort_sensitivity']['best_macro']['test']['relations']['confirmation_mean']['statistics']['defined_pairs'] == 28
    assert set(x['ranking_readout'] for x in result['variant_index'].values()) == {'ranking_head', 'activity_logit', 'cosine'}
    assert not result['test_used_for_selection'] and not result['e58_complete'] and not result['study_complete']
    assert result['new_fits_created'] == 0 and len(result['missing_all_tier3_families']) == 2
    assert module.analyze(list(reports['paths'].values())) == path
    changed = module.read(path); changed['result']['e58_complete'] = True; write_json(path, changed)
    with pytest.raises(ValueError, match='Completed variant report changed'):
        module.analyze(list(reports['paths'].values()))


@pytest.mark.parametrize('damage', ['missing_summary', 'duplicate_summary', 'readout', 'missing_confirmation',
    'analysis_hash', 'runtime', 'source', 'metric_protocol', 'cohort'])
def test_incomplete_or_changed_scientific_evidence_cannot_produce_a_plot(reports, damage):
    paths = list(reports['paths'].values())
    if damage == 'missing_summary':
        paths.pop()
    elif damage == 'duplicate_summary':
        paths[-1] = paths[0]
    elif damage == 'readout':
        record = module.read(paths[0]); record['summary']['ranking_readout'] = 'cosine'; write_json(paths[0], record)
    elif damage == 'missing_confirmation':
        record = module.read(paths[0]); name = next(p for p in record['source']['analysis_sha256'] if 'seed43' in p)
        record['source']['analysis_sha256'].pop(name); write_json(paths[0], record)
    elif damage == 'analysis_hash':
        next(iter(reports['records'].values())).write_text('{}')
    elif damage == 'runtime':
        record = module.read(next(iter(reports['records'].values())))
        runtime = Path(record['job']['output_dir'])/'runtime_qualification.json'
        changed = module.read(runtime); changed['world_size'] = 2; write_json(runtime, changed)
    elif damage == 'source':
        reports['code'].write_text('changed')
    elif damage == 'metric_protocol':
        summary = module.read(paths[0]); p = Path(next(iter(summary['source']['analysis_sha256'])))
        record = module.read(p); record['source']['noise_summary_sha256'] = 'different'; write_json(p, record)
        summary['source']['analysis_sha256'][str(p)] = sha256(p); write_json(paths[0], summary)
    else:
        identity = module.control_plan.job_id('full', 'A', 42, .3)
        p = reports['records'][identity].parent/'checkpoint/test/groups_ranking_score.parquet'
        changed = pd.read_parquet(p); changed.loc[0, 'component'] = 99; changed.to_parquet(p, index=False)
    with pytest.raises(ValueError):
        module.analyze(paths)
    assert not list((reports['work']/'variant_relation').glob('*/complete.json'))


def test_upstream_native_gate_and_changed_cached_artifacts_are_not_bypassed(reports, monkeypatch):
    original = module.controlled_summary.summarize
    def reject(*args):
        raise ValueError('Native qualification rejected')
    monkeypatch.setattr(module.controlled_summary, 'summarize', reject)
    with pytest.raises(ValueError, match='Native qualification'):
        module.analyze(list(reports['paths'].values()))
    monkeypatch.setattr(module.controlled_summary, 'summarize', original)
    path = module.analyze(list(reports['paths'].values()))
    (path.parent/'primary_selected_test.svg').write_text('changed')
    with pytest.raises(ValueError, match='artifact changed'):
        module.analyze(list(reports['paths'].values()))


def test_mid_report_source_changes_cannot_create_completion(reports, monkeypatch):
    original = module.draw
    def changing(*args):
        original(*args); reports['code'].write_text('changed during plots')
    monkeypatch.setattr(module, 'draw', changing)
    with pytest.raises(ValueError, match='evidence changed during reporting'):
        module.analyze(list(reports['paths'].values()))
    assert not list((reports['work']/'variant_relation').glob('*/complete.json'))


def test_registered_grid_rejects_scope_and_budget_changes_and_preflight_is_read_only(reports):
    assert len(module.validate_grid(reports['primary'])) == 28
    for damage in ('missing', 'budget', 'parameter', 'device'):
        primary = copy.deepcopy(reports['primary'])
        if damage == 'missing':
            primary['tuning_jobs'].pop()
        elif damage == 'budget':
            primary['tuning_jobs'][0]['config']['training']['num_train_epochs'] = 10
        elif damage == 'parameter':
            primary['tuning_jobs'][0]['config']['model']['ranking_loss_weight'] = .5
        else:
            primary['gpu_uuids'][0] = 'unauthorized'
        with pytest.raises(ValueError):
            module.validate_grid(primary)
    before = {str(p): sha256(p) for p in reports['work'].rglob('*') if p.is_file()}
    preflight = module.preflight()
    assert preflight['primary_variants'] == 28 and preflight['new_fits_created'] == 0
    assert not preflight['e58_complete'] and not preflight['scientific_comparison_created']
    assert {str(p): sha256(p) for p in reports['work'].rglob('*') if p.is_file()} == before


def test_plot_writes_standalone_raster_and_vector_artifacts(tmp_path):
    definitions = {a['name']: a for a in module.control_plan.control_arms()}
    data = {}
    for seed in (42, 43, 44):
        data[str(seed)] = [dict(arm=arm, A=.1+.02*i, B=.6-.015*i+.005*(seed-42), coverage={})
            for i, arm in enumerate(definitions)]
    destination = tmp_path/'synthetic_variants'
    module.draw(module.seed_views(data), definitions, destination, 'SYNTHETIC plotting check')
    assert destination.with_suffix('.png').read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
    text = destination.with_suffix('.svg').read_text()
    assert 'SYNTHETIC plotting check' in text and 'confirmation mean' in text and 'Split B test macro Spearman' in text
    assert destination.with_suffix('.png').stat().st_size > 10000

