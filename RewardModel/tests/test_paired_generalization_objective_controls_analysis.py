"""Exact objective interventions, matched contrasts and immutable report evidence."""
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization import objective_controls_analysis as module
from experiments.paired_generalization.run import sha256, write_json
from test_paired_generalization_control_plan import ROOT, noise_fixture


def primary(tmp_path):
    return module.control_plan.build_plan(ROOT, tmp_path/'outputs', noise_fixture())


@pytest.mark.parametrize('family,count', [('weights', 30), ('contrastive', 30), ('ranking', 42)])
def test_existing_family_grid_and_targeted_contrasts(tmp_path, family, count):
    jobs = module.validate_grid(primary(tmp_path), family)
    assert len(jobs) == count and all(j['seed'] == 42 for j in jobs)
    spec = module.FAMILIES[family]
    assert set(x for pair in spec['contrasts'].values() for x in pair) <= set(spec['arms'])
    if family == 'weights':
        assert all(reference == 'full' for reference, _ in spec['contrasts'].values())
        for j in jobs:
            assert tuple(j['config']['model'][k] for k in module.weight_selection.WEIGHT_KEYS) == module.weight_selection.WEIGHTS[j['arm']]
    if family == 'ranking':
        assert spec['contrasts']['uniform_minus_top_pair_weights'] == ('pairwise_top', 'pairwise_uniform')
        assert spec['contrasts']['noise_soft_minus_zero_margin_hard_targets'] == ('pairwise_hard_zero_margin', 'pairwise_noise_soft')
        assert spec['contrasts']['assay_balanced_minus_list_normalized'] == ('listwise_normalized', 'assay_balanced')
        for j in jobs:
            if j['arm'] == 'pairwise_noise_soft':
                assert j['config']['model']['ranking_pair_target_temperature'] == pytest.approx(math.sqrt(3.)*({'A': .78, 'B': .76}[j['regime']])/math.pi)


@pytest.mark.parametrize('damage', ['extra_arm', 'missing_fit', 'budget', 'other_loss', 'readout', 'full_mask', 'soft_temperature', 'noise'])
def test_grid_rejects_extra_settings_or_hidden_changes(tmp_path, damage):
    plan = primary(tmp_path); job = next(j for j in plan['tuning_jobs'] if j['arm'] == 'pairwise_noise_soft')
    if damage == 'extra_arm': plan['arms'].append(copy.deepcopy(next(a for a in plan['arms'] if a['name'] == 'pairwise_top')))
    elif damage == 'missing_fit': plan['tuning_jobs'].remove(job)
    elif damage == 'budget': job['config']['training']['num_train_epochs'] = 12
    elif damage == 'other_loss': job['config']['model']['classification_loss_weight'] = .5
    elif damage == 'readout': job['config']['model']['ranking_readout'] = 'cosine'
    elif damage == 'full_mask': next(j for j in plan['tuning_jobs'] if j['arm'] == 'full')['config']['model']['contrastive_mask_policy'] = 'none'
    elif damage == 'soft_temperature': job['config']['model']['ranking_pair_target_temperature'] = 999.
    else: plan['training_noise_rms']['A'] = 0.
    with pytest.raises(ValueError): module.validate_grid(plan, 'ranking')


@pytest.fixture(params=['weights', 'contrastive', 'ranking'])
def reports(tmp_path, monkeypatch, request):
    """Synthetic report assembly; canonical native scientific gates are separate."""
    family = request.param; spec = module.FAMILIES[family]
    root = tmp_path/'project'; root.mkdir(); work = root/'outputs'; directory = work/'controlled'
    monkeypatch.setattr(module, 'ROOT', root); monkeypatch.setattr(module, 'WORK', work)
    monkeypatch.setattr(module, 'software', lambda: dict(synthetic_reporting=True))
    plan = module.control_plan.build_plan(ROOT, work, noise_fixture()); write_json(directory/'plan.json', plan)
    code = root/'qualified.py'; code.write_text('synthetic training source')
    sources = {'qualified.py': sha256(code)}
    runtime = dict(world_size=4, sources=sources,
        signature=module.control_plan.digest_json(dict(sources=sources, plan_sha256=sha256(directory/'plan.json'))))
    locations = {}; records = {}; calls = []
    for number, arm in enumerate(spec['arms']):
        lr = 1. if arm == 'full' else .3; analyses = {}
        for identity, (regime, seed, multiplier) in module.controlled_summary.expected_jobs(arm, lr).items():
            output = work/'controlled/runs'/identity; write_json(output/'runtime_qualification.json', runtime)
            job = dict(id=identity, arm=arm, regime=regime, seed=seed, learning_rate_multiplier=multiplier, output_dir=str(output))
            path = work/'postselection_analysis'/identity/'synthetic/complete.json'
            rules = {rule: dict(checkpoint='early' if rule in ('best_loss', 'patience3_loss') else 'late') for rule in module.controlled_summary.RULES}
            record = dict(job=job, rules=rules, source=dict(software=dict(native_fixture=True), design_sha256='design', noise_summary_sha256='noise'))
            write_json(path, record); records[identity] = path; analyses[str(path)] = sha256(path)
            for checkpoint in ('early', 'late'):
                for split in ('val', 'test'):
                    score = .2+.01*number+.1*(regime == 'A')+.01*(seed-42)-.02*(checkpoint == 'early')
                    frame = pd.DataFrame(dict(group_id=['a', 'b', 'c'], component=[1, 1, 2], size=[3, 4, 1],
                        original_ranking_eligible=[True, True, False], spearman=[score, score+.1, np.nan]))
                    table = path.parent/checkpoint/split/'groups_ranking_score.parquet'; table.parent.mkdir(parents=True); frame.to_parquet(table, index=False)
        path = work/'controlled_summaries'/arm/'synthetic/complete.json'
        write_json(path, dict(source=dict(analysis_sha256=analyses, primary_plan_sha256=sha256(directory/'plan.json'), software=module.controlled_summary.software()),
            summary=dict(arm=arm, ranking_readout='ranking_head', selected_learning_rate_multiplier=lr,
                selected=dict(synthetic_test_score=999 if arm == 'full' else -999), tuning_candidates=dict(synthetic=True))))
        locations[arm] = path
    def canonical(arm, paths):
        calls.append(arm); assert set(paths) == set(module.read(locations[arm])['source']['analysis_sha256'])
        return locations[arm]
    monkeypatch.setattr(module.controlled_summary, 'summarize', canonical)
    selection = work/'weight_selection/static_weight_sweep/selection.json'
    if family == 'weights':
        candidates = []
        for arm in spec['arms']:
            for regime in ('A', 'B'):
                best_lr = 3. if regime == 'A' else (1. if arm == 'full' else .3)
                for lr in module.control_plan.LEARNING_RATE_MULTIPLIERS:
                    score = (.7 if arm == ('full' if regime == 'A' else 'bce_low') else .4)-.2*(lr != best_lr)
                    candidates.append(dict(arm=arm, regime=regime, learning_rate_multiplier=lr,
                        validation_macro_spearman=score, validation_defined_fraction=1.))
        write_json(selection, dict(primary_plan_sha256=sha256(directory/'plan.json'), historical_limit='Historical tuning unknown',
            **module.weight_selection.choose(candidates)))
        monkeypatch.setattr(module.weight_selection, 'destination', lambda: selection)
        monkeypatch.setattr(module.weight_selection, 'verify', lambda: module.read(selection))
    return dict(family=family, spec=spec, root=root, work=work, paths=list(locations.values()), records=records, calls=calls, code=code)


def test_reports_use_correct_references_signs_lr_roles_and_seed_spread(reports):
    family = reports['family']; path = module.analyze(family, reports['paths']); result = module.read(path)
    n = len(reports['spec']['arms']); assert result['accounting']['unique_total_fits'] == 10*n
    assert result['accounting']['all_families_unique_total_fits'] == 150
    assert len(result['source']['runtime_qualification_sha256']) == len(result['source']['analysis_sha256']) == 10*n
    for name, (reference, alternative) in reports['spec']['contrasts'].items():
        change = result['paired_changes'][name]['best_loss']['test']['B']['43']
        delta = .01*(reports['spec']['arms'].index(alternative)-reports['spec']['arms'].index(reference))
        assert change['mean_change'] == pytest.approx(delta) and change['unit_bootstrap']['units'] == 2
        assert change['reference']['arm'] == reference and change['alternative']['arm'] == alternative
        assert change['reference']['learning_rate_multiplier'] == (1. if reference == 'full' else .3)
        assert change['alternative']['learning_rate_multiplier'] == (1. if alternative == 'full' else .3)
        spread = result['paired_change_seed_spread'][name]['best_macro']['test']
        assert spread['B']['confirmation_only']['mean_change']['mean'] == pytest.approx(delta)
        assert spread['B']['confirmation_only']['mean_change']['total'] == 2
        assert spread['A_minus_B']['all_three']['mean_change']['mean'] == pytest.approx(0.)
    assert not result['test_used_for_selection'] and result['new_fits_created'] == 0 and not result['study_complete']
    if family == 'weights':
        choice = result['weight_selection_comparison']
        assert choice['choices']['A']['arm'] == 'full' and choice['choices']['A']['learning_rate_multiplier'] == 3.
        assert not choice['choices']['A']['matches_existing_confirmation_lr']
        assert choice['choices']['B']['arm'] == 'bce_low' and choice['choices']['B']['learning_rate_multiplier'] == .3
        assert choice['different_weights'] and choice['different_learning_rates']
        # The deliberately extreme synthetic test scores must never choose the family winner.
        assert result['settings']['full']['selected']['synthetic_test_score'] == 999
        assert result['source']['weight_selection_sha256'] == {str(module.weight_selection.destination()): sha256(module.weight_selection.destination())}
    else:
        assert result['weight_selection_comparison'] is None and result['source']['weight_selection_sha256'] == {}
    assert module.analyze(family, reports['paths']) == path and len(reports['calls']) == 2*n
    changed = copy.deepcopy(result); changed['paired_changes'] = {}; write_json(path, changed)
    with pytest.raises(ValueError, match='Completed objective-control report changed'): module.analyze(family, reports['paths'])


@pytest.mark.parametrize('damage', ['missing_summary', 'readout', 'confirmation_runtime', 'training_source', 'analysis_hash', 'cached_table'])
def test_invalid_or_changed_scientific_evidence_is_rejected(reports, damage):
    paths = list(reports['paths']); family = reports['family']; arm = reports['spec']['arms'][-1]
    identity = module.control_plan.job_id(arm, 'B', 43, .3); record = module.read(reports['records'][identity])
    if damage == 'missing_summary': paths.pop()
    elif damage == 'readout':
        r = module.read(paths[-1]); r['summary']['ranking_readout'] = 'cosine'; write_json(paths[-1], r)
    elif damage == 'confirmation_runtime':
        p = Path(record['job']['output_dir'])/'runtime_qualification.json'; r = module.read(p); r['world_size'] = 2; write_json(p, r)
    elif damage == 'training_source': reports['code'].write_text('changed')
    elif damage == 'analysis_hash': reports['records'][identity].write_text('{}')
    else:
        r = module.read(module.analyze(family, paths)); Path(next(iter(r['group_tables_sha256']))).write_bytes(b'changed')
    with pytest.raises(ValueError): module.analyze(family, paths)


def test_native_completion_gate_is_not_bypassed(reports, monkeypatch):
    def reject(*args): raise ValueError('Native evaluation is not complete')
    monkeypatch.setattr(module.controlled_summary, 'summarize', reject)
    with pytest.raises(ValueError, match='Native evaluation'): module.analyze(reports['family'], reports['paths'])
    assert not list((reports['work']/'objective_controls_analysis').glob('*/*/complete.json'))


def test_child_analysis_mutation_during_reporting_is_rejected(reports, monkeypatch):
    original = module.controlled_summary.paired_group_change; changed = []
    def mutate(*args):
        if not changed:
            p = next(iter(reports['records'].values())); p.write_text('{}'); changed.append(p)
        return original(*args)
    monkeypatch.setattr(module.controlled_summary, 'paired_group_change', mutate)
    with pytest.raises(ValueError, match='evidence changed during reporting'): module.analyze(reports['family'], reports['paths'])
    assert not list((reports['work']/'objective_controls_analysis').glob('*/*/complete.json'))


@pytest.mark.parametrize('family', list(module.FAMILIES))
def test_preflight_adds_no_fit_selection_or_report(tmp_path, monkeypatch, family):
    work = tmp_path/'outputs'; monkeypatch.setattr(module, 'WORK', work)
    write_json(work/'controlled/plan.json', primary(tmp_path))
    before = {str(p): sha256(p) for p in work.rglob('*') if p.is_file()}
    result = module.preflight(family)
    assert all(not paths for paths in result['summary_candidates'].values())
    assert not result['scientific_comparison_created'] and result['new_fits_created'] == 0
    assert {str(p): sha256(p) for p in work.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('damage', ['unfrozen_selection', 'plan', 'B_lr', 'mid_report_mutation'])
@pytest.mark.parametrize('reports', ['weights'], indirect=True)
def test_weight_selection_provenance_and_lr_roles(reports, monkeypatch, damage):
    selection = module.weight_selection.destination()
    if damage == 'unfrozen_selection':
        def reject(): raise ValueError('Frozen validation selection missing or changed')
        monkeypatch.setattr(module.weight_selection, 'verify', reject)
    elif damage in ('plan', 'B_lr'):
        record = module.read(selection)
        if damage == 'plan': record['primary_plan_sha256'] = 'another-plan'
        else: record['within_setting']['B']['full']['learning_rate_multiplier'] = 3.
        write_json(selection, record)
    else:
        original = module.controlled_summary.paired_group_change
        def mutate(*args):
            selection.write_text('{}')
            return original(*args)
        monkeypatch.setattr(module.controlled_summary, 'paired_group_change', mutate)
    with pytest.raises(ValueError): module.analyze('weights', reports['paths'])
    assert not list((reports['work']/'objective_controls_analysis').glob('*/*/complete.json'))


@pytest.mark.parametrize('damage', ['reference_weight', 'second_weight', 'ranking_weight', 'sampler'])
def test_weight_grid_rejects_hidden_changes(tmp_path, damage):
    plan = primary(tmp_path)
    job = next(j for j in plan['tuning_jobs'] if j['arm'] == ('full' if damage == 'reference_weight' else 'bce_low'))
    if damage == 'reference_weight': job['config']['model']['classification_loss_weight'] = .3
    elif damage == 'second_weight': job['config']['model']['contrastive_loss_weight'] = 2.
    elif damage == 'ranking_weight': job['config']['model']['ranking_loss_weight'] = .5
    else: job['config']['data']['synthetic_sampler_change'] = True
    with pytest.raises(ValueError): module.validate_grid(plan, 'weights')
