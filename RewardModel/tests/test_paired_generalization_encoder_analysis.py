"""Encoder scope, trajectory provenance and cross-arm report assembly."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization import encoder_analysis as module
from experiments.paired_generalization.run import sha256, write_json
from test_paired_generalization_control_plan import ROOT, noise_fixture


def primary(tmp_path):
    return module.control_plan.build_plan(ROOT, tmp_path/'outputs', noise_fixture())


def write_trajectory(job):
    run = Path(job['output_dir']); history = []
    for epoch in range(1, 16):
        step = epoch*10
        for prefix in ('eval', 'train_eval'):
            val = prefix == 'eval'
            values = dict(loss=20.-epoch, ranking_loss=1.+abs(epoch-4)*.01,
                classification_loss=.4, contrastive_loss=18.-epoch,
                macro_spearman=.2+.01*min(epoch, 6)-.002*max(epoch-6, 0) if val else .1+.03*epoch,
                roc_auc=.7, spearman_num_groups=2., spearman_num_defined_groups=2., spearman_defined_fraction=1.)
            metrics = {f'{prefix}_{k}': v for k, v in values.items()}
            path = run/'prediction_exports/trajectory'/f'{prefix}_step{step:08d}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = path.with_suffix('.npz'); payload.write_bytes(b'hash-only synthetic native-prediction fixture')
            write_json(path, dict(format_version=1, phase='trajectory', metric_prefix=prefix,
                ranking_readout='ranking_head', epoch=epoch, step=step, rows=7,
                dataset_fingerprint=f"{job['regime']}_{job['seed']}_{prefix}",
                predictions_file=payload.name, predictions_sha256=sha256(payload), metrics=metrics))
            if val:
                history.append(dict(epoch=epoch, step=step, **metrics))
    write_json(run/'trainer_state.json', dict(epoch=15, global_step=150, log_history=history))


def test_grid_uses_only_the_existing_six_settings_and_equal_budgets(tmp_path):
    jobs = module.validate_grid(primary(tmp_path))
    assert len(jobs) == 36
    assert set(j['arm'] for j in jobs) == {'full', 'frozen', 'last_two_blocks', 'lora4', 'lora16', 'lora64'}


@pytest.mark.parametrize('damage', ['extra_rank', 'missing_fit', 'other_loss', 'budget', 'full_frozen', 'different_data', 'other_gpu'])
def test_grid_rejects_scope_or_comparability_changes(tmp_path, damage):
    plan = primary(tmp_path)
    job = next(j for j in plan['tuning_jobs'] if j['arm'] == 'frozen')
    if damage == 'extra_rank':
        arm = copy.deepcopy(next(a for a in plan['arms'] if a['name'] == 'lora4')); arm['name'] = 'lora8'; plan['arms'].append(arm)
    elif damage == 'missing_fit': plan['tuning_jobs'].remove(job)
    elif damage == 'other_loss': job['config']['model']['ranking_loss_weight'] = .5
    elif damage == 'budget': job['config']['training']['num_train_epochs'] = 12
    elif damage == 'full_frozen': next(j for j in plan['tuning_jobs'] if j['arm'] == 'full')['config']['model']['freeze_protein_encoder'] = True
    elif damage == 'different_data': job['config']['data']['sample_all_assays'] = False
    else: plan['gpu_uuids'][0] = 'GPU-not-authorized'
    with pytest.raises(ValueError): module.validate_grid(plan)


@pytest.fixture
def single(tmp_path):
    job = dict(id='full__A__seed42__lr1p0', regime='A', seed=42, output_dir=str(tmp_path/'run'))
    write_trajectory(job)
    return job


def test_complete_curve_distinguishes_minimum_first_increase_and_rank_peak(single):
    curve = module.trajectory(single)
    assert curve['loss_minimum_epochs'] == dict(loss=15, ranking_loss=4, classification_loss=1, contrastive_loss=15)
    assert curve['first_adjacent_loss_increases']['ranking_loss'] == [4, 5]
    assert curve['first_adjacent_loss_increases']['loss'] is None
    assert curve['validation_macro_peak_epoch'] == 6
    assert curve['final_validation_macro'] == pytest.approx(.242)
    assert curve['final_probe_minus_validation'] == pytest.approx(.308)
    assert len(curve['evidence_sha256']) == 61 and not curve['test_predictions_read']
    assert 'not an independent recomputation' in curve['verification_scope']


@pytest.mark.parametrize('damage', ['missing_epoch', 'duplicate_step', 'payload', 'native_disagreement', 'cohort', 'prefix', 'nonfinite_loss'])
def test_changed_or_incomplete_native_trajectory_is_rejected(single, damage):
    run = Path(single['output_dir']); state_path = run/'trainer_state.json'
    path = run/'prediction_exports/trajectory/eval_step00000020.json'
    if damage in ('missing_epoch', 'duplicate_step'):
        state = module.read(state_path)
        if damage == 'missing_epoch': state['log_history'].pop()
        else: state['log_history'][1]['step'] = 10
        write_json(state_path, state)
    elif damage == 'payload': path.with_suffix('.npz').write_bytes(b'changed')
    else:
        record = module.read(path)
        if damage == 'native_disagreement': record['metrics']['eval_macro_spearman'] = .9
        elif damage == 'cohort': record['dataset_fingerprint'] = 'another-population'
        elif damage == 'prefix': record['metric_prefix'] = 'test'
        else: record['metrics']['eval_loss'] = None
        write_json(path, record)
    with pytest.raises(ValueError): module.trajectory(single)


def test_undefined_macro_has_no_fabricated_peak_or_gap(single):
    rows = module.trajectory(single)['rows']
    for row in rows: row['val']['macro_spearman'] = None
    result = module.describe_curve(rows)
    assert result['validation_macro_peak_epoch'] is None
    assert result['final_minus_peak_macro'] is None and result['final_probe_minus_validation'] is None
    for row in rows: row['val']['macro_spearman'] = .2
    assert module.describe_curve(rows)['validation_macro_peak_epoch'] == 1


@pytest.fixture
def reports(tmp_path, monkeypatch):
    """Synthetic report IO; production native evaluation is deliberately not simulated as acceptance."""
    root = tmp_path/'project'; root.mkdir(); work = root/'outputs'; directory = work/'controlled'
    monkeypatch.setattr(module, 'ROOT', root); monkeypatch.setattr(module, 'WORK', work)
    monkeypatch.setattr(module, 'software', lambda: dict(synthetic_reporting=True))
    plan = module.control_plan.build_plan(ROOT, work, noise_fixture()); write_json(directory/'plan.json', plan)
    code = root/'qualified.py'; code.write_text('synthetic training source')
    sources = {'qualified.py': sha256(code)}
    runtime = dict(world_size=4, sources=sources,
        signature=module.control_plan.digest_json(dict(sources=sources, plan_sha256=sha256(directory/'plan.json'))))
    locations = {}; records = {}; calls = []
    for number, arm in enumerate(module.ARMS):
        lr = .3 if arm == 'frozen' else 1.
        analyses = {}
        for identity, (regime, seed, multiplier) in module.controlled_summary.expected_jobs(arm, lr).items():
            output = work/'controlled/runs'/identity
            job = dict(id=identity, arm=arm, regime=regime, seed=seed, learning_rate_multiplier=multiplier, output_dir=str(output))
            write_trajectory(job); write_json(output/'runtime_qualification.json', runtime)
            path = work/'postselection_analysis'/identity/'synthetic/complete.json'
            rules = {rule: dict(checkpoint='early' if rule in ('best_loss', 'patience3_loss') else 'late') for rule in module.controlled_summary.RULES}
            record = dict(job=job, rules=rules, source=dict(software=dict(native_fixture=True), design_sha256='design', noise_summary_sha256='noise'))
            write_json(path, record); records[identity] = path; analyses[str(path)] = sha256(path)
            for checkpoint in ('early', 'late'):
                for split in ('val', 'test'):
                    score = .2 + .01*number + .1*(regime == 'A') + .01*(seed-42) - .02*(checkpoint == 'early')
                    frame = pd.DataFrame(dict(group_id=['a', 'b', 'c'], component=[1, 1, 2], size=[3, 4, 1],
                        original_ranking_eligible=[True, True, False], spearman=[score, score+.1, np.nan]))
                    table = path.parent/checkpoint/split/'groups_ranking_score.parquet'; table.parent.mkdir(parents=True); frame.to_parquet(table, index=False)
        path = work/'controlled_summaries'/arm/'synthetic/complete.json'
        write_json(path, dict(source=dict(analysis_sha256=analyses, primary_plan_sha256=sha256(directory/'plan.json'), software=module.controlled_summary.software()),
            summary=dict(arm=arm, ranking_readout='ranking_head', selected_learning_rate_multiplier=lr,
                selected=dict(native_scorers=['ranking_score', 'activity_logit', 'cosine']), tuning_candidates=dict(synthetic=True))))
        locations[arm] = path
    def canonical(arm, paths):
        calls.append(arm)
        assert set(paths) == set(module.read(locations[arm])['source']['analysis_sha256'])
        return locations[arm]
    monkeypatch.setattr(module.controlled_summary, 'summarize', canonical)
    return dict(root=root, work=work, paths=list(locations.values()), records=records, calls=calls, code=code)


def test_six_arm_report_preserves_lr_roles_trajectories_and_paired_seed_spread(reports):
    path = module.analyze(reports['paths']); result = module.read(path)
    assert result['accounting']['unique_total_fits'] == 60 and result['accounting']['additional_encoder_fits'] == 50
    assert len(result['native_trajectories']) == 60 and len(result['source']['runtime_qualification_sha256']) == 60
    change = result['encoder_minus_full']['frozen']['best_loss']['test']['B']['43']
    assert change['mean_change'] == pytest.approx(.01) and change['unit_bootstrap']['units'] == 2
    assert change['reference']['learning_rate_multiplier'] == 1. and change['alternative']['learning_rate_multiplier'] == .3
    spread = result['paired_change_seed_spread']['lora64']['best_macro']['test']
    assert spread['B']['confirmation_only']['mean_change']['mean'] == pytest.approx(.05)
    assert spread['B']['confirmation_only']['mean_change']['total'] == 2
    assert spread['A_minus_B']['all_three']['mean_change']['mean'] == pytest.approx(0.)
    assert not result['test_used_for_selection'] and not result['study_complete'] and result['new_fits_created'] == 0
    assert module.analyze(reports['paths']) == path and len(reports['calls']) == 12
    changed = copy.deepcopy(result); changed['accounting']['unique_total_fits'] = 1; write_json(path, changed)
    with pytest.raises(ValueError, match='Completed encoder report changed'): module.analyze(reports['paths'])


@pytest.mark.parametrize('damage', ['missing_summary', 'readout', 'confirmation_runtime', 'training_source', 'analysis_hash', 'matched_probe', 'cached_payload'])
def test_changed_or_incomplete_comparison_evidence_is_rejected(reports, damage):
    paths = list(reports['paths']); identity = module.control_plan.job_id('frozen', 'B', 43, .3)
    record = module.read(reports['records'][identity]); output = Path(record['job']['output_dir'])
    if damage == 'missing_summary': paths.pop()
    elif damage == 'readout':
        result = module.read(paths[-1]); result['summary']['ranking_readout'] = 'cosine'; write_json(paths[-1], result)
    elif damage == 'confirmation_runtime':
        p = output/'runtime_qualification.json'; r = module.read(p); r['world_size'] = 2; write_json(p, r)
    elif damage == 'training_source': reports['code'].write_text('changed source')
    elif damage == 'analysis_hash': reports['records'][identity].write_text('{}')
    elif damage == 'matched_probe':
        for p in (output/'prediction_exports/trajectory').glob('train_eval_*.json'):
            r = module.read(p); r['dataset_fingerprint'] = 'different-but-internally-fixed'; write_json(p, r)
    else:
        result = module.read(module.analyze(paths)); p = next(Path(p) for p in result['artifact_sha256'] if p.endswith('.npz')); p.write_bytes(b'changed')
    with pytest.raises(ValueError): module.analyze(paths)


def test_native_scientific_gate_cannot_be_bypassed(reports, monkeypatch):
    def reject(*args): raise ValueError('Native selected evaluation is incomplete')
    monkeypatch.setattr(module.controlled_summary, 'summarize', reject)
    with pytest.raises(ValueError, match='Native selected evaluation'): module.analyze(reports['paths'])
    assert not list((reports['work']/'encoder_analysis').glob('*/complete.json'))


def test_preflight_is_read_only_and_does_not_claim_results(tmp_path, monkeypatch):
    work = tmp_path/'outputs'; monkeypatch.setattr(module, 'WORK', work)
    write_json(work/'controlled/plan.json', primary(tmp_path))
    before = {str(p): sha256(p) for p in work.rglob('*') if p.is_file()}
    result = module.preflight()
    assert len(result['tuning_jobs']) == 36 and all(not x for x in result['summary_candidates'].values())
    assert result['new_fits_created'] == 0 and not result['scientific_comparison_created']
    assert {str(p): sha256(p) for p in work.rglob('*') if p.is_file()} == before
