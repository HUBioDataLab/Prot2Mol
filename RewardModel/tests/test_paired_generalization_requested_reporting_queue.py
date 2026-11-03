"""Lifecycle qualification with simulated scientific evaluators; no model fits."""
import copy
import os
from pathlib import Path

import pytest

from experiments.paired_generalization import requested_reporting_queue as module
from experiments.paired_generalization.run import sha256, write_json


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root, work = tmp_path, tmp_path / 'outputs'
    source = root / 'RewardModel/experiments/paired_generalization/requested_reporting_queue.py'
    source.parent.mkdir(parents=True); source.write_text('qualified queue source fixture')
    probe_source = source.with_name('requested_probe_protocol.py'); probe_source.write_text('probe protocol source fixture')
    training_source = source.with_name('requested_single_seed.py'); training_source.write_text('training source fixture')
    monkeypatch.setattr(module, 'ROOT', root); monkeypatch.setattr(module, 'WORK', work)
    monkeypatch.setattr(module, '__file__', str(source))
    monkeypatch.setattr(module.probes, '__file__', str(probe_source))
    monkeypatch.setattr(module.evaluation.requested, '__file__', str(training_source))
    monkeypatch.setattr(module.cq, 'runtime_guard', lambda: None)
    monkeypatch.setattr(module.cq, 'boot_id', lambda: 'boot-fixture')
    jobs = [dict(id=f'{arm}__{r}__seed42__lr1p0', arm=arm, regime=r, seed=42,
        learning_rate_multiplier=1.) for arm in ('full', *module.evaluation.requested.ARMS) for r in ('A', 'B')]
    scope = dict(references=dict(jobs=jobs[:2]), jobs=jobs[2:])
    scope_path = work / 'controlled/training/plan.json'; write_json(scope_path, scope)
    identity = dict(pid=700, start_ticks=3000, uid=os.getuid(), cwd=str(root),
        argv=[str(root / '.venv/bin/python'), str(training_source), 'run'])
    processes = {700: identity}
    monkeypatch.setattr(module.cq, 'process_identity', lambda pid: processes.get(pid))
    write_json(scope_path.parent / 'invocation.json', dict(pid=700, plan_sha256=sha256(scope_path)))
    write_json(scope_path.parent / 'status.json', dict(phase='training_fixed_configuration'))
    monkeypatch.setattr(module.evaluation, 'scope', lambda: (scope_path, scope))
    protocol = work / 'tier4/protocol'; write_json(protocol / 'plan.json', dict(scope_plan_sha256=sha256(scope_path)))
    write_json(protocol / 'complete.json', dict(plan_sha256=sha256(protocol / 'plan.json')))
    monkeypatch.setattr(module.probes, 'destination', lambda: protocol)
    monkeypatch.setattr(module.probes, 'verify', lambda: module.read(protocol / 'plan.json'))
    monkeypatch.setattr(module.results, 'software', lambda: dict(source_sha256={}, packages={}))
    evidence = lambda selected: [dict(job_id=j['id'], verified_native_fixture=True) for j in selected]
    monkeypatch.setattr(module.cq, 'require_completed', evidence)
    def complete():
        write_json(scope_path.parent / 'complete.json', dict(
            phase='requested_fixed_config_training_complete_evaluations_and_probes_pending',
            plan_sha256=sha256(scope_path), evidence=evidence(scope['jobs']),
            training_fits_created=24, parameter_search=False, additional_seed_fits=0,
            test_scoring_complete=False, scientific_study_complete=False))
        processes.clear()
    calls, analyzed = [], []
    monkeypatch.setattr(module.evaluation, 'destination', lambda identity: work / 'evaluation' / identity)
    def batch(identities):
        assert (scope_path.parent / 'complete.json').exists()
        assert len(identities) <= 4
        calls.append(list(identities))
        files = {}
        for name in identities:
            path = module.evaluation.destination(name)
            write_json(path / 'plan.json', dict(identity=name))
            write_json(path / 'complete.json', dict(identity=name, native_fixture=True))
            files[str(path / 'complete.json')] = sha256(path / 'complete.json')
        return dict(phase='requested_best_macro_batch_complete', jobs=list(identities),
            new_fits=0, evidence_sha256=files)
    monkeypatch.setattr(module.evaluation, 'run_batch', batch)
    def verify_eval(path):
        record = module.read(path.parent / 'complete.json')
        assert record['identity'] == module.read(path)['identity']
    monkeypatch.setattr(module.evaluation, 'verify_complete', verify_eval)
    def analyze(name):
        analyzed.append(name); path = work / 'analysis' / name / 'complete.json'
        write_json(path, dict(job=next(j for j in jobs if j['id'] == name)))
        return path
    monkeypatch.setattr(module.results, 'analyze', analyze)
    monkeypatch.setattr(module.results, 'verify_analysis', module.read)
    def summarize(paths):
        assert [module.read(p)['job']['id'] for p in paths] == [j['id'] for j in jobs]
        path = work / 'summary/complete.json'; write_json(path, dict(fits=26)); return path
    monkeypatch.setattr(module.results, 'summarize', summarize)
    def gate(paths):
        assert len(paths) == len(set(paths)) == 26
        return dict(requested_fits=26, scientific_probes_complete=False)
    monkeypatch.setattr(module.probes, 'require_tier3', gate)
    path = module.prepare(700, 3000); module.validate(module.read(path))
    return dict(root=root, work=work, target=path.parent, plan=module.read(path), scope_path=scope_path,
        scope=scope, processes=processes, identity=identity, source=source,
        calls=calls, analyzed=analyzed, complete=complete)


def test_waits_without_gpu_work_then_exact_26_fixed_models(prepared, monkeypatch):
    waits = []
    def sleep(seconds):
        waits.append(seconds)
        assert not prepared['calls'] and not prepared['analyzed']
        assert module.read(prepared['target'] / 'status.json')['phase'] == 'waiting_for_requested_training'
        prepared['complete']()
    monkeypatch.setattr(module.time, 'sleep', sleep)
    done = module.run()
    assert waits == [600]
    expected = prepared['plan']['job_ids']
    assert [j for batch in prepared['calls'] for j in batch] == expected
    assert [len(b) for b in prepared['calls']] == [4, 4, 4, 4, 4, 4, 2]
    assert prepared['analyzed'] == expected
    assert done['new_fits'] == done['additional_seed_fits'] == 0
    assert not done['parameter_search'] and not done['scientific_study_complete']
    assert not done['test_used_for_selection'] and done['tier3_gate']['requested_fits'] == 26
    with pytest.raises(RuntimeError, match='already invoked'): module.run()
    assert len(prepared['calls']) == 7


@pytest.mark.parametrize('damage', ['exit', 'pid_reuse', 'stopped', 'incomplete_receipt'])
def test_failed_or_missing_training_never_starts_evaluation(prepared, damage):
    if damage == 'exit': prepared['processes'].clear()
    elif damage == 'pid_reuse': prepared['processes'][700] = {**prepared['identity'], 'start_ticks': 3001}
    elif damage == 'stopped': write_json(prepared['scope_path'].parent / 'status.json', dict(phase='stopped_for_inspection'))
    else:
        prepared['complete'](); path = prepared['scope_path'].parent / 'complete.json'
        record = module.read(path); record['evidence'].pop(); write_json(path, record)
    with pytest.raises((RuntimeError, ValueError)): module.run()
    assert not prepared['calls'] and not prepared['analyzed']
    assert module.read(prepared['target'] / 'status.json')['phase'] == 'stopped_for_inspection'
    assert not (prepared['target'] / 'complete.json').exists()


@pytest.mark.parametrize('damage', ['seed', 'lr', 'jobs', 'gpu', 'source', 'scope'])
def test_scope_drift_is_rejected_before_any_scientific_work(prepared, damage):
    plan = copy.deepcopy(prepared['plan'])
    if damage == 'seed': plan['seed'] = 43
    elif damage == 'lr': plan['learning_rate_multiplier'] = .3
    elif damage == 'jobs': plan['job_ids'].append('full__A__seed44__lr1p0')
    elif damage == 'gpu': plan['gpu_uuids'][0] = 'unauthorized'
    elif damage == 'source': prepared['source'].write_text('changed source')
    else: write_json(prepared['scope_path'], dict(changed=True))
    write_json(prepared['target'] / 'plan.json', plan)
    with pytest.raises(ValueError): module.run()
    assert not prepared['calls'] and not (prepared['target'] / 'invocation.json').exists()


@pytest.mark.parametrize('damage', ['owner', 'start', 'argv'])
def test_prepare_binds_exact_training_process(prepared, damage):
    record = dict(prepared['identity'])
    if damage == 'owner': record['uid'] += 1
    elif damage == 'start': record['start_ticks'] += 1
    else: record['argv'][-1] = 'legacy_queue'
    prepared['processes'][700] = record
    with pytest.raises(ValueError, match='exact live owned'): module.prepare(700, 3000)


@pytest.mark.parametrize('phase', sorted(module.WAIT_PHASES))
def test_resource_wait_does_not_duplicate_evaluation(prepared, monkeypatch, phase):
    prepared['complete'](); execute = module.evaluation.run_batch; once = [True]; waits = []
    def batch(identities):
        if once: once.pop(); return dict(phase=phase)
        return execute(identities)
    monkeypatch.setattr(module.evaluation, 'run_batch', batch)
    def sleep(seconds):
        waits.append(seconds); assert not prepared['calls']
    monkeypatch.setattr(module.time, 'sleep', sleep)
    module.run()
    assert waits == [600] and len(prepared['calls']) == 7


@pytest.mark.parametrize('failure', ['evaluation', 'analysis', 'interrupt', 'unknown_phase'])
def test_failed_work_stops_without_retry_or_final_claim(prepared, monkeypatch, failure):
    prepared['complete']()
    def fail(*_):
        if failure == 'interrupt': raise KeyboardInterrupt()
        raise RuntimeError('scientific worker failed; partial output preserved')
    if failure == 'analysis': monkeypatch.setattr(module.results, 'analyze', fail)
    elif failure == 'unknown_phase':
        monkeypatch.setattr(module.evaluation, 'run_batch', lambda _: dict(phase='legacy_grid_complete'))
    else: monkeypatch.setattr(module.evaluation, 'run_batch', fail)
    with pytest.raises((RuntimeError, ValueError, KeyboardInterrupt)): module.run()
    assert module.read(prepared['target'] / 'status.json')['phase'] == 'stopped_for_inspection'
    assert not (prepared['target'] / 'complete.json').exists()
    count = len(prepared['calls'])
    with pytest.raises(RuntimeError, match='already invoked'): module.run()
    assert len(prepared['calls']) == count


def test_normal_exit_race_requires_completion_receipt(prepared, monkeypatch):
    def exit_during_read(pid):
        prepared['complete'](); return None
    monkeypatch.setattr(module.cq, 'process_identity', exit_during_read)
    assert module.training_ready(prepared['plan'])
    assert module.upstream_completion(prepared['plan'])['evidence']
