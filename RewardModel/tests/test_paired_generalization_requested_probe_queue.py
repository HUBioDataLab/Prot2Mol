"""Bound upstream lifecycle and exact downstream orchestration, without GPU work."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.paired_generalization import requested_probe_queue as module
from experiments.paired_generalization.run import sha256, write_json


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root, work = tmp_path, tmp_path / 'outputs'
    source = root / 'RewardModel/experiments/paired_generalization/requested_probe_queue.py'
    source.parent.mkdir(parents=True); source.write_text('probe supervisor fixture')
    reporter_source = source.with_name('requested_reporting_queue.py'); reporter_source.write_text('reporter fixture')
    monkeypatch.setattr(module, 'ROOT', root); monkeypatch.setattr(module, 'WORK', work)
    monkeypatch.setattr(module, '__file__', str(source)); monkeypatch.setattr(module.reporting, '__file__', str(reporter_source))
    monkeypatch.setattr(module.cq, 'runtime_guard', lambda: None); monkeypatch.setattr(module.cq, 'boot_id', lambda: 'boot')
    monkeypatch.setattr(module, 'software', lambda: dict(source_sha256={str(source.relative_to(root)):sha256(source)}, packages={}))
    protocol = work / 'protocol'; write_json(protocol / 'plan.json', dict(accounting=dict(adaptation_support_fits=8088,
        geometry_exports=6, objective_batch_pairs=64)))
    write_json(protocol / 'complete.json', dict(plan_sha256=sha256(protocol / 'plan.json')))
    monkeypatch.setattr(module.protocol, 'destination', lambda: protocol)
    monkeypatch.setattr(module.protocol, 'verify', lambda: module.read(protocol / 'plan.json'))
    report = work / 'reporting'; ids = [f'fixed_job_{i}' for i in range(26)]
    report_plan = dict(job_ids=ids, probe_plan_sha256=sha256(protocol / 'plan.json'))
    write_json(report / 'plan.json', report_plan)
    monkeypatch.setattr(module.reporting, 'destination', lambda: report)
    monkeypatch.setattr(module.reporting, 'validate', lambda p: p == report_plan or (_ for _ in ()).throw(ValueError('changed reporting plan')))
    identity = dict(pid=700, start_ticks=3000, uid=os.getuid(), cwd=str(root),
        argv=[str(root / '.venv/bin/python'), str(reporter_source), 'run'])
    processes = {700:identity}
    monkeypatch.setattr(module.cq, 'process_identity', lambda pid: processes.get(pid))
    write_json(report / 'invocation.json', dict(pid=700, plan_sha256=sha256(report / 'plan.json')))
    write_json(report / 'status.json', dict(phase='waiting_for_requested_training'))
    upstream = dict(training_complete_fixture=True)
    monkeypatch.setattr(module.reporting, 'upstream_completion', lambda _: upstream)
    monkeypatch.setattr(module.reporting.results, 'software', lambda: dict(fixture=True))
    monkeypatch.setattr(module.reporting.evaluation, 'destination', lambda identity: work / 'evaluation' / identity)
    monkeypatch.setattr(module.reporting.evaluation, 'verify_complete', lambda path:
        module.read(path.parent / 'complete.json')['job_id'] == path.parent.name)
    monkeypatch.setattr(module.reporting.results.diagnostics, 'verify_artifacts', lambda folder, record: None)
    def gate(paths):
        assert len(paths) == len(set(paths)) == 26
        assert [module.read(p)['job_id'] for p in paths] == ids
        return dict(requested_fits=26, analysis_sha256={str(p):sha256(Path(p)) for p in paths}, scientific_probes_complete=False)
    monkeypatch.setattr(module.protocol, 'require_tier3', gate)
    def complete():
        evidence = []
        for identity in ids:
            ep = work / 'evaluation' / identity / 'complete.json'; ap = work / 'analysis' / identity / 'complete.json'
            write_json(ep, dict(job_id=identity)); write_json(ep.parent / 'plan.json', dict(job_id=identity)); write_json(ap, dict(job_id=identity))
            evidence.append(dict(job_id=identity, evaluation=str(ep), evaluation_sha256=sha256(ep), analysis=str(ap), analysis_sha256=sha256(ap)))
        selected = gate([r['analysis'] for r in evidence]); summary = work / 'requested_summary/fixture/complete.json'
        write_json(summary, dict(phase='requested_fixed_seed_comparison_complete', training_fits=26, new_fits=0,
            seed_uncertainty_available=False, test_used_for_selection=False,
            source=dict(analyses_sha256=selected['analysis_sha256'], software=dict(fixture=True))))
        write_json(report / 'upstream_completion.json', upstream)
        write_json(report / 'complete.json', dict(phase='requested_selected_evaluations_and_comparisons_complete_probes_pending',
            plan_sha256=sha256(report / 'plan.json'), upstream_sha256=sha256(report / 'upstream_completion.json'),
            evidence=evidence, summary=str(summary), summary_sha256=sha256(summary), tier3_gate=selected,
            new_fits=0, additional_seed_fits=0, parameter_search=False, test_used_for_selection=False, scientific_study_complete=False))
        processes.clear()
    calls = []
    def executor(name):
        phase = dict(adaptation='requested_adaptation_complete', geometry='requested_reference_geometry_complete',
            objectives='requested_reference_objective_probes_complete')[name]
        def prepare(paths):
            assert len(paths) == 26 and (report / 'complete.json').exists()
            calls.append(('prepare',name)); path = work / name / 'plan.json'; write_json(path, dict(fixture=True)); return path
        def verify_worker(path, regime):
            if name == 'geometry': return {}, {split:dict(analysis=dict(fixture=split)) for split in ('train','val','test')}
            return {}, dict(summary=dict(pairs=32)), dict(qualified=True)
        def run(path):
            calls.append(('run',name)); result = dict(phase=phase, plan_sha256=sha256(path), new_fits=0, scientific_study_complete=False)
            if name == 'adaptation':
                rp = path.parent / 'report.json'; write_json(rp, dict(new_support_fits=8088))
                result.update(report=str(rp), report_sha256=sha256(rp))
            else:
                result['evidence_sha256'] = {}; result['results'] = {}
                for regime in ('A','B'):
                    worker = path.parent / regime / 'complete.json'; write_json(worker, dict(fixture=True))
                    result['evidence_sha256'][regime] = sha256(worker)
                    values = verify_worker(path, regime)
                    result['results'][regime] = ({s:v['analysis'] for s,v in values[1].items()} if name == 'geometry'
                        else dict(summary=values[1]['summary'], qualification=values[2]))
                if name == 'objectives': result['test_data_read'] = False
            write_json(path.parent / 'complete.json', result); return result
        def verify_complete(path):
            value = module.read(path.parent / 'complete.json'); return value, module.read(value['report'])
        return SimpleNamespace(prepare=prepare, run=run, verify_plan=lambda p:module.read(p),
            verify_worker=verify_worker, verify_complete=verify_complete)
    executors = {name:executor(name) for name in module.ORDER}; monkeypatch.setattr(module, 'MODULES', executors)
    path = module.prepare(700, 3000); module.validate(module.read(path))
    return dict(root=root, work=work, target=path.parent, path=path, plan=module.read(path), report=report,
        identity=identity, processes=processes, source=source, complete=complete, calls=calls, executors=executors)


def test_waits_for_exact_report_then_runs_three_probes_in_order(prepared, monkeypatch):
    waits = []
    def sleep(seconds):
        waits.append(seconds); assert not prepared['calls']
        assert module.read(prepared['target'] / 'status.json')['phase'] == 'waiting_for_requested_reporting'
        prepared['complete']()
    monkeypatch.setattr(module.time, 'sleep', sleep)
    result = module.run()
    assert waits == [600]
    assert prepared['calls'] == [(action,name) for name in module.ORDER for action in ('prepare','run')]
    assert list(result['probes']) == list(module.ORDER)
    assert not result['scientific_study_complete'] and result['additional_primary_fits'] == result['additional_seed_fits'] == 0
    assert result['probes']['adaptation']['support_fits'] == 8088
    with pytest.raises(RuntimeError, match='already invoked'): module.run()


@pytest.mark.parametrize('damage', ['exit','pid_reuse','stopped','incomplete','test_selection','changed_analysis','wrong_summary'])
def test_bad_upstream_cannot_start_probes(prepared, damage):
    if damage == 'exit': prepared['processes'].clear()
    elif damage == 'pid_reuse': prepared['processes'][700] = {**prepared['identity'], 'start_ticks':3001}
    elif damage == 'stopped': write_json(prepared['report'] / 'status.json', dict(phase='stopped_for_inspection'))
    else:
        prepared['complete'](); path = prepared['report'] / 'complete.json'; value = module.read(path)
        if damage == 'incomplete': value['evidence'].pop()
        elif damage == 'test_selection': value['test_used_for_selection'] = True
        elif damage == 'changed_analysis': Path(value['evidence'][0]['analysis']).write_text('{}')
        else: value['summary'] = str(prepared['work'] / 'wrong.json')
        write_json(path,value)
    with pytest.raises((RuntimeError,ValueError,FileNotFoundError)): module.run()
    assert not prepared['calls'] and not (prepared['target'] / 'complete.json').exists()
    assert module.read(prepared['target'] / 'status.json')['phase'] == 'stopped_for_inspection'


@pytest.mark.parametrize('damage', ['seed','lr','order','gpu','source','inventory'])
def test_scope_drift_is_rejected_before_invocation(prepared, damage):
    plan = copy.deepcopy(prepared['plan'])
    if damage == 'seed': plan['seed'] = 43
    elif damage == 'lr': plan['learning_rate_multiplier'] = .3
    elif damage == 'order': plan['order'].reverse()
    elif damage == 'gpu': plan['gpu_uuids'][0] = 'unauthorized'
    elif damage == 'source': prepared['source'].write_text('changed')
    else: plan['support_fit_upper_bound'] += 1
    write_json(prepared['path'],plan)
    with pytest.raises(ValueError): module.run()
    assert not (prepared['target'] / 'invocation.json').exists() and not prepared['calls']


@pytest.mark.parametrize('field,value', [('uid',-1),('start_ticks',3001),('argv',['legacy'])])
def test_prepare_requires_exact_owned_process(prepared, field, value):
    prepared['processes'][700] = {**prepared['identity'], field:value}
    with pytest.raises(ValueError, match='exact live owned'): module.prepare(700,3000)


def test_transient_process_inspection_error_rechecks_same_handle_without_relaunch(prepared, monkeypatch):
    seen, waits = [], []
    def inspect(pid):
        seen.append(pid)
        if len(seen) == 1: raise OSError('transient proc observation')
        return prepared['processes'].get(pid)
    monkeypatch.setattr(module.cq, 'process_identity', inspect)
    def sleep(seconds):
        waits.append(seconds); assert not prepared['calls']
        if len(waits) == 1:
            assert module.read(prepared['target'] / 'status.json')['phase'] == 'waiting_for_reporting_observation'
        else: prepared['complete']()
    monkeypatch.setattr(module.time, 'sleep', sleep)
    module.run(); assert waits == [600,600] and seen == [700,700]


@pytest.mark.parametrize('phase', sorted(module.reporting.WAIT_PHASES))
def test_probe_resource_wait_reuses_prepared_plan_without_upstream_restart(prepared, monkeypatch, phase):
    prepared['complete'](); executor = prepared['executors']['geometry']; execute = executor.run; once = [True]; waits = []
    def run(path):
        if once.pop() if once else False: return dict(phase=phase)
        return execute(path)
    executor.run = run
    monkeypatch.setattr(module.time, 'sleep', waits.append)
    module.run(); assert waits == [600]
    assert prepared['calls'].count(('prepare','geometry')) == 1 and prepared['calls'].count(('run','geometry')) == 1


def test_probe_failure_keeps_upstream_and_partial_evidence_without_retry(prepared):
    prepared['complete'](); original = prepared['executors']['geometry'].run
    def fail(path):
        original(path); raise RuntimeError('native geometry failure')
    prepared['executors']['geometry'].run = fail
    with pytest.raises(RuntimeError, match='native geometry failure'): module.run()
    assert (prepared['work'] / 'geometry/complete.json').exists()
    assert (prepared['target'] / 'adaptation_complete.json').exists()
    assert not any(name == 'objectives' for _,name in prepared['calls'])
    with pytest.raises(RuntimeError, match='already invoked'): module.run()


@pytest.mark.parametrize('damage', ['worker','summary','new_fit','test_data'])
def test_native_probe_completion_cannot_overstate_or_change_its_evidence(prepared, damage):
    prepared['complete'](); name = 'objectives'; executor = prepared['executors'][name]
    analyses = [r['analysis'] for r in module.read(prepared['report'] / 'complete.json')['evidence']]
    path = executor.prepare(analyses); executor.run(path)
    target = path.parent / 'complete.json'; record = module.read(target)
    if damage == 'worker': (path.parent / 'A/complete.json').write_text('{}')
    elif damage == 'summary': record['results']['A']['summary']['pairs'] = 31
    elif damage == 'new_fit': record['new_fits'] = 1
    else: record['test_data_read'] = True
    write_json(target,record)
    with pytest.raises(ValueError): module.verify_probe(name,path)
