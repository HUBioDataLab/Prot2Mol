"""Bounded orchestration with real selection/config generation and fake training."""
import copy
import os
from pathlib import Path

import pytest

from experiments.paired_generalization import confirmation_queue as followup, control_queue, control_driver, control_plan
from experiments.paired_generalization.run import sha256,write_json
from test_paired_generalization_control_driver import completed_tuning_fixture


def complete(job):
    output=Path(job['output_dir'])
    write_json(output/'fixture_evidence.json',dict(finished=True,test_evaluated=False))
    write_json(output/'controlled_training_complete.json',dict(
        config_sha256=job['config_sha256'],plan_sha256=job['plan_sha256'],
        evidence_sha256={'fixture_evidence.json':sha256(output/'fixture_evidence.json')}))


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    root=tmp_path/'project';work=root/'RewardModel/outputs/paired_generalization'
    primary=completed_tuning_fixture(work,monkeypatch)
    directory=work/'controlled';parent=directory/'full_tuning_queue'
    module=root/'RewardModel/experiments/paired_generalization/confirmation_queue.py'
    module.parent.mkdir(parents=True);module.write_text('fixture confirmation source')
    source=root/'RewardModel/reward_model/core.py';source.parent.mkdir(parents=True);source.write_text('fixture training source')
    for mod in (followup,control_queue):
        monkeypatch.setattr(mod,'ROOT',root);monkeypatch.setattr(mod,'WORK',work)
    monkeypatch.setattr(followup,'__file__',str(module))
    monkeypatch.setattr(control_queue,'runtime_guard',lambda:None)
    monkeypatch.setattr(control_queue,'boot_id',lambda:'boot')
    jobs=[followup.read(directory/'jobs'/f"{j['id']}.json") for j in primary['tuning_jobs'][:6]]
    for job in jobs:complete(job)
    process=dict(pid=500,start_ticks=1000,uid=os.getuid(),cwd=str(root),argv=['python','control_queue.py','run'])
    processes={500:process}
    monkeypatch.setattr(control_queue,'process_identity',lambda pid:processes.get(pid))
    parent_plan=dict(boot_id='boot',gpu_uuids=list(control_plan.GPU_UUIDS),predecessor=jobs[1],successors=jobs[2:],
        packages={},files_sha256={str((directory/'plan.json').relative_to(root)):sha256(directory/'plan.json')},
        sources_sha256={str(source.relative_to(root)):sha256(source)},processes={})
    write_json(parent/'plan.json',parent_plan)
    write_json(parent/'invocation.json',dict(pid=500,plan_sha256=sha256(parent/'plan.json')))
    write_json(parent/'launch_process.json',dict(identity=process,plan_sha256=sha256(parent/'plan.json')))
    monkeypatch.setattr(control_driver,'baseline_gate',lambda *args:dict(ready=True,evidence=[]))
    holders=[];monkeypatch.setattr(control_queue,'gpu_processes',lambda:holders)
    monkeypatch.setattr(control_queue,'storage_gate',lambda:dict(ready=True,records=[]))
    path=followup.prepare()
    followup.validate_plan(followup.read(path))
    calls=[]
    def execute(job,*_):
        calls.append(job);complete(job)
        return dict(phase='training_complete_test_not_scored')
    monkeypatch.setattr(control_driver,'execute_job',execute)
    return dict(root=root,work=work,directory=directory,parent=parent,jobs=jobs,
        process=process,processes=processes,source=source,holders=holders,calls=calls,
        plan=followup.read(path),target=path.parent)


def ready(value):
    value['processes'].clear()
    parent=value['parent'];plan=followup.read(parent/'plan.json')
    write_json(parent/'complete.json',dict(phase='four_tuning_successors_complete_other_work_pending',
        plan_sha256=sha256(parent/'plan.json'),
        evidence=control_queue.require_completed([plan['predecessor'],*plan['successors']])))


def test_exact_live_parent_is_waited_before_selection_even_with_completion_receipt(prepared,monkeypatch):
    ready(prepared);prepared['processes'][500]=prepared['process']
    def sleep(seconds):
        assert seconds==30 and prepared['calls']==[]
        assert not (prepared['directory']/'selections/full.json').exists()
        assert followup.read(prepared['target']/'status.json')['phase']=='waiting_for_tuning_coordinator'
        prepared['processes'].clear()
    monkeypatch.setattr(followup.time,'sleep',sleep)
    result=followup.run()
    assert [(j['seed'],j['regime']) for j in prepared['calls']]==list(followup.ORDER)
    # A's best LR differs; actual selector must still choose B's LR=1.
    assert all(j['arm']=='full' and j['learning_rate_multiplier']==1. for j in prepared['calls'])
    assert len(result['evidence'])==4 and not result['study_complete']
    assert result['phase']=='full_confirmation_training_complete_other_work_pending'
    with pytest.raises(RuntimeError,match='already invoked'):followup.run()


@pytest.mark.parametrize('damage',['no_receipt','wrong_receipt','missing_fit','changed_source','changed_parent_handle'])
def test_incomplete_or_changed_upstream_never_selects_or_trains(prepared,damage):
    ready(prepared)
    if damage=='no_receipt':(prepared['parent']/'complete.json').unlink()
    elif damage=='wrong_receipt':
        p=prepared['parent']/'complete.json';r=followup.read(p);r['evidence'].pop();write_json(p,r)
    elif damage=='missing_fit':(Path(prepared['jobs'][2]['output_dir'])/'controlled_training_complete.json').unlink()
    elif damage=='changed_source':prepared['source'].write_text('different training')
    else:
        prepared['processes'][500]={**prepared['process'],'start_ticks':1001}
        (prepared['parent']/'complete.json').unlink()
    with pytest.raises((ValueError,RuntimeError,FileNotFoundError)):followup.run()
    assert not prepared['calls'] and not (prepared['directory']/'selections/full.json').exists()


def test_prepare_requires_matching_live_parent_and_no_existing_confirmation(prepared):
    prepared['processes'][500]={**prepared['process'],'uid':os.getuid()+1}
    with pytest.raises(ValueError,match='exact live'):followup.prepare()
    prepared['processes'][500]=prepared['process']
    existing=prepared['directory']/'runs'/control_plan.job_id('full','A',43,1.)
    existing.mkdir()
    with pytest.raises(RuntimeError,match='already exists'):followup.prepare()


@pytest.mark.parametrize('gate',['gpu','storage','lock'])
def test_waiting_gates_release_global_lock_and_do_not_train(prepared,monkeypatch,gate):
    import fcntl
    ready(prepared)
    handle=(prepared['directory']/'driver.lock').open('a')
    if gate=='gpu':prepared['holders'].append(dict(gpu_uuid=control_plan.GPU_UUIDS[0],pid=9))
    elif gate=='storage':monkeypatch.setattr(control_queue,'storage_gate',lambda:dict(ready=False,records=[]))
    else:fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
    waits=[]
    def sleep(seconds):
        waits.append(seconds);assert not prepared['calls']
        fcntl.flock(handle,fcntl.LOCK_UN)
        fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB);fcntl.flock(handle,fcntl.LOCK_UN)
        prepared['holders'].clear()
        monkeypatch.setattr(control_queue,'storage_gate',lambda:dict(ready=True,records=[]))
    monkeypatch.setattr(followup.time,'sleep',sleep)
    try:followup.run()
    finally:handle.close()
    assert waits==[30] and len(prepared['calls'])==4


def test_execution_owns_global_lock_and_failed_training_is_never_retried(prepared,monkeypatch):
    import fcntl
    ready(prepared)
    def fail(job,*_):
        with (prepared['directory']/'driver.lock').open('a') as lock:
            with pytest.raises(BlockingIOError):fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepared['calls'].append(job);Path(job['output_dir']).mkdir()
        raise RuntimeError('training failed')
    monkeypatch.setattr(control_driver,'execute_job',fail)
    with pytest.raises(RuntimeError,match='training failed'):followup.run()
    assert len(prepared['calls'])==1
    assert followup.read(prepared['target']/'status.json')['phase']=='stopped_for_inspection'
    with pytest.raises(RuntimeError,match='already invoked'):followup.run()


def test_success_without_verified_receipt_stops(prepared,monkeypatch):
    ready(prepared)
    monkeypatch.setattr(control_driver,'execute_job',lambda *args:dict(phase='training_complete_test_not_scored'))
    with pytest.raises(RuntimeError,match='without verified completion'):followup.run()
    assert not (prepared['target']/'complete.json').exists()


@pytest.mark.parametrize('damage',['selection_snapshot','confirmation_config','training_source'])
def test_changes_during_fit_stop_before_next_confirmation(prepared,monkeypatch,damage):
    ready(prepared);execute=control_driver.execute_job
    def changed(job,*args):
        result=execute(job,*args)
        if damage=='selection_snapshot':write_json(prepared['target']/'selection.json',dict(changed=True))
        elif damage=='confirmation_config':Path(job['config_path']).write_text('changed')
        else:prepared['source'].write_text('changed')
        return result
    monkeypatch.setattr(control_driver,'execute_job',changed)
    with pytest.raises(ValueError,match='changed'):followup.run()
    assert len(prepared['calls'])==1 and not (prepared['target']/'complete.json').exists()


def test_last_gpu_recheck_waits_only_before_a_fit_starts(prepared,monkeypatch):
    ready(prepared);execute=control_driver.execute_job;once=[True];waits=[]
    def recheck(job,*args):
        if once:
            once.pop();assert not Path(job['output_dir']).exists()
            return dict(phase='waiting_for_authorized_gpus')
        return execute(job,*args)
    monkeypatch.setattr(control_driver,'execute_job',recheck)
    monkeypatch.setattr(followup.time,'sleep',lambda seconds:waits.append(seconds))
    followup.run()
    assert waits==[30] and len(prepared['calls'])==4
