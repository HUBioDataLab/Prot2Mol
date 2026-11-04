"""Adoption/retirement lifecycle tests use simulated processes and native receipts."""
import copy
import os
from pathlib import Path
import signal

import pytest

from experiments.paired_generalization import requested_scoped_training as module
from experiments.paired_generalization.run import sha256, write_json
from test_paired_generalization_requested_single_seed import prepared


def native(job, qualification):
    out = Path(job['output_dir'])
    write_json(out/'controlled_run.json', dict(format_version=1, allow_checkpoint_retention=True, **job))
    write_json(out/'experiment_summary.json', dict(config_sha256=job['config_sha256'], test_evaluated=False, test_metrics={}))
    write_json(out/'trainer_state.json', dict(epoch=15, global_step=150, log_history=[
        dict(epoch=e, step=e*10, eval_macro_spearman=e/100, eval_loss=20-e) for e in range(1,16)]))
    write_json(out/'launch_provenance.json', dict(config_sha256=job['config_sha256']))
    write_json(out/'runtime_qualification.json', qualification)
    for r in range(4):
        write_json(out/f'rank_{r}_complete.json', dict(rank=r, world_size=4, config_sha256=job['config_sha256']))


@pytest.fixture
def scoped(prepared, monkeypatch):
    p=prepared; root=p['root']; work=p['work']; scope=p['plan']
    monkeypatch.setattr(module, 'ROOT', root); monkeypatch.setattr(module, 'WORK', work)
    for obj, name in ((module,'requested_scoped_training.py'), (module.retention,'requested_retention.py')):
        path=p['source'].with_name(name); path.write_text('replacement source fixture')
        monkeypatch.setattr(obj, '__file__', str(path))
    held={}; states={}
    for i,name in enumerate(('training_controller','reporting_controller','probe_controller','training_launcher')):
        pid=700+i
        owner=dict(pid=pid,start_ticks=3000+i,uid=os.getuid(),cwd=str(root),argv=['fixture',name])
        held[name]=owner; p['processes'][pid]=owner
        states[pid]=dict(pid=pid,start_ticks=3000+i,uid=os.getuid(),ppid=700 if i==3 else 1,
            state='T' if i==0 else 'S',exit_code=0)
    held.update(current_job=scope['jobs'][0]['id'],training_signal_sent=False,training_ranks={})
    write_json(module.hold_path(), held); monkeypatch.setattr(module,'HOLD_SHA256',sha256(module.hold_path()))
    write_json(p['target']/'invocation.json',dict(pid=700,plan_sha256=sha256(p['target']/'plan.json')))
    for name in ('requested_reporting_queue','requested_probe_queue'):
        target=work/'controlled'/name
        write_json(target/'plan.json',dict(fixture=name)); write_json(target/'invocation.json',dict(fixture=True))
    native(scope['jobs'][0],scope['references']['qualification'])
    monkeypatch.setattr(module,'process_state',lambda pid:states.get(pid))
    signals=[]
    def kill(pid,sig):
        assert pid in (700,701,702) and states[703]['state']=='Z'
        signals.append((pid,sig)); p['processes'].pop(pid); states[pid]['state']='Z'
    monkeypatch.setattr(module.os,'kill',kill)
    def retain(run,prune=False):
        assert prune
        rows=module.read(run/'trainer_state.json')['log_history']
        result=dict(selections=module.retention.selection_rules(rows), additional_realizations=None)
        write_json(run/'retained/retention.json',result); return result
    monkeypatch.setattr(module.retention,'retain',retain)
    monkeypatch.setattr(module.driver,'execute_job',lambda *_:pytest.fail('Legacy executor is forbidden'))
    monkeypatch.setattr(module.driver,'qualify_runtime',lambda *_:pytest.fail('Reuse existing qualification'))
    monkeypatch.setattr(module.driver,'verify_training_data',lambda *_:None)
    calls=[]
    def execute(command,log,phase,identity):
        import fcntl
        assert phase=='training' and '--nproc_per_node=4' in command
        assert command[command.index('--config')+1].endswith(identity+'.yaml')
        with (p['directory']/'driver.lock').open('a') as lock:
            with pytest.raises(BlockingIOError): fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        calls.append(identity)
        job=next(j for j in scope['jobs'] if j['id']==identity)
        native(job,scope['references']['qualification'])
    monkeypatch.setattr(module.driver,'_run_process',execute)
    path=module.prepare(); module.validate(module.read(path))
    return dict(**p,states=states,signals=signals,native_calls=calls,new_target=path.parent,held=held)


def terminal(p):
    p['states'][703]['state']='Z'; p['processes'].pop(703,None)


def test_adopts_first_fit_once_then_23_fixed_fits_and_explicit_scope_bridge(scoped,monkeypatch):
    p=scoped; waits=[]
    def sleep(seconds):
        waits.append(seconds); assert not p['native_calls'] and not p['signals']; terminal(p)
    monkeypatch.setattr(module.time,'sleep',sleep)
    original_sha=sha256(p['target']/'plan.json')
    done=module.run()
    assert waits==[600]
    assert p['native_calls']==[j['id'] for j in p['plan']['jobs'][1:]]
    assert p['signals']==[(702,signal.SIGTERM),(701,signal.SIGTERM),(700,signal.SIGKILL)]
    assert done==module.read(p['target']/'complete.json')
    assert done['training_fits_created']==24 and done['adopted_existing_fits']==1
    assert done['replacement_fits_launched']==23 and not done['allocation_inference']
    assert sha256(p['target']/'plan.json')==original_sha
    assert module.read(Path(p['plan']['jobs'][0]['output_dir'])/'controlled_training_complete.json')['adopted_existing_fit']
    with pytest.raises(RuntimeError,match='already invoked'):module.run()


@pytest.mark.parametrize('damage',['missing_launcher','reused_launcher','failed_launcher','parent_unheld','rank_live','missing_rank','incomplete_horizon','changed_config'])
def test_ambiguous_or_failed_current_fit_never_signals_or_refits(scoped,damage):
    p=scoped; terminal(p); output=Path(p['plan']['jobs'][0]['output_dir'])
    if damage=='missing_launcher':p['states'].pop(703)
    elif damage=='reused_launcher':p['states'][703]['start_ticks']+=1
    elif damage=='failed_launcher':p['states'][703]['exit_code']=256
    elif damage=='parent_unheld':p['states'][700]['state']='S'
    elif damage=='rank_live':
        plan=module.read(p['new_target']/'plan.json'); owner=p['held']['reporting_controller']
        plan['held']['training_ranks']={'701':owner}; write_json(module.hold_path(),plan['held'])
        monkey_hash=sha256(module.hold_path())
        # Call the terminal guard directly; changing an immutable plan also fails validation.
        with pytest.raises(RuntimeError,match='rank is still live'):module.current_terminal(plan)
        return
    elif damage=='missing_rank':(output/'rank_3_complete.json').unlink()
    elif damage=='incomplete_horizon':
        state=module.read(output/'trainer_state.json');state['epoch']=14;write_json(output/'trainer_state.json',state)
    else:Path(p['plan']['jobs'][0]['config_path']).write_text('changed')
    with pytest.raises((RuntimeError,ValueError,FileNotFoundError)):module.run()
    assert not p['signals'] and not p['native_calls']


def test_partial_successor_is_preserved_and_not_relaunched(scoped):
    p=scoped; terminal(p); output=Path(p['plan']['jobs'][1]['output_dir']);output.mkdir()
    (output/'partial.txt').write_text('preserve')
    with pytest.raises(RuntimeError,match='already has output'):module.run()
    assert not p['native_calls'] and (output/'partial.txt').read_text()=='preserve'
    assert not (p['target']/'complete.json').exists()


def test_changed_retention_source_prevents_signals_and_launch(scoped):
    Path(module.retention.__file__).write_text('changed')
    with pytest.raises(ValueError,match='source/input changed'):module.run()
    assert not scoped['signals'] and not scoped['native_calls']
