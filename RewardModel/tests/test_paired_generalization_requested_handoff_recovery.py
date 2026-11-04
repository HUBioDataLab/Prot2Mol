"""Recover the inspected retirement race; simulated processes, real receipt hashes."""
import os
from pathlib import Path
import signal
import pytest

from experiments.paired_generalization import requested_handoff_recovery as module
from experiments.paired_generalization.run import sha256, write_json
from test_paired_generalization_requested_scoped_training import scoped, terminal
from test_paired_generalization_requested_single_seed import prepared

REAL_SIGNAL = module.signal_held_parent


@pytest.fixture
def recovery(scoped,monkeypatch):
    p=scoped;terminal(p);root=p['root'];work=p['work'];old=p['new_target']
    monkeypatch.setattr(module,'ROOT',root);monkeypatch.setattr(module,'WORK',work)
    source=p['source'].with_name('requested_handoff_recovery.py');source.write_text('recovery source fixture')
    monkeypatch.setattr(module,'__file__',str(source))
    for obj,names in ((module.training,('destination','validate','retire_controllers')),
                      (module.pipeline,('reporting_destination','probe_destination','expected_parent')),
                      (module.pipeline.reporting,('destination',)),(module.pipeline.probes,('destination',))):
        for name in names:monkeypatch.setattr(obj,name,getattr(obj,name))
    for role in ('probe_controller','reporting_controller'):
        owner=p['held'][role];p['processes'].pop(owner['pid']);p['states'].pop(owner['pid'])
    proof=dict(identity=p['held']['probe_controller'],signal=int(signal.SIGTERM),
        verified_not_live=True,training_signals_sent=False)
    write_json(old/'probe_controller_retired.json',proof)
    write_json(old/'status.json',dict(phase='stopped_for_inspection',completed_jobs=[],
        error='Live process has no inspectable command',pid=800))
    native=module.training.native_completion(p['plan']['jobs'][0],p['plan']['references']['qualification'])
    intent=dict(plan_sha256=sha256(old/'plan.json'),terminal_launcher=p['states'][703],
        native_completion_sha256=native,training_signals_sent=False,controllers={r:p['held'][r] for r in
        ('probe_controller','reporting_controller','training_controller')})
    write_json(old/'retirement_intent.json',intent)
    activation=dict(controllers={},plans={})
    for i,role in enumerate(('training','reporting','probes')):
        target=old if role=='training' else work/'controlled'/('requested_scoped_'+role+'_20261104')
        if role!='training':
            write_json(target/'plan.json',dict(fixture=role))
            write_json(target/'status.json',dict(phase='stopped_for_inspection'))
        write_json(target/'invocation.json',dict(pid=800+i,plan_sha256=sha256(target/'plan.json')))
        activation['controllers'][role]=dict(pid=800+i,start_ticks=4000+i,uid=os.getuid(),cwd=str(root),argv=['failed',role])
        activation['plans'][role]=dict(path=str(target/'plan.json'),sha256=sha256(target/'plan.json'))
    write_json(module.activation_path(),activation)
    (module.activation_path().parent/'training_controller.log').write_text('fixture traceback after process signal')
    monkeypatch.setattr(module,'INTENT_SHA256',sha256(old/'retirement_intent.json'))
    monkeypatch.setattr(module,'FAILURE_SHA256',sha256(old/'status.json'))
    monkeypatch.setattr(module,'ACTIVATION_SHA256',sha256(module.activation_path()))
    race=[];original_identity=module.cq.process_identity
    def identity(pid):
        if pid==700 and race:
            race.pop();p['states'][700]['state']='Z';p['processes'].pop(700)
            raise RuntimeError('Live process has no inspectable command')
        return original_identity(pid)
    monkeypatch.setattr(module.cq,'process_identity',identity)
    def kill(owner):
        module.training.require_live(owner,'T')
        p['signals'].append((owner['pid'],signal.SIGKILL));p['states'][700]['state']='S';race.append(True)
    monkeypatch.setattr(module,'signal_held_parent',kill)
    sleeps=[];monkeypatch.setattr(module.time,'sleep',lambda seconds:sleeps.append(seconds))
    path=module.prepare()
    return dict(**p,recovery_target=path.parent,recovery_source=source,sleeps=sleeps)


def test_empty_command_race_repolls_same_handle_then_adopts_A_without_refit(recovery):
    p=recovery;old_manifest=sha256(p['new_target']/'plan.json')
    done=module.run('training')
    assert p['signals']==[(700,signal.SIGKILL)]
    assert .05 in p['sleeps']
    assert p['native_calls']==[j['id'] for j in p['plan']['jobs'][1:]]
    assert done['adopted_existing_fits']==1 and done['replacement_fits_launched']==23
    assert not done['allocation_inference'] and not done['additional_seed_fits']
    assert sha256(p['new_target']/'plan.json')==old_manifest
    assert not (p['new_target']/'retirement_complete.json').exists()
    retired=module.read(p['recovery_target']/'reporting_controller_retired.json')
    assert retired['verified_not_live'] and not retired['signal_sent_by_recovery']
    assert (p['recovery_target']/'retirement_complete.json').exists()
    with pytest.raises(RuntimeError,match='already invoked'):module.run('training')


@pytest.mark.parametrize('damage',['native','failed_status','failed_queue_live','old_reporter_live','recovery_source','old_partial_progress'])
def test_recovery_rejects_changed_evidence_before_any_signal_or_fit(recovery,damage):
    p=recovery
    if damage=='native':
        path=Path(p['plan']['jobs'][0]['output_dir'])/'rank_3_complete.json';path.write_text('{}')
    elif damage=='failed_status':(p['new_target']/'status.json').write_text('{}')
    elif damage in ('failed_queue_live','old_reporter_live'):
        owner=(module.read(module.activation_path())['controllers']['reporting'] if damage=='failed_queue_live'
               else p['held']['reporting_controller'])
        p['processes'][owner['pid']]=owner
        p['states'][owner['pid']]=dict(pid=owner['pid'],start_ticks=owner['start_ticks'],uid=owner['uid'],state='S',ppid=1,exit_code=0)
    elif damage=='recovery_source':p['recovery_source'].write_text('changed')
    else:write_json(p['new_target']/'retirement_complete.json',dict(unexpected=True))
    with pytest.raises((ValueError,RuntimeError)):module.run('training')
    assert not p['signals'] and not p['native_calls']


def test_rebound_parent_commands_name_the_actual_recovery_entrypoint(recovery):
    assert module.pipeline.expected_parent('reporting')[-3:]==['experiments.paired_generalization.requested_handoff_recovery','run','training']
    assert module.pipeline.expected_parent('probes')[-1]=='reporting'
    assert module.pipeline.reporting.destination()==module.reporting_destination()
    assert module.pipeline.probes.destination()==module.probe_destination()


def test_observation_timeout_is_not_terminal_or_permission_to_signal(recovery,monkeypatch):
    p=recovery;owner=p['held']['training_controller']
    monkeypatch.setattr(module.cq,'process_identity',lambda _:(_ for _ in ()).throw(OSError('transient /proc read')))
    with pytest.raises(RuntimeError,match='Terminal state remains unverified'):module.wait_terminal(owner,attempts=3)
    assert not p['signals'] and p['sleeps']==[.05,.05,.05]


def test_reused_pid_is_never_signaled_by_terminal_observation(recovery):
    p=recovery;owner=p['held']['training_controller'];p['states'][700]['start_ticks']+=1
    assert module.wait_terminal(owner)['reason']=='pid_replaced'
    assert not p['signals']


def test_python_without_pidfd_revalidates_only_the_held_parent(recovery,monkeypatch):
    p=recovery;owner=p['held']['training_controller']
    monkeypatch.delattr(module.os,'pidfd_open',raising=False)
    assert REAL_SIGNAL(owner)=='revalidated_held_pid'
    assert p['signals']==[(700,signal.SIGKILL)]


def test_fallback_rejects_changed_owner_before_signaling(recovery,monkeypatch):
    p=recovery;owner=p['held']['training_controller']
    monkeypatch.delattr(module.os,'pidfd_open',raising=False)
    p['processes'][700]={**owner,'start_ticks':owner['start_ticks']+1}
    with pytest.raises(ValueError,match='identity changed'):REAL_SIGNAL(owner)
    assert not p['signals']
