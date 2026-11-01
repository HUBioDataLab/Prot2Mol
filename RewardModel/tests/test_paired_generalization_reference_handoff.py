import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.paired_generalization import reference_handoff as handoff
from experiments.paired_generalization.run import sha256, write_json


def fake_process(proc, pid=123, state='S', start=456):
    directory = proc/str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    fields = [state]+['0']*18+[str(start)]
    (directory/'stat').write_text(f'{pid} (python worker) '+ ' '.join(fields))
    (directory/'cmdline').write_bytes(b'python\0-u\0control_driver.py\0')
    if not (directory/'cwd').exists():
        (directory/'cwd').symlink_to(proc)
    return directory


def test_process_identity_distinguishes_live_missing_zombie_and_reused_pid(tmp_path):
    fake_process(tmp_path)
    original = handoff.process_identity(123, tmp_path)
    inspect = lambda pid: handoff.process_identity(pid, tmp_path)
    assert handoff.live_originals({'driver': original}, inspect) == {'driver': original}
    assert handoff.process_identity(999, tmp_path) is None
    fake_process(tmp_path, start=999)
    assert handoff.live_originals({'driver': original}, inspect) == {}
    fake_process(tmp_path, state='Z')
    assert handoff.process_identity(123, tmp_path) is None
    (tmp_path/'123/stat').write_text('malformed')
    with pytest.raises(IndexError):
        handoff.process_identity(123, tmp_path)  # Not silently treated as exited.


def test_owned_process_validation_rejects_wrong_project_or_training_config(tmp_path, monkeypatch):
    monkeypatch.setattr(handoff, 'ROOT', tmp_path)
    config = tmp_path/'reference.yaml'
    identities = {role: dict(pid=i, uid=os.getuid(), cwd=str(tmp_path), argv=['python', script])
                  for i, (role, script) in enumerate([('driver','control_driver.py'),
                    ('training','run.py'),('protector','protect_trajectory.py')], 1)}
    identities['training']['argv'] += ['torch.distributed.run','--config',str(config)]
    handoff.validate_processes(identities, config)
    with pytest.raises(ValueError, match='configuration'):
        handoff.validate_processes(identities, tmp_path/'other.yaml')
    identities['protector']['cwd'] = str(tmp_path.parent)
    with pytest.raises(ValueError, match='authorized project'):
        handoff.validate_processes(identities, config)


@pytest.mark.parametrize('damage', ['source','config','spec','host','job'])
def test_pinned_plan_refuses_changes(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(handoff, 'ROOT', tmp_path)
    monkeypatch.setattr(handoff, 'WORK', tmp_path/'work')
    monkeypatch.setattr(handoff, 'runtime_guard', lambda: None)
    monkeypatch.setattr(handoff, 'boot_id', lambda: 'host-boot')
    source = tmp_path/'source.py'; source.write_text('original')
    monkeypatch.setattr(handoff, 'software', lambda: {'source': sha256(source)})
    directory = handoff.WORK/'controlled'
    write_json(directory/'plan.json', {'grid':'original'})
    spec = tmp_path/'Paired input ranking generalization.md';spec.write_text('original spec')
    config = tmp_path/'config.yaml';config.write_text('original config')
    job = dict(id=handoff.REFERENCE, config_path=str(config), config_sha256=sha256(config),
               plan_sha256=sha256(directory/'plan.json'))
    job_path = directory/'jobs'/f'{handoff.REFERENCE}.json';write_json(job_path,job)
    plan = dict(reference=job, boot_id='host-boot', gpu_uuids=list(handoff.GPU_UUIDS),
        software=handoff.software(), spec_sha256=sha256(spec), job_metadata_sha256=sha256(job_path))
    handoff.validate_plan(plan)
    if damage == 'host':monkeypatch.setattr(handoff,'boot_id',lambda:'rebooted')
    else:{'source':source,'config':config,'spec':spec,'job':job_path}[damage].write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        handoff.validate_plan(plan)


def test_child_environment_restores_gpu_execution_without_wandb_id_reuse(monkeypatch):
    monkeypatch.setenv('ACCELERATE_USE_CPU','true')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setenv('WANDB_RUN_ID','unrelated-run')
    monkeypatch.setenv('WANDB_RESUME','must')
    for training in (True,False):
        env = handoff.child_environment(training)
        assert env['ACCELERATE_USE_CPU']=='false'
        assert env['CUDA_VISIBLE_DEVICES'].split(',')==list(handoff.GPU_UUIDS)
        assert 'WANDB_RUN_ID' not in env and 'WANDB_RESUME' not in env
        assert env['WANDB_MODE']==('online' if training else 'disabled')


def test_missing_rank_blocks_recovery_and_leaves_payloads_intact(tmp_path, monkeypatch):
    from test_paired_generalization_retention import make_run
    monkeypatch.setattr(handoff, 'WORK', tmp_path/'work')
    run = handoff.WORK/'controlled/runs'/handoff.REFERENCE
    make_run(run)
    launch = handoff.read(run/'launch_provenance.json')
    launch['world_size']=4;launch['config']['training']['num_train_epochs']=15
    write_json(run/'launch_provenance.json',launch)
    summary = handoff.read(run/'experiment_summary.json')
    summary.update(test_evaluated=False,test_metrics={});write_json(run/'experiment_summary.json',summary)
    for rank in range(4):write_json(run/f'rank_{rank}_complete.json',dict(rank=rank,world_size=4,config_sha256='fixture-config'))
    (run/'rank_2_complete.json').unlink()
    monkeypatch.setattr(handoff.protect_trajectory,'verify_complete',lambda path:dict(epochs=list(range(15)),config_sha256='fixture-config'))
    job = dict(id=handoff.REFERENCE,output_dir=str(run),config_sha256='fixture-config',plan_sha256='plan')
    with pytest.raises(FileNotFoundError):handoff.finish_reference(job)
    assert not (run/'controlled_training_complete.json').exists()
    assert len(list(run.glob('checkpoint-*/pytorch_model.bin')))==6
    assert len(list(run.glob('checkpoint-*/optimizer.pt')))==6


def audit_fixture(output, plan):
    write_json(output/'plan.json',plan)
    records=[]
    for epoch in range(1,16):
        for prefix in ('eval','train_eval'):
            directory=output/f'epoch{epoch:02d}'/prefix
            info=dict(epoch=epoch,prefix=prefix)
            write_json(directory/'complete.json',dict(analysis=info,cpu_qualification=False,parameters_unchanged=True))
            records.append(dict(path=str(directory),sha256=sha256(directory/'complete.json'),analysis=info))
    write_json(output/'complete.json',dict(exports=30,test_data_used=False,trajectory=records,plan_sha256=sha256(output/'plan.json')))


def test_audit_receipt_requires_every_epoch_and_partition(tmp_path):
    plan={'fixture':True};audit_fixture(tmp_path,plan)
    assert handoff.verify_reference_audit(plan,tmp_path)==sha256(tmp_path/'complete.json')
    result=handoff.read(tmp_path/'complete.json')
    result['trajectory'][-1]=copy.deepcopy(result['trajectory'][0])
    write_json(tmp_path/'complete.json',result)
    with pytest.raises(ValueError,match='omits or duplicates'):handoff.verify_reference_audit(plan,tmp_path)


@pytest.mark.parametrize('failure',[None,'audit_exit','partial_audit','next_driver_exit'])
def test_handoff_orders_live_wait_completion_audit_and_one_next_job(tmp_path,monkeypatch,failure):
    monkeypatch.setattr(handoff,'ROOT',tmp_path)
    monkeypatch.setattr(handoff,'WORK',tmp_path/'work')
    destination=handoff.WORK/'controlled/reference_handoff'
    write_json(destination/'plan.json',dict(reference={'id':handoff.REFERENCE},processes={'driver':{}}))
    write_json(destination.parent/'plan.json',{'frozen':'grid'})
    events=[]
    monkeypatch.setattr(handoff,'validate_plan',lambda plan:events.append('validate'))
    living=iter([{'driver':{'pid':1}},{}])
    monkeypatch.setattr(handoff,'live_originals',lambda processes:next(living))
    monkeypatch.setattr(handoff.time,'sleep',lambda seconds:events.append(('sleep',seconds)))
    monkeypatch.setattr(handoff,'finish_reference',lambda job:events.append('finish'))
    audit_plan={'software':{'fixture':'software'}}
    monkeypatch.setattr(handoff.reference_contrastive,'build_plan',lambda directory:audit_plan)
    output=handoff.WORK/'tier1/reference_contrastive'/handoff.REFERENCE/handoff.digest_json(audit_plan['software'])[:16]
    if failure=='partial_audit':output.mkdir(parents=True)
    monkeypatch.setattr(handoff.subprocess,'check_output',lambda *args,**kwargs:'')
    job=dict(id='next_frozen_job',config_path=str(tmp_path/'next.yaml'),output_dir=str(tmp_path/'next_run'))
    monkeypatch.setattr(handoff.control_driver,'next_job',lambda directory,plan:job)
    def start(directory,name,command,*,training):
        events.append(name)
        if name=='reference_audit':
            assert not training and command[-3:]==['run','--workers','4']
            def wait():
                if failure=='audit_exit':return 1
                audit_fixture(output,audit_plan);return 0
            return SimpleNamespace(pid=2,wait=wait)
        assert training
        if failure=='next_driver_exit':return SimpleNamespace(pid=3,poll=lambda:1,returncode=1)
        write_json(destination.parent/'driver_status.json',dict(driver_pid=3,pid=4,job_id=job['id'],phase='training'))
        return SimpleNamespace(pid=3,poll=lambda:None)
    monkeypatch.setattr(handoff,'start_child',start)
    monkeypatch.setattr(handoff,'process_identity',lambda pid:dict(pid=pid,argv=['python',job['config_path']]))
    if failure:
        with pytest.raises(RuntimeError):handoff.run()
        assert handoff.read(destination/'status.json')['phase']=='failed_requires_inspection'
        assert not (destination/'complete.json').exists()
        if failure!='next_driver_exit':assert 'next_driver' not in events
    else:
        result=handoff.run()
        assert result['next_job']==job['id'] and not result['study_complete'] and not result['next_training_complete']
        assert events.index(('sleep',30))<events.index('finish')<events.index('reference_audit')<events.index('next_driver')
        assert events.count('reference_audit')==events.count('next_driver')==1
    with pytest.raises(RuntimeError,match='already invoked'):handoff.run()


def test_cpu_coordinator_is_restricted_to_atum(monkeypatch,tmp_path):
    monkeypatch.setattr(handoff,'ROOT',tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    with pytest.raises(RuntimeError,match='Atum'):handoff.runtime_guard()
