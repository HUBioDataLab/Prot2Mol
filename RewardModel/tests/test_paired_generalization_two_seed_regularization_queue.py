"""Real config/selection handoff with simulated training and native evaluation.

This tests orchestration, not scientific GPU results or metric correctness.
"""
import os
from pathlib import Path

import pytest

from experiments.paired_generalization import two_seed_regularization_queue as module
from experiments.paired_generalization.run import sha256,write_json
from test_paired_generalization_two_seed import prepared as evaluation_prepared, legacy_prepared
from test_paired_generalization_confirmation_queue import prepared as confirmation_prepared,complete as complete_training


def finish_training(job):
    output=Path(job['output_dir']);lr=job['learning_rate_multiplier']
    # A prefers .3, B prefers 1. Only the latter may choose confirmations.
    score=({1.:.1,.3:.9,3.:.2} if job['regime']=='A' else {1.:.3,.3:.2,3.:.1})[lr]
    write_json(output/'experiment_summary.json',dict(config_sha256=job['config_sha256'],test_evaluated=False,test_metrics={}))
    for rank in range(4):write_json(output/f'rank_{rank}_complete.json',dict(rank=rank,world_size=4,config_sha256=job['config_sha256']))
    write_json(output/'retained/retention.json',dict(config_sha256=job['config_sha256'],selections={'best_macro':dict(step=10,metric=score)}))
    path=output/'prediction_exports/trajectory/eval_step00000010.json';path.parent.mkdir(parents=True)
    path.with_suffix('.npz').write_bytes(b'synthetic validation payload')
    write_json(path,dict(predictions_sha256=sha256(path.with_suffix('.npz')),metrics=dict(eval_macro_spearman=score,eval_spearman_defined_fraction=1.)))
    complete_training(job)


@pytest.fixture
def prepared(evaluation_prepared,monkeypatch):
    value=evaluation_prepared;root,work=value['root'],value['work'];directory=work/'controlled'
    # Finish the simulated parent's native reporting path, then leave its exact
    # process identity visible to test that a receipt alone cannot trigger work.
    module.full_evaluation_queue.run()
    parent=work/'controlled/full_evaluation_two_seed_queue'
    identity=dict(pid=700,start_ticks=3000,uid=os.getuid(),cwd=str(root),argv=['python','two_seed_full_evaluation_queue.py','run'])
    value['processes'][700]=identity
    write_json(parent/'invocation.json',dict(pid=700,plan_sha256=sha256(parent/'plan.json')))
    write_json(parent/'launch_process.json',dict(identity=identity,plan_sha256=sha256(parent/'plan.json')))
    monkeypatch.setattr(module,'ROOT',root);monkeypatch.setattr(module,'WORK',work)
    source=root/'RewardModel/experiments/paired_generalization/two_seed_regularization_queue.py';source.write_text('fixture regularization queue')
    monkeypatch.setattr(module,'__file__',str(source))
    monkeypatch.setattr(module.legacy_queue,'ROOT',root);monkeypatch.setattr(module.legacy_queue,'WORK',work)
    legacy_source=root/'RewardModel/experiments/paired_generalization/regularization_queue.py'
    legacy_source.write_text('fixture historical regularization queue')
    monkeypatch.setattr(module.legacy_queue,'__file__',str(legacy_source))
    monkeypatch.setattr(module.family_analysis,'software',lambda:dict(source_sha256={},packages={}))
    def family_selection():
        path=work/'regularization_analysis_two_seed/selection.json'
        write_json(path,dict(fixture=True));return path
    def family_report(paths):
        assert len(paths)==3
        assert {module.read(p)['arm'] for p in paths}=={'full',*module.ARMS}
        path=work/'regularization_analysis_two_seed/fixture/complete.json'
        write_json(path,dict(fixture=True));return path
    monkeypatch.setattr(module.family_analysis,'prepare_selection',family_selection)
    monkeypatch.setattr(module.family_analysis,'analyze',family_report)
    primary=module.read(directory/'plan.json')
    for job in primary['tuning_jobs']:
        if job['arm'] in module.ARMS:module.control_plan.materialize_job(job,directory,sha256(directory/'plan.json'))
    path=module.prepare();module.validate_plan(module.read(path))
    value['calls'].clear();value['evaluated'].clear();value['analyzed'].clear()
    def execute(job,*args):
        import fcntl
        with (directory/'driver.lock').open('a') as lock:
            with pytest.raises(BlockingIOError):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        value['calls'].append(job);finish_training(job)
        return dict(phase='training_complete_test_not_scored')
    monkeypatch.setattr(module.control_driver,'execute_job',execute)
    def summarize(arm,paths):
        assert arm in ('full',*module.ARMS) and len(paths)==8
        assert len({module.read(p)['job_id'] for p in paths})==8
        result=work/'controlled_summaries_two_seed'/arm/'fixture/complete.json'
        write_json(result,dict(arm=arm,fixture=True));return result
    monkeypatch.setattr(module.controlled_summary,'summarize',summarize)
    value.update(regularization_target=path.parent,regularization_parent=parent,
        regularization_identity=identity,regularization_plan=module.read(path))
    return value


def ready(value):value['processes'].clear()


def test_live_parent_wait_then_exact_sixteen_registered_fits_and_reports(prepared,monkeypatch):
    waits=[]
    def sleep(seconds):
        waits.append(seconds);assert not prepared['calls'] and not prepared['evaluated']
        assert module.read(prepared['regularization_target']/'status.json')['phase']=='waiting_for_full_reference_report'
        ready(prepared)
    monkeypatch.setattr(module.time,'sleep',sleep)
    result=module.run()
    assert waits==[30] and len(prepared['calls'])==len(prepared['evaluated'])==len(prepared['analyzed'])==16
    for arm in module.ARMS:
        jobs=[j for j in prepared['calls'] if j['arm']==arm]
        assert len(jobs)==8 and len([j for j in jobs if j['seed']==42])==6
        assert {(j['seed'],j['regime']) for j in jobs if j['seed']!=42}=={(43,'A'),(43,'B')}
        assert all(j['learning_rate_multiplier']==1. for j in jobs if j['seed']!=42)
        assert {j['id'] for j in jobs}==set(module.controlled_summary.expected_jobs(arm,1.))
    assert [r['arm'] for r in result['arms']]==list(module.ARMS)
    assert result['training_fits_created']==16 and result['additional_settings_created']==0
    assert result['phase']=='regularization_two_seed_reports_complete_other_work_pending'
    assert sha256(Path(result['family_report']))==result['family_report_sha256']
    assert not list((prepared['directory']/'jobs').glob('weight_decay*__seed44__*.json'))
    assert not result['test_used_for_selection'] and not result['study_complete']
    with pytest.raises(RuntimeError,match='already invoked'):module.run()


@pytest.mark.parametrize('damage',['missing_parent','wrong_parent','missing_parent_fit','parent_analysis','source','pid_reuse'])
def test_incomplete_or_changed_reference_never_starts_training(prepared,damage):
    ready(prepared);parent=prepared['regularization_parent']
    if damage in ('missing_parent','pid_reuse'):(parent/'complete.json').unlink()
    if damage=='pid_reuse':prepared['processes'][700]={**prepared['regularization_identity'],'start_ticks':3001}
    elif damage=='wrong_parent':
        result=module.read(parent/'complete.json');result['evidence'].pop();write_json(parent/'complete.json',result)
    elif damage=='missing_parent_fit':
        (Path(prepared['jobs'][0]['output_dir'])/'controlled_training_complete.json').unlink()
    elif damage=='parent_analysis':Path(module.read(parent/'complete.json')['evidence'][0]['analysis']).write_text('{}')
    elif damage=='source':prepared['source'].write_text('different source')
    with pytest.raises((ValueError,RuntimeError,FileNotFoundError)):module.run()
    assert not prepared['calls'] and not prepared['evaluated']


def test_prepare_requires_live_owned_parent(prepared):
    prepared['processes'][700]={**prepared['regularization_identity'],'uid':os.getuid()+1}
    with pytest.raises(ValueError,match='exact live owned'):module.prepare()


@pytest.mark.parametrize('existing',['training','confirmation_registration','selection','evaluation','analysis','summary'])
def test_existing_work_is_not_adopted_or_retried(prepared,existing):
    work=prepared['work'];arm=next(iter(module.ARMS));identity=module.control_plan.job_id(arm,'A',42,1.)
    paths=dict(training=work/'controlled/runs'/identity,
        confirmation_registration=work/'controlled/jobs'/f'{module.control_plan.job_id(arm,"A",43,1.)}.json',
        selection=work/'controlled/selections'/f'{arm}.json',evaluation=work/'postselection'/identity,
        analysis=work/'postselection_analysis'/identity,summary=work/'controlled_summaries_two_seed'/arm)
    path=paths[existing];path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix=='.json':path.write_text('{}')
    else:path.mkdir()
    with pytest.raises(RuntimeError,match='already exists'):module.prepare()
    ready(prepared)
    with pytest.raises(RuntimeError,match='already exists'):module.run()
    assert not prepared['calls']


@pytest.mark.parametrize('gate',['gpu','quota','lock','last_gpu_check'])
def test_training_waits_without_creating_or_duplicating_work(prepared,monkeypatch,gate):
    import fcntl
    ready(prepared);handle=(prepared['directory']/'driver.lock').open('a')
    execute=module.control_driver.execute_job;once=[True]
    if gate=='gpu':prepared['holders'].append(dict(gpu_uuid=module.control_plan.GPU_UUIDS[0],pid=9))
    elif gate=='quota':monkeypatch.setattr(module.control_queue,'storage_gate',lambda:dict(ready=False))
    elif gate=='lock':fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    else:
        def busy(*args):
            if once:once.pop();return dict(phase='waiting_for_authorized_gpus')
            return execute(*args)
        monkeypatch.setattr(module.control_driver,'execute_job',busy)
    waits=[]
    def sleep(seconds):
        waits.append(seconds);assert not prepared['calls']
        prepared['holders'].clear();fcntl.flock(handle,fcntl.LOCK_UN)
        monkeypatch.setattr(module.control_queue,'storage_gate',lambda:dict(ready=True))
    monkeypatch.setattr(module.time,'sleep',sleep)
    try:module.run()
    finally:handle.close()
    assert waits==[30] and len(prepared['calls'])==16


@pytest.mark.parametrize('failure',['training','missing_training_receipt','evaluation','evaluation_partial_lock'])
def test_failures_stop_without_relaunch_and_no_final_completion(prepared,monkeypatch,failure):
    ready(prepared)
    if failure.startswith('evaluation'):
        def fail(identity,workers):
            prepared['evaluated'].append(identity)
            (prepared['work']/'postselection'/identity).mkdir(parents=True)
            if failure=='evaluation_partial_lock':raise BlockingIOError('after evaluation start')
            raise RuntimeError('native evaluation failed')
        monkeypatch.setattr(module.postselection,'run_evaluation',fail)
    else:
        def fail(job,*args):
            prepared['calls'].append(job);Path(job['output_dir']).mkdir()
            if failure=='training':raise RuntimeError('training failed')
            return dict(phase='training_complete_test_not_scored')
        monkeypatch.setattr(module.control_driver,'execute_job',fail)
    with pytest.raises((ValueError,RuntimeError,FileNotFoundError)):module.run()
    assert not (prepared['regularization_target']/'complete.json').exists()
    assert module.read(prepared['regularization_target']/'status.json')['phase']=='stopped_for_inspection'
    count=len(prepared['calls'])
    with pytest.raises(RuntimeError,match='already invoked'):module.run()
    assert len(prepared['calls'])==count


def test_reference_change_after_training_prevents_validation_selection(prepared,monkeypatch):
    ready(prepared);execute=module.control_driver.execute_job
    def change(*args):
        result=execute(*args);(prepared['regularization_parent']/'complete.json').write_text('{}');return result
    monkeypatch.setattr(module.control_driver,'execute_job',change)
    with pytest.raises(ValueError,match='handoff changed'):module.run()
    assert len(prepared['calls'])==1 and not prepared['evaluated']
    assert not (prepared['directory']/'selections'/f'{next(iter(module.ARMS))}.json').exists()


def test_earlier_arm_evidence_is_rechecked_after_second_arm_finishes(prepared,monkeypatch):
    ready(prepared);summarize=module.controlled_summary.summarize
    def change(arm,paths):
        result=summarize(arm,paths)
        if arm==list(module.ARMS)[-1]:
            first=module.read(prepared['regularization_target']/'arms'/f'{next(iter(module.ARMS))}.json')
            Path(first['evaluations'][0]['analysis']).write_text('{}')
        return result
    monkeypatch.setattr(module.controlled_summary,'summarize',change)
    with pytest.raises(ValueError,match='Earlier evaluation evidence changed'):module.run()
    assert len(prepared['calls'])==16 and not (prepared['regularization_target']/'complete.json').exists()


@pytest.mark.parametrize('operation',[module.train,module.evaluate])
def test_seed44_cannot_enter_either_scientific_worker(operation):
    with pytest.raises(ValueError,match='seed42/43'):
        operation(dict(seed=44,phase='confirmation'),None,None,None)


def test_old_seed44_registration_prevents_replacement_launch(prepared):
    arm=next(iter(module.ARMS))
    path=prepared['directory']/'jobs'/f'{module.control_plan.job_id(arm,"A",44,1.)}.json'
    write_json(path,dict(seed=44))
    with pytest.raises(RuntimeError,match='already exists'):module.prepare()
    ready(prepared)
    with pytest.raises(RuntimeError,match='already exists'):module.run()
    assert not prepared['calls']


def test_two_seed_policy_is_bound_to_queue(prepared):
    plan=module.read(prepared['regularization_target']/'plan.json')
    assert plan['seeds']==[42,43] and plan['confirmation_seeds']==[43]
    assert plan['policy_sha256']==sha256(module.two_seed_policy.destination()/'policy.json')
    policy=module.read(module.two_seed_policy.destination()/'policy.json');policy['seeds'].append(44)
    write_json(module.two_seed_policy.destination()/'policy.json',policy)
    ready(prepared)
    with pytest.raises(ValueError):module.run()
    assert not prepared['calls']
