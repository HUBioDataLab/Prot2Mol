"""E33: after the full-reference report, finish the two registered decay arms.

Exactly six tuning and two confirmation fits per arm (seeds42/43), then selected evaluation
and reporting. No extra settings, test-driven selection, retries or weight release.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

import yaml

if __package__:
    from .run import ROOT,WORK,sha256,write_json
    from . import control_queue,control_driver,control_plan,two_seed_full_evaluation_queue as full_evaluation_queue
    from . import two_seed_policy,regularization_queue as legacy_queue,two_seed_regularization_analysis as family_analysis
    from . import postselection,selected_analysis,two_seed_summary as controlled_summary
else:
    from run import ROOT,WORK,sha256,write_json
    import control_queue,control_driver,control_plan,two_seed_full_evaluation_queue as full_evaluation_queue
    import two_seed_policy,regularization_queue as legacy_queue,two_seed_regularization_analysis as family_analysis
    import postselection,selected_analysis,two_seed_summary as controlled_summary

read=control_queue.read
immutable=control_queue.immutable
ARMS={'weight_decay_0p001':.001,'weight_decay_0p1':.1}


def destination():return WORK/'controlled/regularization_two_seed_queue'


tuning_jobs=legacy_queue.tuning_jobs


def require_unstarted():
    # Check all historical registrations too, including any unexpected seed44.
    legacy_queue.require_unstarted()
    for arm in ARMS:
        if (WORK/'controlled_summaries_two_seed'/arm).exists():
            raise RuntimeError('Regularization summary already exists; inspect before scheduling')
    if (WORK/'regularization_analysis_two_seed').exists():
        raise RuntimeError('Regularization family report already exists; inspect before scheduling')


def prepare():
    control_queue.runtime_guard();two_seed_policy.verify();parent=WORK/'controlled/full_evaluation_two_seed_queue'
    parent_plan=read(parent/'plan.json');full_evaluation_queue.validate_plan(parent_plan)
    invocation,launch=read(parent/'invocation.json'),read(parent/'launch_process.json')
    identity=control_queue.process_identity(invocation['pid'])
    if (identity is None or identity!=launch['identity'] or identity['uid']!=os.getuid() or
        identity['cwd']!=str(ROOT.resolve()) or invocation['plan_sha256']!=sha256(parent/'plan.json') or
        launch['plan_sha256']!=sha256(parent/'plan.json')):
        raise ValueError('Prepare against the exact live owned full-evaluation coordinator')
    jobs=tuning_jobs(read(WORK/'controlled/plan.json'));require_unstarted()
    files=dict(parent_plan['files_sha256']);files.update(family_analysis.software()['source_sha256'])
    paths=[parent/name for name in ('plan.json','invocation.json','launch_process.json')]+[Path(__file__).resolve(),Path(legacy_queue.__file__).resolve(),two_seed_policy.destination()/'policy.json']
    for entries in jobs.values():
        for job in entries:paths.extend([Path(job['config_path']),WORK/'controlled/jobs'/f"{job['id']}.json"])
    for path in paths:files[str(path.relative_to(ROOT))]=sha256(path)
    plan=dict(parent_process=identity,parent_plan_sha256=sha256(parent/'plan.json'),
        boot_id=control_queue.boot_id(),gpu_uuids=list(control_plan.GPU_UUIDS),workers=4,
        arms=list(ARMS),seeds=[42,43],confirmation_seeds=[43],policy_sha256=sha256(two_seed_policy.destination()/'policy.json'),tuning_jobs=jobs,files_sha256=files,
        scope='Only two registered E33 decay settings, each six tuning and two confirmation fits followed by selected evaluations, primary reports and the three-setting E33 comparison; reuse completed full reference; no other arms, new settings, allocation supplementation or weight release')
    immutable(destination()/'plan.json',plan);return destination()/'plan.json'


def validate_plan(plan):
    control_queue.runtime_guard();two_seed_policy.verify();parent=WORK/'controlled/full_evaluation_two_seed_queue/plan.json'
    if (read(destination()/'plan.json')!=plan or plan['boot_id']!=control_queue.boot_id() or
        plan['gpu_uuids']!=list(control_plan.GPU_UUIDS) or plan['workers']!=4 or plan['arms']!=list(ARMS) or
        sha256(parent)!=plan['parent_plan_sha256'] or plan['seeds']!=[42,43] or plan['confirmation_seeds']!=[43] or
        plan['policy_sha256']!=sha256(two_seed_policy.destination()/'policy.json')):raise ValueError('Regularization queue scope, parent, host or devices changed')
    full_evaluation_queue.validate_plan(read(parent))
    for name,digest in plan['files_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Frozen regularization input changed: '+name)
    if tuning_jobs(read(WORK/'controlled/plan.json'))!=plan['tuning_jobs']:raise ValueError('Frozen tuning registration changed')


def parent_completion(plan):
    parent=WORK/'controlled/full_evaluation_two_seed_queue';complete=read(parent/'complete.json')
    scope=full_evaluation_queue.upstream_completion(read(parent/'plan.json'))
    if (read(parent/'upstream_completion.json')!=scope or
        complete['phase']!='full_two_seed_evaluation_and_summary_complete_other_work_pending' or
        complete['plan_sha256']!=plan['parent_plan_sha256'] or
        complete['upstream_sha256']!=sha256(parent/'upstream_completion.json') or
        len(complete['evidence'])!=8 or {r['job_id'] for r in complete['evidence']}!={j['id'] for j in scope['jobs']}):
        raise ValueError('Full-reference evaluation did not complete its exact scope')
    paths={str(parent/name):sha256(parent/name) for name in ('complete.json','upstream_completion.json')}
    for record in complete['evidence']:
        if Path(record['evaluation_complete'])!=WORK/'postselection'/record['job_id']/'complete.json':
            raise ValueError('Parent evaluation belongs to another fit')
        for key in ('evaluation_complete','analysis'):
            if sha256(Path(record[key]))!=record[key+'_sha256']:raise ValueError('Parent evaluation evidence changed')
            paths[record[key]]=record[key+'_sha256']
    summary=controlled_summary.summarize('full',[r['analysis'] for r in complete['evidence']])
    if str(summary)!=complete['summary'] or sha256(summary)!=complete['summary_sha256']:
        raise ValueError('Parent summary differs from verified primary analyses')
    paths[str(summary)]=sha256(summary)
    return dict(files_sha256=paths,full_summary=str(summary),full_summary_sha256=sha256(summary))


def assert_inputs(plan,upstream):
    validate_plan(plan)
    for path,digest in upstream['files_sha256'].items():
        if sha256(Path(path))!=digest:raise ValueError('Completed full-reference handoff changed')


def verify_confirmation(arm,selection):
    directory=WORK/'controlled';primary=read(directory/'plan.json')
    decision,path=postselection.verify_selection(directory,arm,primary)
    if decision!=selection['selection']:raise ValueError('Regularization validation selection changed')
    jobs=selection['confirmation_jobs'];lr=decision['learning_rate_multiplier']
    if [(j['seed'],j['regime']) for j in jobs]!=[(43,'A'),(43,'B')]:
        raise ValueError('Expected only the two seed43 A/B confirmation fits')
    for job in jobs:
        identity=control_plan.job_id(arm,job['regime'],job['seed'],lr)
        config=directory/'configs'/f'{identity}.yaml';output=directory/'runs'/identity
        expected=dict(id=identity,arm=arm,regime=job['regime'],seed=job['seed'],learning_rate_multiplier=lr,
            phase='confirmation',selection_sha256=sha256(path),config_path=str(config),config_sha256=sha256(config),
            output_dir=str(output),plan_sha256=sha256(directory/'plan.json'))
        original=next(j for j in primary['tuning_jobs'] if j['arm']==arm and j['regime']==job['regime'] and j['learning_rate_multiplier']==lr)
        generated=copy.deepcopy(original['config']);generated['training'].update(seed=job['seed'],output_dir=str(output))
        if job!=expected or read(directory/'jobs'/f'{identity}.json')!=job or yaml.safe_load(config.read_text())!=generated:
            raise ValueError('Frozen regularization confirmation registration changed')


def train(job,plan,upstream,status):
    two_seed_policy.require_allowed_job(job)
    import fcntl
    directory=WORK/'controlled';output=Path(job['output_dir'])
    while True:
        assert_inputs(plan,upstream)
        if job['phase']=='confirmation':verify_confirmation(job['arm'],read(destination()/'selections'/f"{job['arm']}.json"))
        if output.exists():raise RuntimeError('Training output already exists; refusing automatic retry')
        waiting=None
        with (directory/'driver.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:waiting=dict(phase='waiting_for_global_driver_lock')
            else:
                busy,storage=control_queue.gpu_processes(),control_queue.storage_gate()
                if busy:waiting=dict(phase='waiting_for_authorized_gpus',live_processes=busy)
                elif not storage['ready']:waiting=dict(phase='waiting_for_storage',storage=storage)
                else:
                    primary=read(directory/'plan.json');gate=control_driver.baseline_gate(WORK,primary)
                    if not gate['ready']:raise ValueError('Completed baseline gate changed')
                    status('training',job_id=job['id'],storage=storage)
                    result=control_driver.execute_job(job,primary,gate)
                    if result.get('phase')=='waiting_for_authorized_gpus' and not output.exists():waiting=result
                    elif result.get('phase')!='training_complete_test_not_scored':raise ValueError('Unexpected training result')
                    else:
                        evidence=control_queue.require_completed([job]);assert_inputs(plan,upstream)
                        return evidence[0]
        status(job_id=job['id'],**waiting);time.sleep(30)


def evaluate(job,plan,upstream,status):
    two_seed_policy.require_allowed_job(job)
    identity=job['id'];output=WORK/'postselection'/identity
    while True:
        assert_inputs(plan,upstream)
        if output.exists() or (WORK/'postselection_analysis'/identity).exists():raise RuntimeError('Evaluation output already exists; refusing retry')
        busy,storage=control_queue.gpu_processes(),control_queue.storage_gate()
        if busy:waiting=dict(phase='waiting_for_authorized_gpus',live_processes=busy)
        elif not storage['ready']:waiting=dict(phase='waiting_for_storage',storage=storage)
        else:
            status('evaluating_fixed_checkpoints',job_id=identity,storage=storage)
            try:result=postselection.run_evaluation(identity,workers=plan['workers'])
            except BlockingIOError:
                if output.exists():raise RuntimeError('Evaluation started before lock error; inspect before retry')
                waiting=dict(phase='waiting_for_global_driver_lock')
            else:
                if result.get('phase')=='waiting_for_authorized_gpus' and not output.exists():waiting=result
                elif result.get('phase')=='fixed_weight_evaluation_complete':break
                else:raise ValueError('Unexpected native evaluation result')
        status(job_id=identity,**waiting);time.sleep(30)
    assert_inputs(plan,upstream);selected_analysis.verify_completed_evaluations(output)
    status('analyzing_fixed_predictions',job_id=identity)
    analysis=selected_analysis.analyze_job(identity)
    controlled_summary.verify_analysis(analysis,WORK/'tier1/similarity_composition/design')
    assert_inputs(plan,upstream)
    return dict(job_id=identity,evaluation_complete=str(output/'complete.json'),
        evaluation_complete_sha256=sha256(output/'complete.json'),analysis=str(analysis),analysis_sha256=sha256(analysis))


def run():
    import fcntl
    target=destination();plan=read(target/'plan.json');validate_plan(plan)
    with (target/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (target/'invocation.json').exists():raise RuntimeError('Regularization queue already invoked; inspect, do not restart')
        immutable(target/'invocation.json',dict(pid=os.getpid(),started_unix=time.time(),plan_sha256=sha256(target/'plan.json')))
        def status(phase,**values):write_json(target/'status.json',dict(phase=phase,pid=os.getpid(),checked_unix=time.time(),**values))
        finished=[]
        try:
            while control_queue.process_identity(plan['parent_process']['pid'])==plan['parent_process']:
                validate_plan(plan);status('waiting_for_full_reference_report',parent_process=plan['parent_process']);time.sleep(30)
            upstream=parent_completion(plan);assert_inputs(plan,upstream);require_unstarted()
            immutable(target/'upstream_completion.json',upstream)
            for arm in plan['arms']:
                trained=[]
                for job in plan['tuning_jobs'][arm]:
                    evidence=train(job,plan,upstream,status);trained.append(evidence)
                    immutable(target/'training'/f"{job['id']}.json",evidence)
                while True:
                    assert_inputs(plan,upstream)
                    with (WORK/'controlled/driver.lock').open('a') as driver_lock:
                        try:fcntl.flock(driver_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:pass
                        else:
                            selection=two_seed_policy.confirm(arm);verify_confirmation(arm,selection)
                            if any(Path(j['output_dir']).exists() for j in selection['confirmation_jobs']):raise RuntimeError('Confirmation output already exists')
                            immutable(target/'selections'/f'{arm}.json',selection);break
                    status('waiting_for_global_driver_lock',arm=arm);time.sleep(30)
                for job in selection['confirmation_jobs']:
                    verify_confirmation(arm,selection)
                    evidence=train(job,plan,upstream,status);trained.append(evidence)
                    immutable(target/'training'/f"{job['id']}.json",evidence)
                jobs=[*plan['tuning_jobs'][arm],*selection['confirmation_jobs']];verify_confirmation(arm,selection)
                if {j['id'] for j in jobs}!=set(controlled_summary.expected_jobs(arm,selection['selection']['learning_rate_multiplier'])):
                    raise ValueError('Regularization fits differ from the exact eight-fit scope')
                evaluations=[]
                for job in jobs:
                    evidence=evaluate(job,plan,upstream,status);evaluations.append(evidence)
                    immutable(target/'evaluations'/f"{job['id']}.json",evidence)
                status('summarizing_primary_arm',arm=arm)
                summary=controlled_summary.summarize(arm,[r['analysis'] for r in evaluations])
                assert_inputs(plan,upstream);verify_confirmation(arm,selection)
                if control_queue.require_completed(jobs)!=trained:raise ValueError('Training completion evidence changed')
                for record in evaluations:
                    for key in ('evaluation_complete','analysis'):
                        if sha256(Path(record[key]))!=record[key+'_sha256']:raise ValueError('Evaluation evidence changed')
                result=dict(arm=arm,selection_sha256=sha256(target/'selections'/f'{arm}.json'),training=trained,
                    evaluations=evaluations,summary=str(summary),summary_sha256=sha256(summary))
                immutable(target/'arms'/f'{arm}.json',result);finished.append(result)
            assert_inputs(plan,upstream)
            for arm_result in finished:
                arm=arm_result['arm'];selection_path=target/'selections'/f'{arm}.json'
                if (sha256(selection_path)!=arm_result['selection_sha256'] or
                    sha256(Path(arm_result['summary']))!=arm_result['summary_sha256']):
                    raise ValueError('Earlier completed regularization arm changed')
                selection=read(selection_path);verify_confirmation(arm,selection)
                if control_queue.require_completed([*plan['tuning_jobs'][arm],*selection['confirmation_jobs']])!=arm_result['training']:
                    raise ValueError('Earlier training completion changed')
                for record in arm_result['evaluations']:
                    for key in ('evaluation_complete','analysis'):
                        if sha256(Path(record[key]))!=record[key+'_sha256']:raise ValueError('Earlier evaluation evidence changed')
            status('comparing_regularization_settings')
            family_selection=family_analysis.prepare_selection()
            family_report=family_analysis.analyze([upstream['full_summary'],*[r['summary'] for r in finished]])
            assert_inputs(plan,upstream)
            result=dict(phase='regularization_two_seed_reports_complete_other_work_pending',
                family_selection=str(family_selection),family_selection_sha256=sha256(family_selection),
                family_report=str(family_report),family_report_sha256=sha256(family_report),
                plan_sha256=sha256(target/'plan.json'),upstream_sha256=sha256(target/'upstream_completion.json'),
                arms=finished,finished_unix=time.time(),study_complete=False,test_used_for_selection=False,
                training_fits_created=16,additional_settings_created=0)
            immutable(target/'complete.json',result);status(result['phase'],completed_arms=list(ARMS));return result
        except Exception as error:
            status('stopped_for_inspection',error_type=type(error).__name__,error=str(error),completed_arms=[r['arm'] for r in finished]);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'))
    args=parser.parse_args();print(prepare() if args.action=='prepare' else json.dumps(run(),indent=2))
