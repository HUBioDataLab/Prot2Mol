"""User two-seed amendment: evaluate and report eight completed full-arm fits.

No training, LR selection, supplemental allocation work or other arms run here.
Require the verified cancellation and seed42/43 completions; never retry partial work.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

if __package__:
    from .run import ROOT,WORK,sha256,write_json
    from . import control_queue,confirmation_queue,postselection,selected_analysis
    from . import two_seed_summary as controlled_summary, two_seed_policy, full_evaluation_queue as legacy_queue
else:
    from run import ROOT,WORK,sha256,write_json
    import control_queue,confirmation_queue,postselection,selected_analysis
    import two_seed_summary as controlled_summary, two_seed_policy, full_evaluation_queue as legacy_queue

read=control_queue.read
immutable=control_queue.immutable


def destination():
    return WORK/'controlled/full_evaluation_two_seed_queue'


def possible_jobs():
    plan=read(WORK/'controlled/plan.json')
    return {identity for lr in plan['learning_rate_multipliers'] for identity in controlled_summary.expected_jobs('full',lr)}


def prepare():
    control_queue.runtime_guard()
    policy_path=two_seed_policy.prepare()
    scope=upstream_completion()
    legacy_path=WORK/'controlled/full_evaluation_queue/plan.json'
    legacy_plan=read(legacy_path);legacy_queue.validate_plan(legacy_plan)
    for job in scope['jobs']:
        if (WORK/'postselection'/job['id']).exists() or (WORK/'postselection_analysis'/job['id']).exists():
            raise RuntimeError('Full-arm evaluation/analysis already exists; inspect before scheduling')
    if (WORK/'controlled_summaries_two_seed/full').exists():raise RuntimeError('Full-arm summary already exists')
    code=controlled_summary.software();files=dict(legacy_plan['files_sha256'])
    files.update(code['source_sha256'])
    paths=[legacy_path,policy_path,Path(__file__).resolve(),
        two_seed_policy.destination()/'cancellation_intent.json',
        two_seed_policy.destination()/'cancellation_complete.json']
    for job in scope['jobs']:
        paths.extend([Path(job['config_path']),WORK/'controlled/jobs'/f"{job['id']}.json",
            Path(job['output_dir'])/'controlled_training_complete.json'])
    for path in paths:files[str(path.relative_to(ROOT))]=sha256(path)
    result=dict(boot_id=control_queue.boot_id(),gpu_uuids=list(controlled_summary.control_plan.GPU_UUIDS),
        workers=4,seeds=[42,43],policy_sha256=sha256(policy_path),
        legacy_evaluation_plan_sha256=sha256(legacy_path),files_sha256=files,packages=code['packages'],
        scope=scope,description='Exactly eight completed full-arm fits, seeds42/43: fixed-checkpoint evaluations, CPU diagnostics and two-seed summary. No fitting, selection, seed44, other arms, allocation supplement or weight release.')
    immutable(destination()/'plan.json',result)
    return destination()/'plan.json'


def validate_plan(plan):
    control_queue.runtime_guard();two_seed_policy.verify()
    legacy_path=WORK/'controlled/full_evaluation_queue/plan.json'
    if (read(destination()/'plan.json')!=plan or plan['boot_id']!=control_queue.boot_id() or
        plan['gpu_uuids']!=list(controlled_summary.control_plan.GPU_UUIDS) or plan['workers']!=4 or
        plan['seeds']!=[42,43] or plan['policy_sha256']!=sha256(two_seed_policy.destination()/'policy.json') or
        plan['legacy_evaluation_plan_sha256']!=sha256(legacy_path)):
        raise ValueError('Two-seed evaluation queue scope, host or policy changed')
    legacy_queue.validate_plan(read(legacy_path))
    from importlib.metadata import version
    if any(version(name)!=value for name,value in plan['packages'].items()):
        raise ValueError('Evaluation/reporting packages changed')
    for name,digest in plan['files_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Frozen evaluation/reporting input changed: '+name)
    if upstream_completion()!=plan['scope']:
        raise ValueError('Completed two-seed scope changed')


def upstream_completion(plan=None):
    two_seed_policy.require_cancelled()
    parent=WORK/'controlled/full_confirmation_queue'
    parent_plan=read(parent/'plan.json');selection=read(parent/'selection.json')
    confirmation_queue.verify_confirmation(selection)
    if confirmation_queue.require_tuning_completion(parent_plan)!=read(parent/'upstream_completion.json'):
        raise ValueError('Full tuning completion changed')
    confirmed=[j for j in selection['confirmation_jobs'] if j['seed']==43]
    if [(j['seed'],j['regime']) for j in confirmed]!=[(43,'A'),(43,'B')]:
        raise ValueError('Expected exactly A/B seed43 confirmations')
    primary=read(WORK/'controlled/plan.json')
    decision,path=postselection.verify_selection(WORK/'controlled','full',primary)
    if decision!=selection['selection']:raise ValueError('Frozen validation selection changed')
    jobs=[*parent_plan['tuning_jobs'],*confirmed]
    expected=controlled_summary.expected_jobs('full',decision['learning_rate_multiplier'])
    if len(jobs)!=8 or {j['id'] for j in jobs}!=set(expected):
        raise ValueError('Expected exactly eight registered seed42/43 full-arm fits')
    for job in jobs:two_seed_policy.require_allowed_job(job)
    evidence=control_queue.require_completed(jobs)
    return dict(jobs=jobs,training_evidence=evidence,selection_sha256=sha256(path),
        parent_selection_sha256=sha256(parent/'selection.json'),
        cancellation_sha256=two_seed_policy.CANCELLATION_SHA256)


def assert_scope(plan,scope):
    validate_plan(plan)
    if upstream_completion(plan)!=scope:raise ValueError('Completed upstream scope changed')


def run():
    import fcntl
    target=destination();plan=read(target/'plan.json');validate_plan(plan)
    with (target/'queue.lock').open('a') as queue_lock:
        fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (target/'invocation.json').exists():raise RuntimeError('Evaluation queue already invoked; inspect, do not restart')
        immutable(target/'invocation.json',dict(pid=os.getpid(),started_unix=time.time(),plan_sha256=sha256(target/'plan.json')))

        def status(phase,**values):
            write_json(target/'status.json',dict(phase=phase,pid=os.getpid(),checked_unix=time.time(),**values))

        completed=[]
        try:
            scope=upstream_completion(plan);assert_scope(plan,scope)
            # A separately invoked evaluator must not be silently adopted or
            # retried. All scientific outputs belong to this one invocation.
            for job in scope['jobs']:
                if (WORK/'postselection'/job['id']).exists() or (WORK/'postselection_analysis'/job['id']).exists():
                    raise RuntimeError('Evaluation/analysis output already exists; refusing automatic retry')
            if (WORK/'controlled_summaries_two_seed/full').exists():raise RuntimeError('Full-arm summary already exists; inspect before reporting')
            immutable(target/'upstream_completion.json',scope)
            for job in scope['jobs']:
                identity=job['id'];output=WORK/'postselection'/identity
                while True:
                    assert_scope(plan,scope)
                    if output.exists():raise RuntimeError('Partial/existing evaluation requires inspection')
                    busy,storage=control_queue.gpu_processes(),control_queue.storage_gate()
                    if busy:
                        waiting=dict(phase='waiting_for_authorized_gpus',live_processes=busy)
                    elif not storage['ready']:
                        waiting=dict(phase='waiting_for_storage',storage=storage)
                    else:
                        status('evaluating_fixed_checkpoints',job_id=identity,storage=storage)
                        started=time.time()
                        try:result=postselection.run_evaluation(identity,workers=plan['workers'])
                        except BlockingIOError:
                            if output.exists():raise RuntimeError('Evaluation started before lock error; inspect before retry')
                            waiting=dict(phase='waiting_for_global_driver_lock')
                        else:
                            if result.get('phase')=='waiting_for_authorized_gpus':
                                if output.exists():raise RuntimeError('Evaluation output appeared while waiting for GPUs')
                                waiting=result
                            elif result.get('phase')=='fixed_weight_evaluation_complete':
                                break
                            else:raise ValueError('Unexpected native evaluator result')
                    status(job_id=identity,**waiting);time.sleep(30)
                assert_scope(plan,scope)
                selected_analysis.verify_completed_evaluations(output)
                status('analyzing_fixed_predictions',job_id=identity)
                analysis=selected_analysis.analyze_job(identity)
                controlled_summary.verify_analysis(analysis,WORK/'tier1/similarity_composition/design')
                assert_scope(plan,scope)
                evidence=dict(job_id=identity,evaluation_complete=str(output/'complete.json'),
                    evaluation_complete_sha256=sha256(output/'complete.json'),analysis=str(analysis),
                    analysis_sha256=sha256(analysis),evaluation_and_analysis_seconds=time.time()-started)
                immutable(target/'jobs'/f'{identity}.json',evidence);completed.append(evidence)
            status('summarizing_full_primary_arm')
            summary=controlled_summary.summarize('full',[r['analysis'] for r in completed])
            assert_scope(plan,scope)
            for record in completed:
                if (sha256(Path(record['evaluation_complete']))!=record['evaluation_complete_sha256'] or
                    sha256(Path(record['analysis']))!=record['analysis_sha256']):
                    raise ValueError('Completed reporting evidence changed')
            result=dict(phase='full_two_seed_evaluation_and_summary_complete_other_work_pending',
                plan_sha256=sha256(target/'plan.json'),upstream_sha256=sha256(target/'upstream_completion.json'),
                evidence=completed,summary=str(summary),summary_sha256=sha256(summary),
                finished_unix=time.time(),study_complete=False,test_used_for_selection=False)
            immutable(target/'complete.json',result);status(result['phase'],completed_jobs=[r['job_id'] for r in completed])
            return result
        except Exception as error:
            status('stopped_for_inspection',error_type=type(error).__name__,error=str(error),completed_jobs=[r['job_id'] for r in completed])
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    args=parser.parse_args()
    print(prepare() if args.action=='prepare' else json.dumps(run(),indent=2))
