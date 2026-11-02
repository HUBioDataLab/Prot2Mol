"""After full-arm confirmations, evaluate and report exactly its ten primary fits.

No training, LR selection, supplemental allocation work or other arms run here.
Wait for the exact parent handle and verified completion; never retry partial work.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

if __package__:
    from .run import ROOT,WORK,sha256,write_json
    from . import control_queue,confirmation_queue,postselection,selected_analysis,controlled_summary
else:
    from run import ROOT,WORK,sha256,write_json
    import control_queue,confirmation_queue,postselection,selected_analysis,controlled_summary

read=control_queue.read
immutable=control_queue.immutable


def destination():
    return WORK/'controlled/full_evaluation_queue'


def possible_jobs():
    plan=read(WORK/'controlled/plan.json')
    return {identity for lr in plan['learning_rate_multipliers'] for identity in controlled_summary.expected_jobs('full',lr)}


def prepare():
    control_queue.runtime_guard()
    parent=WORK/'controlled/full_confirmation_queue'
    plan=read(parent/'plan.json');confirmation_queue.validate_plan(plan)
    invocation,launch=read(parent/'invocation.json'),read(parent/'launch_process.json')
    identity=control_queue.process_identity(invocation['pid'])
    if (identity is None or identity!=launch['identity'] or identity['uid']!=os.getuid() or
        identity['cwd']!=str(ROOT.resolve()) or invocation['plan_sha256']!=sha256(parent/'plan.json') or
        launch['plan_sha256']!=sha256(parent/'plan.json')):
        raise ValueError('Prepare only against the exact live owned confirmation coordinator')
    for job_id in possible_jobs():
        if (WORK/'postselection'/job_id).exists() or (WORK/'postselection_analysis'/job_id).exists():
            raise RuntimeError('Full-arm evaluation/analysis already exists; inspect before scheduling')
    if (WORK/'controlled_summaries/full').exists():raise RuntimeError('Full-arm summary already exists')
    code=controlled_summary.software();files=dict(plan['files_sha256'])
    files.update(code['source_sha256'])
    paths=[parent/name for name in ('plan.json','invocation.json','launch_process.json')]
    paths += [Path(__file__).resolve(),WORK/'tier1/similarity_composition/design/complete.json',
              WORK/'tier1/replicates/noise_summary.json']
    for path in paths:files[str(path.relative_to(ROOT))]=sha256(path)
    result=dict(parent_process=identity,parent_plan_sha256=sha256(parent/'plan.json'),
        boot_id=control_queue.boot_id(),gpu_uuids=list(controlled_summary.control_plan.GPU_UUIDS),
        workers=4,files_sha256=files,packages=code['packages'],
        scope='Only ten primary full-arm fits: native fixed-checkpoint evaluations, per-fit CPU diagnostics and complete primary summary; no training, new selection, other arms, allocation supplement or weight release')
    immutable(destination()/'plan.json',result)
    return destination()/'plan.json'


def validate_plan(plan):
    control_queue.runtime_guard()
    parent=WORK/'controlled/full_confirmation_queue/plan.json'
    if (read(destination()/'plan.json')!=plan or plan['boot_id']!=control_queue.boot_id() or
        plan['gpu_uuids']!=list(controlled_summary.control_plan.GPU_UUIDS) or plan['workers']!=4 or
        sha256(parent)!=plan['parent_plan_sha256']):
        raise ValueError('Evaluation queue plan, parent, host or devices changed')
    confirmation_queue.validate_plan(read(parent))
    from importlib.metadata import version
    if any(version(name)!=value for name,value in plan['packages'].items()):
        raise ValueError('Evaluation/reporting packages changed')
    for name,digest in plan['files_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Frozen evaluation/reporting input changed: '+name)


def upstream_completion(plan):
    parent=WORK/'controlled/full_confirmation_queue'
    parent_plan=read(parent/'plan.json');complete=read(parent/'complete.json')
    selection=read(parent/'selection.json')
    confirmation_queue.verify_confirmation(selection)
    if confirmation_queue.require_tuning_completion(parent_plan)!=read(parent/'upstream_completion.json'):
        raise ValueError('Full tuning completion changed')
    confirmed=control_queue.require_completed(selection['confirmation_jobs'])
    if (complete['phase']!='full_confirmation_training_complete_other_work_pending' or
        complete['plan_sha256']!=plan['parent_plan_sha256'] or complete['evidence']!=confirmed or
        complete['selection_sha256']!=sha256(parent/'selection.json')):
        raise ValueError('Confirmation queue did not complete its exact four fits')
    primary=read(WORK/'controlled/plan.json')
    decision,path=postselection.verify_selection(WORK/'controlled','full',primary)
    if decision!=selection['selection']:raise ValueError('Frozen validation selection changed')
    jobs=[*parent_plan['tuning_jobs'],*selection['confirmation_jobs']]
    expected=controlled_summary.expected_jobs('full',decision['learning_rate_multiplier'])
    if len(jobs)!=10 or {j['id'] for j in jobs}!=set(expected):
        raise ValueError('Expected exactly ten registered full-arm fits')
    evidence=control_queue.require_completed(jobs)
    return dict(jobs=jobs,training_evidence=evidence,parent_complete_sha256=sha256(parent/'complete.json'),
        selection_sha256=sha256(path),parent_selection_sha256=sha256(parent/'selection.json'))


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
            while control_queue.process_identity(plan['parent_process']['pid'])==plan['parent_process']:
                validate_plan(plan)
                status('waiting_for_confirmation_coordinator',parent_process=plan['parent_process'])
                time.sleep(30)
            scope=upstream_completion(plan);assert_scope(plan,scope)
            # A separately invoked evaluator must not be silently adopted or
            # retried. All scientific outputs belong to this one invocation.
            for job in scope['jobs']:
                if (WORK/'postselection'/job['id']).exists() or (WORK/'postselection_analysis'/job['id']).exists():
                    raise RuntimeError('Evaluation/analysis output already exists; refusing automatic retry')
            if (WORK/'controlled_summaries/full').exists():raise RuntimeError('Full-arm summary already exists; inspect before reporting')
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
            result=dict(phase='full_primary_evaluation_and_summary_complete_other_work_pending',
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
