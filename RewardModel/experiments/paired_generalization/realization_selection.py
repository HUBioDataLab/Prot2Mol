"""Select LRs independently for dependent B allocations and cover their union.

Uses only the frozen validation trajectories. Primary A/B selection is unchanged.
Extra confirmations share the exact B training support and run only after the
primary arm's four confirmations. No test scoring or checkpoint release occurs.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import yaml

if __package__:
    from .run import ROOT, WORK, sha256
    from . import control_plan, control_driver, control_queue, postselection
    from . import realization_predictions, realization_protocol
else:
    from run import ROOT, WORK, sha256
    import control_plan, control_driver, control_queue, postselection
    import realization_predictions, realization_protocol


def read(path):
    return json.loads(Path(path).read_text())


def destination(arm):
    if not arm or Path(arm).name != arm:
        raise ValueError('Expected an arm name, not a path')
    return WORK/'controlled/realization_selections'/arm


def select_allocations(candidates, allocations):
    """The three inputs contain validation-only epoch choices, never test scores."""
    if (len(candidates) != 3 or
        {r['learning_rate_multiplier'] for r in candidates} != set(control_plan.LEARNING_RATE_MULTIPLIERS)):
        raise ValueError('All three distinct LR candidates are required')
    if len(set(allocations)) != 3:
        raise ValueError('Exactly three frozen allocations are required')
    for item in candidates:
        if item['test_metrics_computed'] is not False or set(item['selections']) != set(allocations):
            raise ValueError('Expected validation-only evidence for every allocation')
    selections = {}
    for name in sorted(allocations):
        scores, checkpoints = [], {}
        for candidate in sorted(candidates, key=lambda r:r['learning_rate_multiplier']):
            item = candidate['selections'][name]
            history = item['validation_history']
            if [r['epoch'] for r in history] != list(range(1,16)):
                raise ValueError('Every candidate needs all 15 validation epochs')
            chosen = realization_predictions.choose_epochs(history)
            if item['selected'] != chosen:
                raise ValueError('Checkpoint choice differs from validation trajectory')
            lr = candidate['learning_rate_multiplier']
            scores.append(dict(learning_rate_multiplier=lr,
                validation_macro_spearman=chosen['macro_spearman'],
                validation_defined_fraction=chosen['defined_fraction']))
            checkpoints[str(lr)] = dict(job_id=candidate['job_id'], **chosen)
        selection = control_plan.select_learning_rate(scores)
        selection.update(criterion='maximum allocation validation macro Spearman; ties choose lower LR; earliest epoch breaks checkpoint ties',
            checkpoint=checkpoints[str(selection['learning_rate_multiplier'])],
            candidate_checkpoints=checkpoints)
        selections[name] = selection
    return selections


def confirmation_requests(arm, selections, primary_multiplier):
    requests = []
    union = sorted({s['learning_rate_multiplier'] for s in selections.values()})
    for seed in control_plan.CONFIRMATION_SEEDS:
        for lr in union:
            requests.append(dict(id=control_plan.job_id(arm,'B',seed,lr),seed=seed,
                learning_rate_multiplier=lr,
                reuse_primary=lr==primary_multiplier,
                allocations=sorted(name for name,s in selections.items() if s['learning_rate_multiplier']==lr)))
    return requests


def build_selection(arm):
    """Revalidate completed native fits and all allocation export payloads."""
    directory = WORK/'controlled'
    plan = read(directory/'plan.json')
    if plan['spec_sha256'] != sha256(ROOT/'Paired input ranking generalization.md'):
        raise ValueError('Study specification changed')
    primary,primary_path = postselection.verify_selection(directory,arm,plan)
    contract_path = directory/'realizations/plan.json'
    contract,_,_ = realization_protocol.load_contract(contract_path.parent)
    if (contract['training_identical'] is not True or
        contract['control_plan_sha256'] != sha256(directory/'plan.json') or
        contract['inference'] != dict(batch_size=32,autocast='bfloat16',scalar_exports_only=True)):
        raise ValueError('Allocation training support or inference protocol changed')
    candidates,evidence = [],[]
    for job in (j for j in plan['tuning_jobs'] if j['arm']==arm and j['regime']=='B'):
        metadata = postselection.registered_job(directory,job['id'],plan)
        run = Path(metadata['output_dir'])
        result = realization_predictions.verify_validation(run)
        path = realization_predictions.destination(run)/'validation_complete.json'
        retained = read(run/'retained/retention.json')['additional_realizations']
        if (Path(retained['path']) != path or retained['sha256'] != sha256(path) or
            result['config_sha256'] != metadata['config_sha256'] or
            result['contract_sha256'] != sha256(contract_path)):
            raise ValueError('Allocation selection differs from completed training/retention evidence')
        candidates.append(dict(job_id=job['id'],learning_rate_multiplier=job['learning_rate_multiplier'],
            test_metrics_computed=result['test_metrics_computed'],selections=result['selections']))
        evidence.append(dict(job_id=job['id'],metadata_sha256=sha256(directory/'jobs'/f"{job['id']}.json"),
            completion_sha256=sha256(run/'controlled_training_complete.json'),
            validation_path=str(path),validation_sha256=sha256(path)))
    selections = select_allocations(candidates,contract['realizations'])
    requests = confirmation_requests(arm,selections,primary['learning_rate_multiplier'])
    sources = [Path(__file__),*(Path(m.__file__) for m in
        (control_plan,control_driver,control_queue,postselection,realization_predictions,realization_protocol))]
    return dict(format_version=1,arm=arm,primary_plan_sha256=sha256(directory/'plan.json'),
        primary_selection_sha256=sha256(primary_path),contract_sha256=sha256(contract_path),
        sources_sha256={str(p.resolve().relative_to(ROOT.resolve())):sha256(p) for p in sources},
        evidence=evidence,selections=selections,confirmation_requests=requests,
        winning_lr_union=sorted({s['learning_rate_multiplier'] for s in selections.values()}),
        primary_learning_rate_multiplier=primary['learning_rate_multiplier'],
        confirmation_counts=dict(required=len(requests),reused_primary=sum(r['reuse_primary'] for r in requests),
            additional=sum(not r['reuse_primary'] for r in requests)),
        test_metrics_computed=False,primary_selection_unchanged=True,
        interpretation=contract['limitations'],
        remaining='Confirmation fits, selected-model test evaluation, allocation full losses and averaging; no checkpoint release')


def prepare(arm):
    selection = build_selection(arm)
    control_queue.immutable(destination(arm)/'selection.json',selection)
    return selection


def verify_selection(arm):
    path = destination(arm)/'selection.json'
    selected = read(path)
    if build_selection(arm) != selected:
        raise ValueError('Frozen allocation LR selection or provenance changed')
    return selected


def expected_supplement(request, selection, plan):
    arm = selection['arm']
    matches = [j for j in plan['tuning_jobs'] if j['arm']==arm and j['regime']=='B' and
               j['learning_rate_multiplier']==request['learning_rate_multiplier']]
    if len(matches) != 1 or request['reuse_primary']:
        raise ValueError('Supplement must match exactly one additional B training configuration')
    config = copy.deepcopy(matches[0]['config'])
    config['training'].update(seed=request['seed'],output_dir=str(WORK/'controlled/runs'/request['id']))
    return dict(id=request['id'],arm=arm,regime='B',seed=request['seed'],
        learning_rate_multiplier=request['learning_rate_multiplier'],phase='allocation_confirmation',config=config,
        allocation_selection_sha256=sha256(destination(arm)/'selection.json'))


def verify_registration(job, expected, plan_digest):
    directory = WORK/'controlled'
    config = directory/'configs'/f"{expected['id']}.yaml"
    metadata = {key:value for key,value in expected.items() if key!='config'}
    metadata.update(config_path=str(config),config_sha256=sha256(config),
        output_dir=expected['config']['training']['output_dir'],plan_sha256=plan_digest)
    if (job != metadata or read(directory/'jobs'/f"{job['id']}.json") != metadata or
        yaml.safe_load(config.read_text()) != expected['config']):
        raise ValueError('Supplemental confirmation registration changed')


def run_next(arm):
    """Run at most one extra fit under the existing global lock; never retry."""
    import fcntl
    control_queue.runtime_guard()
    if os.environ.get('ACCELERATE_USE_CPU','false').lower() not in ('false','0','no'):
        raise RuntimeError('GPU-launching coordinator must not force Accelerate CPU mode')
    directory = WORK/'controlled'
    with (directory/'driver.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(phase='waiting_for_global_driver_lock')
        selection = verify_selection(arm)
        plan = read(directory/'plan.json')
        # Keep the primary queue's four-job order and configurations untouched.
        for seed in control_plan.CONFIRMATION_SEEDS:
            for regime in ('A','B'):
                identity = control_plan.job_id(arm,regime,seed,selection['primary_learning_rate_multiplier'])
                path = directory/'jobs'/f'{identity}.json'
                if not path.exists() or not control_driver.completed_job(read(path)):
                    return dict(phase='waiting_for_primary_confirmations',job_id=identity)
                metadata = postselection.registered_job(directory,identity,plan)
                if (metadata['phase'] != 'confirmation' or metadata['seed'] != seed or metadata['regime'] != regime or
                    metadata['selection_sha256'] != selection['primary_selection_sha256']):
                    raise ValueError('Primary confirmation differs from its frozen selection')
        completed = []
        for request in selection['confirmation_requests']:
            if request['reuse_primary']:
                metadata = postselection.registered_job(directory,request['id'],plan)
            else:
                expected = expected_supplement(request,selection,plan)
                metadata = control_plan.materialize_job(expected,directory,selection['primary_plan_sha256'])
                verify_registration(metadata,expected,selection['primary_plan_sha256'])
                if not control_driver.completed_job(metadata):
                    if Path(metadata['output_dir']).exists():
                        raise RuntimeError('Supplement output already exists; inspect, never automatically retry')
                    gate = control_driver.baseline_gate(WORK,plan)
                    if not gate['ready']:
                        raise RuntimeError('Completed baseline gate is no longer satisfied')
                    busy,storage = control_queue.gpu_processes(),control_queue.storage_gate()
                    if busy:return dict(phase='waiting_for_authorized_gpus',live_processes=busy)
                    if not storage['ready']:return dict(phase='waiting_for_storage',storage=storage)
                    result = control_driver.execute_job(metadata,plan,gate)
                    verify_selection(arm)
                    verify_registration(metadata,expected,selection['primary_plan_sha256'])
                    if result.get('phase') == 'waiting_for_authorized_gpus':return result
                    control_queue.require_completed([metadata])
                    return dict(phase='one_allocation_confirmation_complete',job_id=metadata['id'],study_complete=False)
            allocation = realization_predictions.verify_validation(Path(metadata['output_dir']))
            path = realization_predictions.destination(Path(metadata['output_dir']))/'validation_complete.json'
            retained = read(Path(metadata['output_dir'])/'retained/retention.json')['additional_realizations']
            if (retained['sha256'] != sha256(path) or Path(retained['path']) != path or
                allocation['config_sha256'] != metadata['config_sha256'] or
                allocation['contract_sha256'] != selection['contract_sha256']):
                raise ValueError('Confirmation allocation exports differ from completed training')
            completed.append(dict(job_id=metadata['id'],allocations=request['allocations'],
                completion_sha256=sha256(Path(metadata['output_dir'])/'controlled_training_complete.json'),
                validation_path=str(path),validation_sha256=sha256(path)))
        result = dict(phase='allocation_confirmation_union_complete_other_work_pending',
            selection_sha256=sha256(destination(arm)/'selection.json'),evidence=completed,study_complete=False)
        control_queue.immutable(destination(arm)/'confirmation_complete.json',result)
        return result


def preflight(arm):
    """Read-only readiness check; an incomplete fit is not a failed fit."""
    directory = WORK/'controlled'
    plan = read(directory/'plan.json')
    jobs = [j for j in plan['tuning_jobs'] if j['arm']==arm]
    if len(jobs)!=6:raise ValueError('Expected six registered primary tuning jobs')
    contract,_,_ = realization_protocol.load_contract(directory/'realizations')
    missing = [j['id'] for j in jobs if not control_driver.completed_job(read(directory/'jobs'/f"{j['id']}.json"))]
    return dict(phase='waiting_for_primary_tuning_and_selection' if missing or not
        (directory/'selections'/f'{arm}.json').exists() else 'ready_to_verify_allocation_selection',
        incomplete_tuning=missing,primary_selection_exists=(directory/'selections'/f'{arm}.json').exists(),
        allocations=sorted(contract['realizations']),training_identical=contract['training_identical'],
        test_metrics_computed=False,study_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('preflight','prepare','verify','run-next'))
    parser.add_argument('--arm',required=True)
    args = parser.parse_args()
    print(json.dumps({'preflight':preflight,'prepare':prepare,'verify':verify_selection,'run-next':run_next}[args.action](args.arm),indent=2))
