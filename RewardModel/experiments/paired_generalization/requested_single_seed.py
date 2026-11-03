"""User-selected experiments at seed42/base LR, with no parameter search.

Reuse frozen primary configurations and the qualified four-GPU training driver.
Only the requested 24 fits run here; evaluations and Tier4 probes are separate.
Historical registry labels do not authorize their former tuning/seed grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import yaml

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from . import control_plan, control_queue, control_driver
else:
    from run import ROOT, WORK, sha256, write_json
    import control_plan, control_queue, control_driver

read = control_queue.read
immutable = control_queue.immutable
SEED = 42
LR_MULTIPLIER = 1.0
TRAINING_EXPERIMENTS = ('E34', 'E35', 'E36', 'E44', 'E45', 'E46', 'E47')
FOLLOWUP_EXPERIMENTS = tuple(f'E{i:02d}' for i in range(49, 59))
AUDIT_EXPERIMENTS = ('E01', 'E02', 'E03', 'E04', 'E05', 'E08', 'E09', 'E19', 'E20')
ARMS = ('frozen', 'last_two_blocks', 'lora4', 'lora16', 'lora64',
        'detach_contrastive', 'listwise_normalized', 'assay_balanced',
        'pairwise_top', 'pairwise_uniform', 'pairwise_hard_zero_margin', 'pairwise_noise_soft')
STOP_SHA256 = '319642db6fcfba77c4d8ecf5cc54d05d52397042c7e1003b15025849a3aae971'


def destination():
    return WORK / 'controlled/requested_single_seed_20261103'


def require_retired():
    target = WORK / 'protocol_amendments/stop_hpo_20261103'
    if sha256(target / 'stop_complete.json') != STOP_SHA256:
        raise ValueError('User-directed stop receipt changed')
    done, intent = read(target / 'stop_complete.json'), read(target / 'stop_intent.json')
    if (done['phase'] != 'user_requested_stop_complete' or
        done['intent_sha256'] != sha256(target / 'stop_intent.json') or
        not done['parameter_optimization_cancelled'] or not done['outputs_preserved']):
        raise ValueError('Prior parameter-search queues were not verifiably stopped')
    for group in intent['groups'].values():
        for identity in group['members']:
            if control_queue.process_identity(identity['pid']) == identity:
                raise RuntimeError('Retired task process is still live')
    return done


def scope_jobs(primary):
    directory = WORK / 'controlled'
    declared = [a for a in primary['arms'] if set(a['experiment_ids']) & set(TRAINING_EXPERIMENTS)]
    if (tuple(a['name'] for a in declared) != ARMS or
        primary['gpu_uuids'] != list(control_plan.GPU_UUIDS)):
        raise ValueError('Requested experiment variants or authorized devices changed')
    jobs = []
    for arm in declared:
        for regime in ('A', 'B'):
            identity = control_plan.job_id(arm['name'], regime, SEED, LR_MULTIPLIER)
            matches = [j for j in primary['tuning_jobs'] if j['id'] == identity]
            if len(matches) != 1:
                raise ValueError('Expected exactly one original fixed configuration: ' + identity)
            registered = matches[0]
            output = directory / 'runs' / identity
            generated = control_plan.make_config(primary['bases'][regime], arm,
                regime=regime, seed=SEED, multiplier=LR_MULTIPLIER, output=output,
                training_noise_rms=primary['training_noise_rms'])
            expected = dict(id=identity, arm=arm['name'], regime=regime, seed=SEED,
                learning_rate_multiplier=LR_MULTIPLIER, phase='tuning', config=generated)
            metadata_path = directory / 'jobs' / (identity + '.json')
            config_path = directory / 'configs' / (identity + '.yaml')
            metadata = read(metadata_path)
            expected_metadata = {k: v for k, v in expected.items() if k != 'config'}
            expected_metadata.update(config_path=str(config_path), config_sha256=sha256(config_path),
                output_dir=str(output), plan_sha256=sha256(directory / 'plan.json'))
            if (registered != expected or metadata != expected_metadata or
                yaml.safe_load(config_path.read_text()) != generated):
                raise ValueError('Requested fixed configuration or registration changed: ' + identity)
            jobs.append(metadata)
    return jobs


def seed_evidence():
    """Use only already completed base-LR B validation to choose the user's seed."""
    records = []
    for seed in (42, 43):
        identity = control_plan.job_id('full', 'B', seed, LR_MULTIPLIER)
        job = read(WORK / 'controlled/jobs' / (identity + '.json'))
        evidence = control_queue.require_completed([job])[0]
        path = Path(job['output_dir']) / 'trainer_state.json'
        state = read(path)
        rows = [r for r in state['log_history'] if 'eval_macro_spearman' in r and 'eval_loss' in r]
        if state['epoch'] != 15 or [r['epoch'] for r in rows] != list(range(1, 16)):
            raise ValueError('Reference seed comparison needs both complete 15-epoch histories')
        best = max(rows, key=lambda r: (r['eval_macro_spearman'], -r['epoch']))
        records.append(dict(seed=seed, best_epoch=best['epoch'],
            best_validation_macro_spearman=best['eval_macro_spearman'],
            trainer_state_sha256=sha256(path), completion=evidence))
    chosen = max(records, key=lambda r: (r['best_validation_macro_spearman'], -r['seed']))
    if chosen['seed'] != SEED:
        raise ValueError('Previously observed best scaffold-validation seed changed')
    return dict(selected_seed=SEED, candidates=records, test_used_for_choice=False,
        additional_seed_fits=0, seed_uncertainty_available=False,
        limitation='Seed42 and base LR have prior validation exposure; this is not independent seed confirmation.')


def reference_inputs():
    directory = WORK / 'controlled'
    jobs = [read(directory / 'jobs' / (control_plan.job_id('full', r, SEED, LR_MULTIPLIER) + '.json'))
            for r in ('A', 'B')]
    evidence = control_queue.require_completed(jobs)
    runtime_path = Path(jobs[0]['output_dir']) / 'runtime_qualification.json'
    qualification = read(runtime_path)
    primary_path = directory / 'plan.json'
    signature = control_plan.digest_json(dict(sources=qualification['sources'], plan_sha256=sha256(primary_path)))
    cache_path = directory / 'qualifications' / signature[:16] / 'complete.json'
    if (qualification['world_size'] != 4 or qualification['signature'] != signature or
        read(cache_path) != qualification or
        read(Path(jobs[1]['output_dir']) / 'runtime_qualification.json') != qualification):
        raise ValueError('Use the completed common four-GPU runtime qualification')
    files = dict(qualification['sources'])
    for name, digest in qualification['evidence_sha256'].items():
        files[str((cache_path.parent / name).relative_to(ROOT))] = digest
    for path in [runtime_path, cache_path, *[Path(r['path']) for r in evidence]]:
        files[str(path.relative_to(ROOT))] = sha256(path)
    for name, digest in files.items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Completed reference/runtime evidence changed: ' + name)
    return dict(jobs=jobs, completion=evidence, qualification=qualification, files_sha256=files)


def require_unstarted(jobs):
    for job in jobs:
        if Path(job['output_dir']).exists():
            raise RuntimeError('Requested fit already has output; inspect before any continuation: ' + job['id'])


def prepare():
    control_queue.runtime_guard(); require_retired()
    primary_path = WORK / 'controlled/plan.json'; primary = read(primary_path)
    if primary['spec_sha256'] != sha256(ROOT / 'Paired input ranking generalization.md'):
        raise ValueError('Original scientific specification changed')
    jobs = scope_jobs(primary); require_unstarted(jobs)
    references = reference_inputs(); seeds = seed_evidence()
    files = dict(references['files_sha256'])
    stop = WORK / 'protocol_amendments/stop_hpo_20261103'
    paths = [primary_path, ROOT / 'Paired input ranking generalization.md', Path(__file__).resolve(),
             Path(control_queue.__file__).resolve(), stop / 'stop_intent.json', stop / 'stop_complete.json']
    for job in [*jobs, *references['jobs']]:
        paths.extend([Path(job['config_path']), WORK / 'controlled/jobs' / (job['id'] + '.json')])
    for seed in (42, 43):
        paths.append(WORK / 'controlled/runs' / control_plan.job_id('full', 'B', seed, LR_MULTIPLIER) / 'trainer_state.json')
    for path in paths:
        files[str(path.relative_to(ROOT))] = sha256(path)
    from importlib.metadata import version
    plan = dict(format_version=1, boot_id=control_queue.boot_id(), seed=SEED,
        learning_rate_multiplier=LR_MULTIPLIER, gpu_uuids=list(control_plan.GPU_UUIDS), workers=4,
        training_experiments=list(TRAINING_EXPERIMENTS), followup_experiments=list(FOLLOWUP_EXPERIMENTS),
        audit_experiments=list(AUDIT_EXPERIMENTS), arms=list(ARMS), jobs=jobs, references=references,
        seed_choice=seeds, files_sha256=files,
        packages={n: version(n) for n in ('torch', 'transformers', 'datasets', 'numpy', 'PyYAML')},
        accounting=dict(new_training_fits=24, reused_reference_fits=2, new_seed_fits=0,
            lr_candidates_per_variant=1, confirmation_fits=0, epochs_per_fit=15),
        user_scope='E34-E35-E36-E44-E45-E46-E47, then E49-E58; E01-E05 and E08-E09-E19-E20. Single best existing seed; skip parameter optimization and repeat full-reference training.',
        scope='Only these 24 original base-LR seed42 configurations; both splits; no LR, weight-decay, loss-weight or seed search. Keep all requested LoRA ranks and matched ranking controls as scientific ablations. Reuse full A/B references. Evaluations and Tier4 probes remain required separate work.',
        registry_note='Original metadata phase=tuning is retained for historical provenance only. This manifest authorizes one fixed configuration per variant/split; do not call the legacy next_job or confirm functions.')
    immutable(destination() / 'plan.json', plan)
    return destination() / 'plan.json'


def validate_plan(plan):
    control_queue.runtime_guard(); require_retired()
    primary = read(WORK / 'controlled/plan.json')
    if (read(destination() / 'plan.json') != plan or plan['boot_id'] != control_queue.boot_id() or
        plan['seed'] != SEED or plan['learning_rate_multiplier'] != LR_MULTIPLIER or
        plan['gpu_uuids'] != list(control_plan.GPU_UUIDS) or plan['workers'] != 4 or
        plan['arms'] != list(ARMS) or plan['training_experiments'] != list(TRAINING_EXPERIMENTS) or
        plan['followup_experiments'] != list(FOLLOWUP_EXPERIMENTS) or plan['audit_experiments'] != list(AUDIT_EXPERIMENTS)):
        raise ValueError('Requested single-seed scope, host or devices changed')
    from importlib.metadata import version
    if any(version(n) != v for n, v in plan['packages'].items()):
        raise ValueError('Qualified runtime packages changed')
    for name, digest in plan['files_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Frozen requested-experiment input changed: ' + name)
    if (scope_jobs(primary) != plan['jobs'] or reference_inputs() != plan['references'] or
        seed_evidence() != plan['seed_choice']):
        raise ValueError('Fixed configurations or completed reference evidence changed')


def run():
    import fcntl
    target = destination(); plan = read(target / 'plan.json'); validate_plan(plan)
    with (target / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (target / 'invocation.json').exists():
            raise RuntimeError('Requested queue already invoked; inspect its handle, do not restart')
        import os
        immutable(target / 'invocation.json', dict(pid=os.getpid(), started_unix=time.time(), plan_sha256=sha256(target / 'plan.json')))
        def status(phase, **values):
            write_json(target / 'status.json', dict(phase=phase, pid=os.getpid(), checked_unix=time.time(), **values))
        completed = []
        try:
            require_unstarted(plan['jobs'])
            for job in plan['jobs']:
                while True:
                    validate_plan(plan)
                    control_queue.require_completed(completed)
                    require_unstarted([job])
                    with (WORK / 'controlled/driver.lock').open('a') as driver_lock:
                        try:
                            fcntl.flock(driver_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            waiting = dict(phase='waiting_for_global_driver_lock')
                        else:
                            busy, storage = control_queue.gpu_processes(), control_queue.storage_gate()
                            if busy:
                                waiting = dict(phase='waiting_for_authorized_gpus', live_processes=busy)
                            elif not storage['ready']:
                                waiting = dict(phase='waiting_for_storage', storage=storage)
                            else:
                                primary = read(WORK / 'controlled/plan.json')
                                gate = control_driver.baseline_gate(WORK, primary)
                                if not gate['ready']:
                                    raise ValueError('Completed baseline evidence is no longer ready')
                                status('training_fixed_configuration', job_id=job['id'], storage=storage)
                                result = control_driver.execute_job(job, primary, gate)
                                if result.get('phase') == 'waiting_for_authorized_gpus' and not Path(job['output_dir']).exists():
                                    waiting = result
                                elif result.get('phase') == 'training_complete_test_not_scored':
                                    validate_plan(plan)
                                    evidence = control_queue.require_completed([job])[0]
                                    immutable(target / 'jobs' / (job['id'] + '.json'), evidence)
                                    completed.append(job)
                                    break
                                else:
                                    raise RuntimeError('Unexpected fixed-configuration training result')
                    status(job_id=job['id'], **waiting); time.sleep(60)
            evidence = control_queue.require_completed(completed)
            result = dict(phase='requested_fixed_config_training_complete_evaluations_and_probes_pending',
                plan_sha256=sha256(target / 'plan.json'), evidence=evidence,
                training_fits_created=len(completed), parameter_search=False, additional_seed_fits=0,
                test_scoring_complete=False, scientific_study_complete=False, finished_unix=time.time())
            immutable(target / 'complete.json', result); status(result['phase'])
            return result
        except Exception as error:
            status('stopped_for_inspection', error_type=type(error).__name__, error=str(error),
                completed_jobs=[j['id'] for j in completed])
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    args = parser.parse_args()
    print(prepare() if args.action == 'prepare' else json.dumps(run(), indent=2))
