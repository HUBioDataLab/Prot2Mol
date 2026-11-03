"""Best-macro native evaluation for the user's fixed seed42/base-LR scope.

No learning-rate reconstruction or fitting. Completed legacy predictions may be
reused after their original provenance is verified; partial outputs stay intact.
One process per model runs on one explicitly assigned authorized GPU.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from . import requested_single_seed as requested, postselection, control_queue
    from .control_plan import GPU_UUIDS, digest_json
else:
    from run import ROOT, WORK, sha256, write_json
    import requested_single_seed as requested, postselection, control_queue
    from control_plan import GPU_UUIDS, digest_json

read = control_queue.read
immutable = control_queue.immutable


def destination(identity):
    return WORK / 'requested_evaluation' / identity


def scope():
    """Read-only scope validation usable before either CPU or GPU execution."""
    path = requested.destination() / 'plan.json'
    plan = read(path)
    if (plan['seed'] != 42 or plan['learning_rate_multiplier'] != 1. or
        plan['boot_id'] != control_queue.boot_id() or plan['gpu_uuids'] != list(GPU_UUIDS) or
        plan['workers'] != 4 or plan['arms'] != list(requested.ARMS) or
        plan['training_experiments'] != list(requested.TRAINING_EXPERIMENTS) or
        plan['followup_experiments'] != list(requested.FOLLOWUP_EXPERIMENTS) or
        plan['audit_experiments'] != list(requested.AUDIT_EXPERIMENTS)):
        raise ValueError('Requested single-seed execution scope changed')
    from importlib.metadata import version
    if any(version(n) != v for n, v in plan['packages'].items()):
        raise ValueError('Qualified training packages changed')
    for name, digest in plan['files_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Requested scope evidence changed: ' + name)
    primary = read(WORK / 'controlled/plan.json')
    requested.require_retired()
    if (requested.scope_jobs(primary) != plan['jobs'] or
        requested.reference_inputs() != plan['references'] or
        requested.seed_evidence() != plan['seed_choice']):
        raise ValueError('Requested configurations or reference evidence changed')
    return path, plan


def software():
    from importlib.metadata import version
    paths = sorted((ROOT / 'RewardModel/reward_model').rglob('*.py'))
    paths += [Path(__file__).with_name(n) for n in (
        'requested_evaluation.py', 'requested_single_seed.py', 'postselection.py',
        'checkpoint_evaluation.py', 'control_plan.py', 'control_queue.py',
        'control_driver.py', 'retain_controlled.py', 'run.py')]
    return dict(source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in paths},
        packages={n: version(n) for n in ('torch', 'transformers', 'datasets', 'numpy')})


def legacy_completed(identity, checkpoint, metadata, data):
    """Only completed native outputs qualify, never an unfinished test pass."""
    old_path = WORK / 'postselection' / identity / 'plan.json'
    if not old_path.exists():
        return {}
    old = read(old_path)
    if old != postselection.build_evaluation_plan(WORK / 'controlled', identity):
        raise ValueError('Legacy evaluation plan changed')
    if (old['job'] != metadata or old['data'] != data or
        checkpoint not in old['checkpoints'] or
        old['rules']['best_macro']['checkpoint'] != checkpoint['id']):
        raise ValueError('Legacy evaluation is not the same selected model and data')
    found = {}
    for split in ('val', 'test'):
        output = old_path.parent / checkpoint['id'] / split
        if not (output / 'complete.json').exists():
            continue
        result = read(output / 'complete.json')
        gpu = result['source']['gpu_uuid']
        if gpu not in GPU_UUIDS:
            raise ValueError('Legacy evaluation did not use an authorized GPU')
        source = dict(plan_sha256=sha256(old_path), checkpoint=checkpoint, split=split, gpu_uuid=gpu)
        postselection.verify_evaluation(output, source)
        found[split] = dict(output=str(output), complete_sha256=sha256(output / 'complete.json'),
            original_plan=str(old_path), original_plan_sha256=sha256(old_path), source=source)
    return found


def build_plan(identity):
    scope_path, authorized = scope()
    allowed = [*authorized['references']['jobs'], *authorized['jobs']]
    matches = [j for j in allowed if j['id'] == identity]
    if len(matches) != 1 or Path(identity).name != identity:
        raise ValueError('Evaluation is outside the requested 26 fixed fits')
    primary_path = WORK / 'controlled/plan.json'; primary = read(primary_path)
    job = postselection.registered_job(WORK / 'controlled', identity, primary)
    if job != matches[0] or job['seed'] != 42 or job['learning_rate_multiplier'] != 1.:
        raise ValueError('Fixed evaluation configuration changed')
    run = Path(job['output_dir'])
    completed = read(run / 'controlled_training_complete.json')
    expected_evidence = {'experiment_summary.json', 'trainer_state.json', 'retained/retention.json',
        *{f'rank_{rank}_complete.json' for rank in range(4)}}
    if (completed.get('job_id') != identity or completed.get('phase') != 'training_complete_test_not_scored' or
        set(completed['evidence_sha256']) != expected_evidence):
        raise ValueError('Training completion lacks the native four-rank evidence')
    summary = read(run / 'experiment_summary.json')
    if summary['test_evaluated'] or summary['test_metrics']:
        raise ValueError('Training unexpectedly evaluated the test partition')
    for rank in range(4):
        done = read(run / f'rank_{rank}_complete.json')
        if done != dict(done, rank=rank, world_size=4, config_sha256=job['config_sha256']):
            raise ValueError('Training rank completion changed')
    state, retained = read(run / 'trainer_state.json'), read(run / 'retained/retention.json')
    rows = [r for r in state['log_history'] if 'eval_macro_spearman' in r and 'eval_loss' in r]
    if (state['epoch'] != 15 or [r['epoch'] for r in rows] != list(range(1, 16)) or
        state['global_step'] != rows[-1]['step'] or any(a['step'] >= b['step'] for a, b in zip(rows, rows[1:]))):
        raise ValueError('Selected evaluation needs the complete 15-epoch horizon')
    if retained['selections'] != postselection.selection_rules(rows):
        raise ValueError('Retained selection differs from validation history')
    qualification = read(run / 'runtime_qualification.json')
    if qualification != authorized['references']['qualification']:
        raise ValueError('Requested model used a different training qualification')
    from reward_model.training.config import load_reward_training_config
    config = load_reward_training_config(job['config_path'])
    if config.training.num_train_epochs != 15:
        raise ValueError('Requested training configuration changed its horizon')
    model_config = read(run / 'config.json')
    postselection.verify_model_config(model_config, config.model)
    rule = retained['selections']['best_macro']
    path = run / f"checkpoint-{rule['step']}"
    preserved = {r['path']: r['sha256'] for r in retained['preserved_weights']}
    relative = str((path / 'pytorch_model.bin').relative_to(run))
    if relative not in preserved or sha256(path / 'pytorch_model.bin') != preserved[relative]:
        raise ValueError('Best-macro checkpoint weights changed')
    if read(path / 'config.json') != model_config:
        raise ValueError('Best-macro model configuration changed')
    checkpoint = dict(id=path.name, path=str(path), weights_sha256=preserved[relative],
        model_config_sha256=sha256(path / 'config.json'))
    regime = job['regime']
    data = {split: dict(source=primary['bases'][regime]['data'][f'{split}_parquet_path'],
        source_sha256=primary['source_data_sha256'][regime][split],
        cache=str(Path(config.data.tokenized_dataset_dir) / f'{split}_examples'),
        cache_sha256=primary['tokenized_data_sha256'][regime][split]) for split in ('val', 'test')}
    known = None
    if config.model.contrastive_mask_policy == 'training_known':
        if config.data.train_view_manifest is not None:
            raise ValueError('Requested primary fits cannot silently use a training subset')
        known = dict(source=config.data.train_parquet_path,
            source_sha256=primary['source_data_sha256'][regime]['train'],
            cache=str(Path(config.data.tokenized_dataset_dir) / 'train_examples'),
            cache_sha256=primary['tokenized_data_sha256'][regime]['train'],
            provenance=read(run / 'known_positive_index.json'))
    return dict(format_version=1, job=job, scope_plan_sha256=sha256(scope_path),
        primary_plan_sha256=sha256(primary_path),
        training_complete_sha256=sha256(run / 'controlled_training_complete.json'),
        retention_sha256=sha256(run / 'retained/retention.json'),
        trainer_state_sha256=sha256(run / 'trainer_state.json'),
        runtime_qualification_sha256=sha256(run / 'runtime_qualification.json'),
        checkpoint=checkpoint, rules={'best_macro': dict(**rule, checkpoint=checkpoint['id'])},
        data=data, known_positive_source=known, software=software(),
        reused=legacy_completed(identity, checkpoint, job, data),
        seed=42, learning_rate_multiplier=1., parameter_search=False, test_used_for_selection=False,
        interpretation='One validation-macro-selected checkpoint per requested fit. Fixed seed42/base LR have prior validation exposure. Single-device inference can differ numerically from DDP; saved predictions define reported metrics. No seed uncertainty or causal interpretation of the cross-population A/B gap.')


def prepare(identity):
    plan = build_plan(identity)
    if ROOT == Path('/home/atabey/Prot2Mol'):
        if __package__: from .prepare_random import archive_directory
        else: from prepare_random import archive_directory
        archive_directory(WORK / 'requested_evaluation')
    path = destination(identity) / 'plan.json'
    immutable(path, plan)
    return path


def evaluation(path, plan, split, *, cpu_qualification=False):
    """Verify either a new receipt or an explicitly registered complete reuse."""
    reused = plan['reused'].get(split)
    if reused:
        output = Path(reused['output'])
        if (cpu_qualification or sha256(Path(reused['original_plan'])) != reused['original_plan_sha256'] or
            sha256(output / 'complete.json') != reused['complete_sha256']):
            raise ValueError('Reused evaluation evidence changed')
        source = reused['source']
    else:
        output = path.parent / plan['checkpoint']['id'] / split
        receipt = read(output / 'complete.json')
        gpu = receipt['source']['gpu_uuid']
        if (cpu_qualification and gpu is not None) or (not cpu_qualification and gpu not in GPU_UUIDS):
            raise ValueError('Unexpected selected evaluation device')
        source = dict(plan_sha256=sha256(path), checkpoint=plan['checkpoint'], split=split, gpu_uuid=gpu)
    return postselection.verify_evaluation(output, source, cpu_qualification=cpu_qualification), output


def verify_complete(path, *, cpu_qualification=False):
    path = Path(path); plan = read(path); done = read(path.parent / 'complete.json')
    if plan != build_plan(plan['job']['id']):
        raise ValueError('Requested evaluation selection or source changed')
    if (done['phase'] != 'requested_best_macro_evaluation_complete' or
        done['plan_sha256'] != sha256(path) or set(done['evidence_sha256']) != {'val', 'test'} or
        done['cpu_qualification'] != cpu_qualification):
        raise ValueError('Selected evaluation completion omits or changes a partition')
    evaluations = {}
    for split in ('val', 'test'):
        result, output = evaluation(path, plan, split, cpu_qualification=cpu_qualification)
        if sha256(output / 'complete.json') != done['evidence_sha256'][split]:
            raise ValueError('Selected evaluation receipt changed')
        evaluations[split] = result
    return plan, evaluations


def worker(path, gpu_index, *, cpu_qualification=False):
    if type(gpu_index) is not int or not 0 <= gpu_index < 4:
        raise ValueError('Expected one authorized GPU index')
    device = None if cpu_qualification else GPU_UUIDS[gpu_index]
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != ('' if cpu_qualification else device) or
        os.environ.get('ACCELERATE_USE_CPU', '').lower() != ('true' if cpu_qualification else 'false') or
        (not cpu_qualification and ROOT != Path('/home/atabey/Prot2Mol'))):
        raise RuntimeError('Use the assigned authorized GPU, or explicit CPU qualification')
    path = Path(path); plan = read(path)
    if plan != build_plan(plan['job']['id']):
        raise ValueError('Frozen selected evaluation plan changed')
    if (path.parent / 'complete.json').exists():
        verify_complete(path, cpu_qualification=cpu_qualification); return
    import torch
    from datasets import load_from_disk
    from reward_model.model import load_reward_model
    from reward_model.training.config import load_reward_training_config
    from reward_model.training.known_positives import KnownPositiveIndex
    if __package__: from .checkpoint_evaluation import evaluate_fixed_model
    else: from checkpoint_evaluation import evaluate_fixed_model
    torch.set_num_threads(4)
    config = load_reward_training_config(plan['job']['config_path'])
    known = None
    if plan['known_positive_source'] is not None:
        source = plan['known_positive_source']; postselection.verify_data(source)
        known = KnownPositiveIndex.from_training_dataset(load_from_disk(source['cache']),
            active_threshold=config.model.contrastive_active_threshold)
        if known.provenance != source['provenance']:
            raise ValueError('Reconstructed training-known positives changed')
    model = None; evidence = {}
    for split in ('val', 'test'):
        data = plan['data'][split]
        postselection.verify_data(data)
        output = path.parent / plan['checkpoint']['id'] / split
        if split not in plan['reused'] and not (output / 'complete.json').exists():
            if output.exists():
                raise RuntimeError('Partial selected evaluation requires inspection; no automatic retry')
            if model is None:
                model = load_reward_model(plan['checkpoint']['path'], device=torch.device('cpu'))
                postselection.verify_model_config(model.config.to_dict(), config.model,
                    encoder_widths={s: getattr(model, s + '_encoder').config.hidden_size for s in ('protein', 'molecule')})
                if json.loads(json.dumps(model.config.to_dict())) != read(Path(plan['checkpoint']['path']) / 'config.json'):
                    raise ValueError('Loaded checkpoint configuration changed')
                model.to('cpu' if cpu_qualification else 'cuda')
            source = dict(plan_sha256=sha256(path), checkpoint=plan['checkpoint'], split=split, gpu_uuid=device)
            evaluate_fixed_model(model, load_from_disk(data['cache']), config, output,
                prefix='eval' if split == 'val' else 'test', known_positive_index=known,
                source=source, cpu=cpu_qualification)
        _, verified_output = evaluation(path, plan, split, cpu_qualification=cpu_qualification)
        evidence[split] = sha256(verified_output / 'complete.json')
    if read(path) != plan or build_plan(plan['job']['id']) != plan:
        raise ValueError('Selected evaluation inputs changed during inference')
    immutable(path.parent / 'complete.json', dict(phase='requested_best_macro_evaluation_complete',
        plan_sha256=sha256(path), evidence_sha256=evidence, cpu_qualification=cpu_qualification,
        test_used_for_selection=False, new_fits=0, finished_unix=time.time()))
    verify_complete(path, cpu_qualification=cpu_qualification)


def run_batch(identities):
    """At most four already-complete models; shared lock excludes training."""
    import fcntl
    control_queue.runtime_guard()
    if not 1 <= len(identities) <= 4 or len(set(identities)) != len(identities):
        raise ValueError('Expected one to four distinct requested fits')
    with (WORK / 'controlled/driver.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return dict(phase='waiting_for_global_driver_lock')
        holders = control_queue.gpu_processes()
        if holders: return dict(phase='waiting_for_authorized_gpus', live_processes=holders)
        storage = control_queue.storage_gate()
        if not storage['ready']: return dict(phase='waiting_for_storage', storage=storage)
        paths = [prepare(identity) for identity in identities]
        children, handles = [], []
        try:
            for index, path in enumerate(paths):
                if (path.parent / 'complete.json').exists():
                    verify_complete(path); continue
                env = postselection.launch_environment(identities[index])
                env.update(CUDA_VISIBLE_DEVICES=GPU_UUIDS[index], WANDB_MODE='disabled')
                handle = (path.parent / 'worker.log').open('a'); handles.append(handle)
                command = [str(ROOT / '.venv/bin/python'), str(Path(__file__)), 'worker',
                    '--plan', str(path), '--gpu-index', str(index)]
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                children.append(child)
                write_json(path.parent / 'status.json', dict(phase='evaluating_best_macro',
                    driver_pid=os.getpid(), worker_pid=child.pid, gpu_uuid=GPU_UUIDS[index],
                    plan_sha256=sha256(path), started_unix=time.time()))
            codes = [c.wait() for c in children]
            if any(codes): raise RuntimeError('Selected evaluation failed; inspect without restarting: ' + str(codes))
        finally:
            for child in children:
                if child.poll() is None: child.terminate(); child.wait()
            for handle in handles: handle.close()
        for path in paths: verify_complete(path)
        return dict(phase='requested_best_macro_batch_complete', jobs=list(identities),
            evidence_sha256={str(p.parent / 'complete.json'): sha256(p.parent / 'complete.json') for p in paths},
            new_fits=0, scientific_study_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('prepare').add_argument('--job-id', required=True)
    commands.add_parser('run').add_argument('--job-ids', nargs='+', required=True)
    sub = commands.add_parser('worker'); sub.add_argument('--plan', type=Path, required=True)
    sub.add_argument('--gpu-index', type=int, required=True)
    args = parser.parse_args()
    if args.action == 'prepare': print(prepare(args.job_id))
    elif args.action == 'run': print(json.dumps(run_batch(args.job_ids), indent=2))
    else: worker(args.plan, args.gpu_index)
