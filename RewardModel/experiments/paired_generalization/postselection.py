"""Freeze and evaluate retained primary-grid models after validation selection.

Training outputs are read-only. Every LR candidate contributes its best-macro
checkpoint to the A/B plot; only the common B-selected LR receives the full
checkpoint-rule comparison. Additional B allocations use a separate protocol.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .control_plan import (GPU_UUIDS, CONFIRMATION_SEEDS, _write_immutable,
                               select_learning_rate, token_cache_hashes, validation_candidate)
    from .control_driver import completed_job, busy_authorized_devices, launch_environment
    from .retain_controlled import selection_rules
else:
    from run import ROOT, WORK, sha256, write_json
    from control_plan import (GPU_UUIDS, CONFIRMATION_SEEDS, _write_immutable,
                              select_learning_rate, token_cache_hashes, validation_candidate)
    from control_driver import completed_job, busy_authorized_devices, launch_environment
    from retain_controlled import selection_rules


def read_json(path):
    return json.loads(Path(path).read_text())


def immutable_json(path, value):
    _write_immutable(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def verify_model_config(saved, launch_config, *, encoder_widths=None):
    """Accept only encoder resolution of unspecified launch widths.

    Planning can validate the serialized configuration without loading weights.
    The inference worker additionally checks widths against the loaded encoders.
    No launch or saved configuration is mutated.
    """
    actual = json.loads(json.dumps(saved))
    expected = json.loads(json.dumps(launch_config.to_dict()))
    if actual.keys() != expected.keys():
        raise ValueError('Saved model configuration fields differ from the launch')
    for side in ('protein', 'molecule'):
        key = side+'_hidden_size'
        width = actual[key]
        if type(width) is not int or width <= 0:
            raise ValueError(f'Saved {key} is not a resolved positive integer')
        if encoder_widths is not None and width != encoder_widths[side]:
            raise ValueError(f'Saved {key} differs from its loaded encoder')
        if expected[key] is None:
            expected[key] = width
    differing = sorted(key for key in actual if actual[key] != expected[key])
    if differing:
        raise ValueError(f'Saved model configuration differs from the launch: {differing}')


def registered_job(directory, identity, plan):
    """Check registration against the frozen configuration, not only its hash."""
    directory = Path(directory)
    if Path(identity).name != identity:
        raise ValueError('Expected a job ID, not a path')
    metadata = read_json(directory/'jobs'/f'{identity}.json')
    if (metadata['id'] != identity or metadata['plan_sha256'] != sha256(directory/'plan.json') or
        Path(metadata['config_path']) != directory/'configs'/f'{identity}.yaml' or
        Path(metadata['output_dir']) != directory/'runs'/identity or
        sha256(Path(metadata['config_path'])) != metadata['config_sha256']):
        raise ValueError('Job registration or configuration changed')
    candidates = [job for job in plan['tuning_jobs']
        if job['arm'] == metadata['arm'] and job['regime'] == metadata['regime'] and
        job['learning_rate_multiplier'] == metadata['learning_rate_multiplier']]
    if len(candidates) != 1:
        raise ValueError('Job is outside the frozen primary grid')
    expected = copy.deepcopy(candidates[0]['config'])
    if metadata['phase'] == 'confirmation':
        if metadata['seed'] not in CONFIRMATION_SEEDS:
            raise ValueError('Unexpected confirmation seed')
        expected['training'].update(seed=metadata['seed'], output_dir=metadata['output_dir'])
    elif metadata['phase'] != 'tuning' or identity != candidates[0]['id'] or metadata['seed'] != candidates[0]['seed']:
        raise ValueError('Unexpected tuning registration')
    import yaml
    if yaml.safe_load(Path(metadata['config_path']).read_text()) != expected:
        raise ValueError('Job configuration differs from the frozen comparison')
    if not completed_job(metadata):
        raise RuntimeError(f'Training/retention is not complete: {identity}')
    return metadata


def verify_selection(directory, arm, plan):
    """Reconstruct the decision exclusively from all six validation candidates."""
    directory = Path(directory)
    candidates, evidence = [], []
    jobs = [j for j in plan['tuning_jobs'] if j['arm'] == arm]
    if len(jobs) != 6 or {j['regime'] for j in jobs} != {'A', 'B'}:
        raise ValueError('Expected all six matched tuning jobs')
    for job in jobs:
        metadata = registered_job(directory, job['id'], plan)
        candidate, source = validation_candidate(metadata)
        evidence.append(source)
        if metadata['regime'] == 'B':
            candidates.append(candidate)
    expected = dict(arm=arm, plan_sha256=sha256(directory/'plan.json'),
                    **select_learning_rate(candidates), evidence=evidence)
    path = directory/'selections'/f'{arm}.json'
    if read_json(path) != expected:
        raise ValueError('Frozen LR decision differs from validation-only reconstruction')
    return expected, path


def build_evaluation_plan(directory, identity):
    directory = Path(directory)
    plan = read_json(directory/'plan.json')
    metadata = registered_job(directory, identity, plan)
    decision, selection_path = verify_selection(directory, metadata['arm'], plan)
    selected = metadata['learning_rate_multiplier'] == decision['learning_rate_multiplier']
    if metadata['phase'] == 'confirmation' and (
        not selected or metadata['selection_sha256'] != sha256(selection_path)):
        raise ValueError('Confirmation does not use the frozen B-selected LR')
    run = Path(metadata['output_dir'])
    retained = read_json(run/'retained/retention.json')
    state = read_json(run/'trainer_state.json')
    records = [r for r in state['log_history'] if 'eval_macro_spearman' in r and 'eval_loss' in r]
    if retained['selections'] != selection_rules(records):
        raise ValueError('Checkpoint rules differ from validation history')
    from reward_model.training.config import load_reward_training_config
    config = load_reward_training_config(metadata['config_path'])
    model_config = read_json(run/'config.json')
    verify_model_config(model_config, config.model)
    names = list(retained['selections']) if selected else ['best_macro']
    rules, checkpoints = {}, {}
    preserved = {item['path']: item['sha256'] for item in retained['preserved_weights']}

    def register(path):
        relative = str((path/'pytorch_model.bin').relative_to(run))
        if relative not in preserved or sha256(path/'pytorch_model.bin') != preserved[relative]:
            raise ValueError('Retained checkpoint is absent or changed')
        if read_json(path/'config.json') != model_config:
            raise ValueError('Retained model configuration changed')
        key = str(path.relative_to(run)).replace('/', '__')
        checkpoints.setdefault(key, dict(id=key, path=str(path), weights_sha256=preserved[relative],
                                        model_config_sha256=sha256(path/'config.json')))
        return key

    for name in names:
        rule = retained['selections'][name]
        rules[name] = dict(**rule, checkpoint=register(run/f"checkpoint-{rule['step']}"))
    if selected:
        rules['average_from_loss_minimum'] = dict(**retained['averaging'],
            checkpoint=register(run/'retained/average_from_loss_minimum'))
    # Rules referring to the same epoch share inference. A singleton average
    # remains explicit: its bytes can differ despite equal tensors.
    regime = metadata['regime']
    data = {split: dict(source=plan['bases'][regime]['data'][f'{split}_parquet_path'],
        source_sha256=plan['source_data_sha256'][regime][split],
        cache=str(Path(config.data.tokenized_dataset_dir)/f'{split}_examples'),
        cache_sha256=plan['tokenized_data_sha256'][regime][split]) for split in ('val', 'test')}
    known_positive_source = None
    if config.model.contrastive_mask_policy == 'training_known':
        if config.data.train_view_manifest is not None:
            raise ValueError('Primary-grid evaluation cannot silently use a training subset')
        known_positive_source = dict(source=config.data.train_parquet_path,
            source_sha256=plan['source_data_sha256'][regime]['train'],
            cache=str(Path(config.data.tokenized_dataset_dir)/'train_examples'),
            cache_sha256=plan['tokenized_data_sha256'][regime]['train'],
            provenance=read_json(run/'known_positive_index.json'),
            provenance_sha256=sha256(run/'known_positive_index.json'))
    from importlib.metadata import version
    sources = sorted((ROOT/'RewardModel/reward_model').rglob('*.py'))
    sources += [Path(__file__), *(Path(__file__).with_name(name) for name in
        ('checkpoint_evaluation.py', 'control_plan.py', 'control_driver.py', 'retain_controlled.py', 'run.py'))]
    software = dict(source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sources},
                    packages={name:version(name) for name in ('torch','transformers','datasets','numpy')})
    return dict(format_version=1, job=metadata, primary_plan_sha256=sha256(directory/'plan.json'),
        selection_sha256=sha256(selection_path), training_complete_sha256=sha256(run/'controlled_training_complete.json'),
        retention_sha256=sha256(run/'retained/retention.json'), selected_learning_rate=selected,
        checkpoints=list(checkpoints.values()), rules=rules, data=data, known_positive_source=known_positive_source,
        seed_role='tuning_exposed' if metadata['phase'] == 'tuning' else 'confirmation', software=software,
        interpretation='Fixed primary held-out partitions. No test score selects a model or budget. Checkpoint rules share the B-selected LR. Other LR candidates contribute best-macro results only. Additional B allocations and seed aggregation remain separate.')


def prepare(identity):
    plan = build_evaluation_plan(WORK/'controlled', identity)
    if ROOT == Path('/home/atabey/Prot2Mol'):
        if __package__:
            from .prepare_random import archive_directory
        else:
            from prepare_random import archive_directory
        archive_directory(WORK/'postselection')
    path = WORK/'postselection'/identity/'plan.json'
    immutable_json(path, plan)
    return path


def verify_data(source):
    if (sha256(Path(source['source'])) != source['source_sha256'] or
        token_cache_hashes(Path(source['cache'])) != source['cache_sha256']):
        raise ValueError('Evaluation data differs from the frozen training plan')


def verify_evaluation(output, source, *, cpu_qualification=False):
    """Reuse only a complete native evaluation with unchanged source/payload."""
    if __package__:
        from .checkpoint_evaluation import read_native_predictions
    else:
        from checkpoint_evaluation import read_native_predictions
    result = read_json(output/'complete.json')
    if (result['source'] != source or not result['parameters_unchanged'] or result['optimizer_created'] or
        result['cpu_qualification'] != cpu_qualification or result['world_size'] != 1):
        raise ValueError('Fixed-weight evaluation provenance changed')
    if result['prefix'] != ('eval' if source['split'] == 'val' else 'test'):
        raise ValueError('Fixed-weight evaluation partition changed')
    manifest = Path(result['prediction_manifest'])
    expected = output/'prediction_exports/fixed_checkpoint'/f"{result['prefix']}_step00000000.json"
    if manifest != expected or sha256(manifest) != result['prediction_manifest_sha256']:
        raise ValueError('Fixed-weight evaluation manifest changed')
    _, record = read_native_predictions(manifest)
    if result['metrics'] != record['metrics'] or result['prediction_payload_sha256'] != record['predictions_sha256']:
        raise ValueError('Fixed-weight evaluation metrics changed')
    return result


def evaluate_worker(path, index, workers, *, cpu_qualification=False):
    import torch
    from datasets import load_from_disk
    from reward_model.model import load_reward_model
    from reward_model.training.config import load_reward_training_config
    from reward_model.training.known_positives import KnownPositiveIndex
    if __package__:
        from .checkpoint_evaluation import evaluate_fixed_model
    else:
        from checkpoint_evaluation import evaluate_fixed_model
    if not 1 <= workers <= 4 or not 0 <= index < workers:
        raise RuntimeError('Invalid fixed-model worker assignment')
    device = None if cpu_qualification else GPU_UUIDS[index]
    if cpu_qualification:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('ACCELERATE_USE_CPU') != 'true':
            raise RuntimeError('CPU qualification requires an explicitly CPU-only process')
    elif os.environ.get('CUDA_VISIBLE_DEVICES') != device:
        raise RuntimeError('Fixed-model worker must use its assigned authorized GPU UUID')
    torch.set_num_threads(4)
    path = Path(path)
    plan = read_json(path)
    if plan != build_evaluation_plan(WORK/'controlled', plan['job']['id']):
        raise ValueError('Frozen evaluation plan changed')
    config = load_reward_training_config(plan['job']['config_path'])
    datasets = {}
    for split, source in plan['data'].items():
        verify_data(source)
        datasets[split] = load_from_disk(source['cache'])
    known = None
    if plan['known_positive_source'] is not None:
        source = plan['known_positive_source']
        verify_data(source)
        known = KnownPositiveIndex.from_training_dataset(load_from_disk(source['cache']),
            active_threshold=config.model.contrastive_active_threshold)
        if known.provenance != source['provenance']:
            raise ValueError('Reconstructed training-positive index changed')
    for checkpoint in plan['checkpoints'][index::workers]:
        model = None
        for split in ('val', 'test'):
            output = path.parent/checkpoint['id']/split
            source = dict(plan_sha256=sha256(path), checkpoint=checkpoint, split=split,
                          gpu_uuid=device)
            if (output/'complete.json').exists():
                verify_evaluation(output, source, cpu_qualification=cpu_qualification)
                continue
            if output.exists():
                raise RuntimeError(f'Partial evaluation requires inspection: {output}')
            if model is None:
                model = load_reward_model(checkpoint['path'], device=torch.device('cpu'))
                verify_model_config(model.config.to_dict(), config.model, encoder_widths={
                    side:getattr(model, side+'_encoder').config.hidden_size
                    for side in ('protein', 'molecule')})
                if json.loads(json.dumps(model.config.to_dict())) != read_json(Path(checkpoint['path'])/'config.json'):
                    raise ValueError('Loaded model configuration differs from its saved checkpoint')
                model.to('cpu' if cpu_qualification else 'cuda')
            evaluate_fixed_model(model, datasets[split], config, output,
                prefix='eval' if split == 'val' else 'test', known_positive_index=known, source=source,
                cpu=cpu_qualification)
            verify_evaluation(output, source, cpu_qualification=cpu_qualification)
        del model
        torch.cuda.empty_cache()


def run_evaluation(identity, workers=4):
    import fcntl
    if ROOT != Path('/home/atabey/Prot2Mol') or not 1 <= workers <= 4:
        raise RuntimeError('Scientific evaluation is restricted to Atum and at most four authorized GPUs')
    with (WORK/'controlled/driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        busy = busy_authorized_devices(subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True))
        if busy:
            return dict(phase='waiting_for_authorized_gpus', live_processes=busy)
        path = prepare(identity)
        children, handles = [], []
        try:
            for index in range(workers):
                env = launch_environment(identity)
                env.update(CUDA_VISIBLE_DEVICES=GPU_UUIDS[index], WANDB_MODE='disabled')
                handle = (path.parent/f'worker{index}.log').open('a')
                handles.append(handle)
                command = [str(ROOT/'.venv/bin/python'), str(Path(__file__)), 'worker',
                           '--plan', str(path), '--index', str(index), '--workers', str(workers)]
                children.append(subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT))
            write_json(path.parent/'status.json', dict(phase='evaluating', driver_pid=os.getpid(),
                worker_pids=[p.pid for p in children], started_unix=time.time(), plan_sha256=sha256(path)))
            codes = [p.wait() for p in children]
            if any(codes):
                raise RuntimeError(f'Evaluation workers failed; inspect logs before retry: {codes}')
        finally:
            # A launch error must not orphan already started owned workers.
            for child in children:
                if child.poll() is None:
                    child.terminate()
                    child.wait()
            for handle in handles:
                handle.close()
        plan = read_json(path)
        evidence = {}
        for index in range(workers):
            for checkpoint in plan['checkpoints'][index::workers]:
                for split in ('val', 'test'):
                    output = path.parent/checkpoint['id']/split
                    source = dict(plan_sha256=sha256(path), checkpoint=checkpoint, split=split, gpu_uuid=GPU_UUIDS[index])
                    verify_evaluation(output, source)
                    evidence[str(output.relative_to(path.parent))] = sha256(output/'complete.json')
        result = dict(phase='fixed_weight_evaluation_complete', plan_sha256=sha256(path), evidence_sha256=evidence,
                      interpretation='Native predictions/losses complete; statistical comparisons and final report remain separate.')
        immutable_json(path.parent/'complete.json', result)
        write_json(path.parent/'status.json', result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for name in ('prepare', 'run'):
        sub = commands.add_parser(name)
        sub.add_argument('--job-id', required=True)
        if name == 'run':
            sub.add_argument('--workers', type=int, default=4)
    worker = commands.add_parser('worker')
    worker.add_argument('--plan', type=Path, required=True)
    worker.add_argument('--index', type=int, required=True)
    worker.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(prepare(args.job_id))
    elif args.action == 'run':
        print(json.dumps(run_evaluation(args.job_id, args.workers), indent=2))
    else:
        evaluate_worker(args.plan, args.index, args.workers)
