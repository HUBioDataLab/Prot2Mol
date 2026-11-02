"""E41: native evaluation of completed whole-unit curve fits.

The exact full-data endpoint is reused from the primary grid. This module adds
no fitting or tuning and does not launch automatically; all evaluation uses the
existing shared GPU lock and the authorized devices.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from . import control_plan, control_driver, learning_curve_plan, learning_curve_driver, postselection
    from .retain_controlled import selection_rules
else:
    from run import ROOT, WORK, sha256, write_json
    import control_plan, control_driver, learning_curve_plan, learning_curve_driver, postselection
    from retain_controlled import selection_rules

read = postselection.read_json
immutable = postselection.immutable_json


def validate_grid(curve, primary):
    fractions = learning_curve_plan.FRACTIONS
    if (curve['fractions'] != list(fractions) or curve['subset_seed'] != learning_curve_plan.SUBSET_SEED or
        primary['gpu_uuids'] != list(control_plan.GPU_UUIDS)):
        raise ValueError('Curve fractions, subset seed or authorized devices changed')
    full = next(a for a in primary['arms'] if a['name'] == 'full')
    if full['model'] or full['training']:
        raise ValueError('Curve endpoint must be the unmodified full reference')
    expected = {control_plan.job_id('unit_curve_'+learning_curve_plan.fraction_name(f), r, 42, lr)
        for f in fractions[:-1] for r in ('A', 'B') for lr in control_plan.LEARNING_RATE_MULTIPLIERS}
    if len(curve['tuning_jobs']) != 18 or {j['id'] for j in curve['tuning_jobs']} != expected:
        raise ValueError('Need exactly the eighteen registered curve tuning fits')
    for job in curve['tuning_jobs']:
        fraction = job['fraction']; name = learning_curve_plan.fraction_name(fraction); regime = job['regime']
        if (fraction == 1. or job['arm'] != 'unit_curve_'+name or job['seed'] != 42 or job['phase'] != 'tuning' or
            job['id'] != control_plan.job_id(job['arm'], regime, 42, job['learning_rate_multiplier'])):
            raise ValueError('Curve fit identity changed or duplicates the full endpoint')
        expected_config = control_plan.make_config(primary['bases'][regime], full, regime=regime,
            seed=42, multiplier=job['learning_rate_multiplier'], output=WORK/'learning_curves/runs'/job['id'],
            training_noise_rms=primary['training_noise_rms'])
        view = curve['views'][regime][name]
        expected_config['data'].update(train_view_manifest=view['path'], train_view_sha256=view['sha256'])
        if job['config'] != expected_config:
            raise ValueError('Curve changes more than its registered whole-unit training view')
        from reward_model.training.config import RewardModelConfig
        model = RewardModelConfig(**job['config']['model'])
        if model.contrastive_mask_policy != 'reference' or model.ranking_readout != 'ranking_head':
            raise ValueError('Curve must retain the reference mask and ranking readout')
    return curve['tuning_jobs']


def prepared():
    curve, primary = learning_curve_driver.verify_prepared()
    validate_grid(curve, primary)
    return curve, primary


def view_contract(curve, primary, regime, fraction):
    """Recheck selected units and subset-local labels without loading tokens."""
    from reward_model.training.train_views import check_whole_units, subset_pair_labels
    name = learning_curve_plan.fraction_name(fraction); item = curve['views'][regime][name]
    path = Path(item['path']); record = read(path)
    if (sha256(path) != item['sha256'] or record['format_version'] != 1 or
        record['regime'] != regime or record['fraction_requested'] != fraction or
        record['subset_seed'] != learning_curve_plan.SUBSET_SEED or
        record['label_policy'] != 'within_selected_training_pair_median' or record['activity_threshold'] != 6. or
        record['unit_definition'] != ('whole assay' if regime == 'A' else 'whole scaffold-assay component') or
        record['statistics'] != curve['statistics'][regime][name]):
        raise ValueError('Curve training-view identity or statistics changed')
    source = record['source']; base = primary['bases'][regime]['data']
    if (source['parquet_path'] != base['train_parquet_path'] or
        source['parquet_sha256'] != primary['source_data_sha256'][regime]['train'] or
        source['cache_dir'] != str(Path(base['tokenized_dataset_dir'])/'train_examples') or
        source['cache_files_sha256'] != primary['tokenized_data_sha256'][regime]['train']):
        raise ValueError('Curve view is not based on the original training partition')
    evidence = {str(path): sha256(path)}
    for key in ('indices', 'labels', 'source_metadata'):
        artifact = Path(record[key]['path'])
        if sha256(artifact) != record[key]['sha256']:
            raise ValueError('Curve view payload changed')
        evidence[str(artifact)] = sha256(artifact)
    metadata = pd.read_parquet(record['source_metadata']['path'])
    indices = np.load(record['indices']['path'], allow_pickle=False)
    labels = np.load(record['labels']['path'], allow_pickle=False)
    if len(metadata) != source['rows']:
        raise ValueError('Curve metadata does not cover its source')
    check_whole_units(metadata.unit.to_numpy(), indices)
    nested, order = learning_curve_plan.nested_unit_indices(metadata.unit.to_numpy())
    if not np.array_equal(indices, nested[fraction]) or not np.array_equal(labels, subset_pair_labels(metadata, indices)):
        raise ValueError('Curve differs from the frozen nested subset or subset-local labels')
    order_path = path.parent/'unit_order.npy'
    actual_order = np.load(order_path, allow_pickle=False)
    if (sha256(order_path) != record['unit_order_sha256'] or
        sha256(order_path) != curve['artifacts'][regime]['unit_order_sha256'] or
        not np.array_equal(actual_order.astype(str), order.astype(str))):
        raise ValueError('Curve unit permutation changed')
    evidence[str(order_path)] = sha256(order_path)
    if record['statistics']['rows'] != len(indices):
        raise ValueError('Realized training size changed')
    if fraction == 1. and (not np.array_equal(indices, np.arange(len(metadata))) or
        not np.array_equal(labels, metadata.binary_label.to_numpy())):
        raise ValueError('Full endpoint does not preserve the primary rows and labels')
    provenance = dict(manifest_path=str(path.resolve()), manifest_sha256=item['sha256'],
        view_fingerprint=item['sha256'], source_fingerprint=source['fingerprint'],
        label_policy=record['label_policy'], unit_definition=record['unit_definition'],
        statistics=record['statistics'], token_payloads_copied=False)
    return dict(provenance=provenance, input_sha256=evidence, source=source)


def training_evidence(metadata, primary_digest, expected_view=None):
    output = Path(metadata['output_dir']); runtime_path = output/'runtime_qualification.json'
    completion = read(output/'controlled_training_complete.json')
    required = {'experiment_summary.json', 'trainer_state.json', 'retained/retention.json',
                *(f'rank_{i}_complete.json' for i in range(4))}
    if not required <= set(completion['evidence_sha256']):
        raise ValueError('Training completion omits required native evidence')
    runtime = read(runtime_path)
    if (runtime['world_size'] != 4 or runtime['signature'] != control_plan.digest_json(
        dict(sources=runtime['sources'], plan_sha256=primary_digest))):
        raise ValueError('Curve training qualification differs from the primary source contract')
    evidence = {str(runtime_path): sha256(runtime_path)}
    for name, digest in runtime['sources'].items():
        path = ROOT/name
        if not path.resolve().is_relative_to(ROOT.resolve()) or sha256(path) != digest:
            raise ValueError('Qualified training implementation changed')
        evidence[str(path)] = digest
    summary_path = output/'experiment_summary.json'; summary = read(summary_path)
    if summary['config_sha256'] != metadata['config_sha256'] or summary['test_evaluated'] or summary['test_metrics']:
        raise ValueError('Training summary config changed or test data were evaluated')
    state = read(output/'trainer_state.json')
    history = [r for r in state['log_history'] if 'eval_macro_spearman' in r and 'eval_loss' in r]
    if (state['epoch'] != 15 or [r['epoch'] for r in history] != list(range(1, 16)) or
        state['global_step'] != history[-1]['step'] or
        any(a['step'] >= b['step'] for a, b in zip(history, history[1:]))):
        raise ValueError('Every tuning, confirmation and shared endpoint fit must finish all fifteen epochs')
    for rank in range(4):
        path = output/f'rank_{rank}_complete.json'; done = read(path)
        if (done['rank'], done['world_size'], done['config_sha256']) != (rank, 4, metadata['config_sha256']):
            raise ValueError('All four ranks must finish the registered training configuration')
        evidence[str(path)] = sha256(path)
    if expected_view is not None and (summary['training_view'] != expected_view['provenance'] or
        summary['train_examples'] != expected_view['provenance']['statistics']['rows']):
        raise ValueError('Completed training did not use the registered subset and labels')
    if expected_view is None and summary.get('training_view') is not None:
        raise ValueError('Primary full endpoint unexpectedly used a training subset')
    for path in (summary_path, output/'trainer_state.json', output/'controlled_training_complete.json',
                 Path(metadata['config_path']), output/'retained/retention.json'):
        evidence[str(path)] = sha256(path)
    return runtime, evidence


def shared_endpoint(primary):
    """Read and verify all ten existing fits; never materialize confirmations."""
    directory = WORK/'controlled'; digest = sha256(directory/'plan.json')
    decision, path = postselection.verify_selection(directory, 'full', primary)
    identities = [j['id'] for j in primary['tuning_jobs'] if j['arm'] == 'full']
    identities += [control_plan.job_id('full', r, s, decision['learning_rate_multiplier'])
                   for s in (43, 44) for r in ('A', 'B')]
    if len(identities) != 10 or len(set(identities)) != 10:
        raise ValueError('Need six tuning and four shared endpoint confirmations')
    evidence = {str(path): sha256(path)}; qualification = None
    for identity in identities:
        job = postselection.registered_job(directory, identity, primary)
        if job['phase'] == 'confirmation' and job['selection_sha256'] != sha256(path):
            raise ValueError('Shared endpoint confirmation selection changed')
        runtime, files = training_evidence(job, digest)
        if qualification is None:
            qualification = runtime
        elif runtime != qualification:
            raise ValueError('Shared endpoint training implementations differ')
        evidence.update(files)
    return dict(job_ids=identities, selection_sha256=sha256(path), input_sha256=evidence,
        qualification=qualification, refitted=False)


def verify_selection(curve, primary, fraction, views, qualification):
    directory = WORK/'learning_curves'; digest = sha256(directory/'plan.json')
    candidates = []; sources = []; evidence = {}
    for job in (j for j in curve['tuning_jobs'] if j['fraction'] == fraction):
        metadata = postselection.registered_job(directory, job['id'], curve)
        if metadata['fraction'] != fraction:
            raise ValueError('Tuning fit fraction changed')
        runtime, files = training_evidence(metadata, sha256(WORK/'controlled/plan.json'), views[job['regime']])
        if runtime != qualification:
            raise ValueError('Curve and full endpoint training qualifications differ')
        evidence.update(files)
        candidate, source = control_plan.validation_candidate(metadata); sources.append(source)
        if job['regime'] == 'B':
            candidates.append(candidate)
    if len(sources) != 6 or len(candidates) != 3:
        raise ValueError('Need all six fraction-specific tuning fits')
    expected = dict(fraction=fraction, plan_sha256=digest,
        **control_plan.select_learning_rate(candidates), evidence=sources)
    path = directory/'selections'/f'{learning_curve_plan.fraction_name(fraction)}.json'
    if read(path) != expected:
        raise ValueError('Curve LR choice differs from validation-only reconstruction')
    evidence[str(path)] = sha256(path)
    return expected, path, evidence


def retained_checkpoints(metadata, selected):
    from reward_model.training.config import load_reward_training_config
    run = Path(metadata['output_dir']); state = read(run/'trainer_state.json')
    history = [r for r in state['log_history'] if 'eval_macro_spearman' in r and 'eval_loss' in r]
    if state['epoch'] != 15 or [r['epoch'] for r in history] != list(range(1, 16)):
        raise ValueError('Curve evaluation requires the complete fifteen-epoch history')
    retained = read(run/'retained/retention.json')
    if retained['selections'] != selection_rules(history):
        raise ValueError('Curve checkpoint rules differ from validation history')
    config = load_reward_training_config(metadata['config_path']); model_config = read(run/'config.json')
    postselection.verify_model_config(model_config, config.model)
    preserved = {x['path']: x['sha256'] for x in retained['preserved_weights']}
    checkpoints = {}; rules = {}
    def register(path):
        relative = str((path/'pytorch_model.bin').relative_to(run))
        if relative not in preserved or sha256(path/'pytorch_model.bin') != preserved[relative]:
            raise ValueError('Retained curve checkpoint is absent or changed')
        if read(path/'config.json') != model_config:
            raise ValueError('Retained curve model configuration changed')
        identity = str(path.relative_to(run)).replace('/', '__')
        checkpoints.setdefault(identity, dict(id=identity, path=str(path), weights_sha256=preserved[relative],
            model_config_sha256=sha256(path/'config.json')))
        return identity
    for name in (list(retained['selections']) if selected else ['best_macro']):
        rule = retained['selections'][name]
        rules[name] = dict(**rule, checkpoint=register(run/f"checkpoint-{rule['step']}"))
    if selected:
        rules['average_from_loss_minimum'] = dict(**retained['averaging'],
            checkpoint=register(run/'retained/average_from_loss_minimum'))
    return list(checkpoints.values()), rules


def build_evaluation_plan(identity):
    curve, primary = prepared(); directory = WORK/'learning_curves'
    job = postselection.registered_job(directory, identity, curve)
    fraction = job['fraction']
    if fraction not in learning_curve_plan.FRACTIONS[:-1] or job['id'] != control_plan.job_id(
        'unit_curve_'+learning_curve_plan.fraction_name(fraction), job['regime'], job['seed'], job['learning_rate_multiplier']):
        raise ValueError('Only registered smaller-fraction jobs use this evaluator')
    endpoint = shared_endpoint(primary)
    views = {r: view_contract(curve, primary, r, fraction) for r in ('A', 'B')}
    full_views = {r: view_contract(curve, primary, r, 1.) for r in ('A', 'B')}
    decision, selection_path, hashes = verify_selection(curve, primary, fraction, views, endpoint['qualification'])
    selected = job['learning_rate_multiplier'] == decision['learning_rate_multiplier']
    if job['phase'] == 'confirmation' and (not selected or job['selection_sha256'] != sha256(selection_path)):
        raise ValueError('Curve confirmation must use its frozen B-selected LR')
    runtime, files = training_evidence(job, sha256(WORK/'controlled/plan.json'), views[job['regime']])
    if runtime != endpoint['qualification']:
        raise ValueError('Curve confirmation training implementation changed')
    hashes.update(files); hashes.update(endpoint['input_sha256'])
    for view in (*views.values(), *full_views.values()):
        hashes.update(view['input_sha256'])
    hashes[str(WORK/'controlled/plan.json')] = sha256(WORK/'controlled/plan.json')
    hashes[str(directory/'plan.json')] = sha256(directory/'plan.json')
    checkpoints, rules = retained_checkpoints(job, selected)
    regime = job['regime']; data = {split: dict(source=primary['bases'][regime]['data'][f'{split}_parquet_path'],
        source_sha256=primary['source_data_sha256'][regime][split],
        cache=str(Path(primary['bases'][regime]['data']['tokenized_dataset_dir'])/f'{split}_examples'),
        cache_sha256=primary['tokenized_data_sha256'][regime][split]) for split in ('val', 'test')}
    from importlib.metadata import version
    sources = sorted((ROOT/'RewardModel/reward_model').rglob('*.py'))
    sources += [Path(__file__), *(Path(__file__).with_name(n) for n in ('postselection.py', 'checkpoint_evaluation.py',
        'learning_curve_plan.py', 'learning_curve_driver.py', 'control_plan.py', 'control_driver.py', 'retain_controlled.py', 'run.py'))]
    software = dict(source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sources},
        packages={n: version(n) for n in ('torch', 'transformers', 'datasets', 'numpy', 'pandas')})
    for name, digest in hashes.items():
        if sha256(Path(name)) != digest:
            raise ValueError('Curve evaluation source changed during planning')
    return dict(format_version=1, job=job, curve_plan_sha256=sha256(directory/'plan.json'),
        primary_plan_sha256=sha256(WORK/'controlled/plan.json'), selection_sha256=sha256(selection_path),
        selected_learning_rate=selected, seed_role='tuning_exposed' if job['seed'] == 42 else 'confirmation',
        shared_full_endpoint=endpoint, training_view=views[regime], input_sha256=hashes,
        checkpoints=checkpoints, rules=rules, data=data, software=software, test_used_for_selection=False,
        new_fits_created=0, interpretation='E41 fixed held-out partitions, subset-local training labels, fifteen epochs. Each fraction retains its own frozen B-validation LR. Selected LR gets all checkpoint rules; other seed42 LRs get best-macro only. Full endpoint is reused. No fitting, test-driven selection, subset redraw or automatic launch.')


def prepare(identity):
    plan = build_evaluation_plan(identity)
    if ROOT == Path('/home/atabey/Prot2Mol'):
        if __package__:
            from .prepare_random import archive_directory
        else:
            from prepare_random import archive_directory
        archive_directory(WORK/'learning_curve_evaluation')
    path = WORK/'learning_curve_evaluation'/identity/'plan.json'
    immutable(path, plan)
    return path


def evaluate_worker(path, index, workers, *, cpu_qualification=False):
    import torch
    from datasets import load_from_disk
    from reward_model.model import load_reward_model
    from reward_model.training.config import load_reward_training_config
    if __package__:
        from .checkpoint_evaluation import evaluate_fixed_model
    else:
        from checkpoint_evaluation import evaluate_fixed_model
    if not 1 <= workers <= 4 or not 0 <= index < workers:
        raise RuntimeError('Invalid curve evaluation worker assignment')
    device = None if cpu_qualification else control_plan.GPU_UUIDS[index]
    if cpu_qualification:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('ACCELERATE_USE_CPU') != 'true':
            raise RuntimeError('CPU qualification requires an explicitly CPU-only process')
    elif os.environ.get('CUDA_VISIBLE_DEVICES') != device:
        raise RuntimeError('Curve worker must use its assigned authorized GPU UUID')
    torch.set_num_threads(2); path = Path(path); plan = read(path)
    if plan != build_evaluation_plan(plan['job']['id']):
        raise ValueError('Frozen curve evaluation plan changed')
    config = load_reward_training_config(plan['job']['config_path'])
    if config.model.contrastive_mask_policy != 'reference':
        raise ValueError('Curve evaluation retains the registered reference mask')
    datasets = {}
    for split, source in plan['data'].items():
        postselection.verify_data(source); datasets[split] = load_from_disk(source['cache'])
    for checkpoint in plan['checkpoints'][index::workers]:
        model = None
        for split in ('val', 'test'):
            output = path.parent/checkpoint['id']/split
            source = dict(plan_sha256=sha256(path), checkpoint=checkpoint, split=split, gpu_uuid=device)
            if (output/'complete.json').exists():
                postselection.verify_evaluation(output, source, cpu_qualification=cpu_qualification)
                continue
            if output.exists():
                raise RuntimeError(f'Partial curve evaluation requires inspection: {output}')
            if model is None:
                model = load_reward_model(checkpoint['path'], device=torch.device('cpu'))
                postselection.verify_model_config(model.config.to_dict(), config.model,
                    encoder_widths={s: getattr(model, s+'_encoder').config.hidden_size for s in ('protein', 'molecule')})
                if json.loads(json.dumps(model.config.to_dict())) != read(Path(checkpoint['path'])/'config.json'):
                    raise ValueError('Loaded curve model differs from its checkpoint')
                model.to('cpu' if cpu_qualification else 'cuda')
            evaluate_fixed_model(model, datasets[split], config, output, prefix='eval' if split == 'val' else 'test',
                known_positive_index=None, source=source, cpu=cpu_qualification)
            postselection.verify_evaluation(output, source, cpu_qualification=cpu_qualification)
        del model
        torch.cuda.empty_cache()


def verify_complete(directory, *, cpu_qualification=False):
    directory = Path(directory); path = directory/'plan.json'; plan = read(path); complete = read(directory/'complete.json')
    if (complete['phase'] != 'curve_fixed_weight_evaluation_complete' or complete['plan_sha256'] != sha256(path) or
        plan != build_evaluation_plan(plan['job']['id'])):
        raise ValueError('Curve evaluation completion or selection changed')
    expected = {f"{c['id']}/{s}" for c in plan['checkpoints'] for s in ('val', 'test')}
    if set(complete['evidence_sha256']) != expected:
        raise ValueError('Curve completion omits or adds checkpoint partitions')
    for name, digest in complete['evidence_sha256'].items():
        if sha256(directory/name/'complete.json') != digest:
            raise ValueError('Curve evaluation receipt changed')
    results = {}
    for checkpoint in plan['checkpoints']:
        results[checkpoint['id']] = {}
        for split in ('val', 'test'):
            output = directory/checkpoint['id']/split; receipt = read(output/'complete.json'); device = receipt['source']['gpu_uuid']
            if (cpu_qualification and device is not None) or (not cpu_qualification and device not in control_plan.GPU_UUIDS):
                raise ValueError('Curve evaluation has unexpected device provenance')
            source = dict(plan_sha256=sha256(path), checkpoint=checkpoint, split=split, gpu_uuid=device)
            results[checkpoint['id']][split] = postselection.verify_evaluation(output, source, cpu_qualification=cpu_qualification)
    return plan, results


def run_evaluation(identity, workers=4):
    import fcntl
    if ROOT != Path('/home/atabey/Prot2Mol') or not 1 <= workers <= 4:
        raise RuntimeError('Scientific curve evaluation requires Atum and at most four authorized GPUs')
    with (WORK/'controlled/driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        busy = control_driver.busy_authorized_devices(subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True))
        if busy:
            return dict(phase='waiting_for_authorized_gpus', live_processes=busy)
        path = prepare(identity); children = []; handles = []
        try:
            for index in range(workers):
                env = control_driver.launch_environment(identity)
                env.update(CUDA_VISIBLE_DEVICES=control_plan.GPU_UUIDS[index], WANDB_MODE='disabled')
                handle = (path.parent/f'worker{index}.log').open('a'); handles.append(handle)
                command = [str(ROOT/'.venv/bin/python'), str(Path(__file__)), 'worker', '--plan', str(path),
                    '--index', str(index), '--workers', str(workers)]
                children.append(subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT))
            write_json(path.parent/'status.json', dict(phase='evaluating', driver_pid=os.getpid(),
                worker_pids=[p.pid for p in children], started_unix=time.time(), plan_sha256=sha256(path)))
            codes = [p.wait() for p in children]
            if any(codes):
                raise RuntimeError(f'Curve evaluation workers failed; inspect logs before retry: {codes}')
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate(); child.wait()
            for handle in handles:
                handle.close()
        plan = read(path); evidence = {}
        for index in range(workers):
            for checkpoint in plan['checkpoints'][index::workers]:
                for split in ('val', 'test'):
                    output = path.parent/checkpoint['id']/split
                    source = dict(plan_sha256=sha256(path), checkpoint=checkpoint, split=split, gpu_uuid=control_plan.GPU_UUIDS[index])
                    postselection.verify_evaluation(output, source)
                    evidence[str(output.relative_to(path.parent))] = sha256(output/'complete.json')
        result = dict(phase='curve_fixed_weight_evaluation_complete', plan_sha256=sha256(path),
            evidence_sha256=evidence, interpretation='Native curve predictions and losses complete; statistical learning-curve reporting remains separate.')
        immutable(path.parent/'complete.json', result); verify_complete(path.parent)
        write_json(path.parent/'status.json', result)
        return result


def preflight():
    curve, primary = prepared(); directory = WORK/'learning_curves'
    return dict(primary_plan_sha256=sha256(WORK/'controlled/plan.json'), curve_plan_sha256=sha256(directory/'plan.json'),
        registered_tuning_jobs=len(curve['tuning_jobs']), new_fits_created=0, scientific_evaluation_created=False,
        training_completion_markers={j['id']: (directory/'runs'/j['id']/'controlled_training_complete.json').exists()
            for j in curve['tuning_jobs']},
        selection_markers={str(f): (directory/'selections'/f'{learning_curve_plan.fraction_name(f)}.json').exists()
            for f in learning_curve_plan.FRACTIONS[:-1]},
        interpretation='Read-only preparation inventory. Marker presence is not verified completion. Each evaluation requires all ten full-endpoint fits, six fraction-specific tuning fits, frozen B selection, exact training view and retained weights.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('preflight')
    for name in ('prepare', 'run'):
        sub = commands.add_parser(name); sub.add_argument('--job-id', required=True)
        if name == 'run':
            sub.add_argument('--workers', type=int, default=4)
    worker = commands.add_parser('worker'); worker.add_argument('--plan', type=Path, required=True)
    worker.add_argument('--index', type=int, required=True); worker.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()
    if args.action == 'preflight':
        print(json.dumps(preflight(), indent=2))
    elif args.action == 'prepare':
        print(prepare(args.job_id))
    elif args.action == 'run':
        print(json.dumps(run_evaluation(args.job_id, args.workers), indent=2))
    else:
        evaluate_worker(args.plan, args.index, args.workers)
