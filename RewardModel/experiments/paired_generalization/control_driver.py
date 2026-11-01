"""Run one ready controlled job, without replacing existing or live work.

The active goal can call this again after completion. This driver deliberately
does not infer experiment completion from training completion: test scoring,
mechanistic probes and the remaining MD still have to be performed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import yaml

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .control_plan import GPU_UUIDS, confirm, digest_json, _base_config, token_cache_hashes
    from .retain_controlled import retain
else:
    from run import ROOT, WORK, sha256, write_json
    from control_plan import GPU_UUIDS, confirm, digest_json, _base_config, token_cache_hashes
    from retain_controlled import retain

BASELINE_PATHS = (
    'linear/{regime}/molecule_only_ridge', 'linear/{regime}/protein_only_bce',
    'nearest_scaffold_lookup/{regime}', 'independent_task_ridge/{regime}',
    'two_step_ridge/{regime}', 'linear_matrix_filter/{regime}',
    'kronecker_ridge/{regime}', 'protein_shuffled_kronecker/{regime}',
    'bilinear_completion_3000iter/{regime}', 'lambdamart/{regime}',
)


def baseline_gate(work, plan):
    missing, evidence = [], []
    for regime in ('A', 'B'):
        for template in BASELINE_PATHS:
            path = Path(work)/'tier2'/template.format(regime=regime)/'complete.json'
            if not path.exists():
                missing.append(str(path))
                continue
            record = json.loads(path.read_text())
            if record['regime'] != regime or record['data_sha256'] != plan['source_data_sha256'][regime]:
                raise ValueError(f'Baseline data provenance changed: {path}')
            if set(record['metrics']) != {'val', 'test'}:
                raise ValueError(f'Baseline final evaluations are incomplete: {path}')
            evidence.append(dict(path=str(path), sha256=sha256(path)))
    return dict(ready=not missing, missing=missing, evidence=evidence)


def busy_authorized_devices(text):
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        device, pid = [part.strip() for part in line.split(',', 1)]
        if device in GPU_UUIDS:
            records.append(dict(gpu_uuid=device, pid=int(pid)))
    return records


def verify_training_data(plan, regime):
    data = plan['bases'][regime]['data']
    for split in ('train', 'val'):
        if sha256(Path(data[f'{split}_parquet_path'])) != plan['source_data_sha256'][regime][split]:
            raise ValueError(f'{regime} {split} source data changed after plan preparation')
        directory = Path(data['tokenized_dataset_dir'])/f'{split}_examples'
        if token_cache_hashes(directory) != plan['tokenized_data_sha256'][regime][split]:
            raise ValueError(f'{regime} {split} token cache changed after plan preparation')


def completed_job(metadata):
    output = Path(metadata['output_dir'])
    path = output/'controlled_training_complete.json'
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if record['config_sha256'] != metadata['config_sha256'] or record['plan_sha256'] != metadata['plan_sha256']:
        raise ValueError('Completed control job has different configuration/plan')
    for name, digest in record['evidence_sha256'].items():
        if sha256(output/name) != digest:
            raise ValueError('Completed control evidence changed')
    return True


def next_job(directory, plan):
    directory = Path(directory)
    plan_digest = sha256(directory/'plan.json')
    for arm in plan['arms']:
        for job in (j for j in plan['tuning_jobs'] if j['arm'] == arm['name']):
            metadata = json.loads((directory/'jobs'/f"{job['id']}.json").read_text())
            if metadata['plan_sha256'] != plan_digest:
                raise ValueError('Job belongs to another plan')
            if not completed_job(metadata):
                return metadata
        # This checks all A/B tuning completions and only uses B validation to
        # materialize the shared seed-43/44 LR, with immutable selection evidence.
        confirmation = confirm(arm['name'])
        for metadata in confirmation['confirmation_jobs']:
            if not completed_job(metadata):
                return metadata
    return None


def launch_environment(job_id):
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=','.join(GPU_UUIDS), HF_HOME=str(ROOT/'.hf_cache'),
        TORCH_HOME=str(ROOT/'.cache/torch'), XDG_CACHE_HOME=str(ROOT/'.cache'),
        TRITON_CACHE_DIR=str(ROOT/'.cache/triton'), CUDA_CACHE_PATH=str(ROOT/'.cache/cuda'),
        TMPDIR=str(ROOT/'.cache/paired_generalization_tmp'),
        WANDB_DIR=str(WORK), WANDB_CACHE_DIR=str(ROOT/'.wandb_cache'),
        WANDB_CONFIG_DIR=str(ROOT/'.cache/wandb_config'), WANDB_DATA_DIR=str(ROOT/'.cache/wandb_data'),
        WANDB_PROJECT='Prot2Mol_RewardModel', WANDB_ENTITY='hubiodatascilab',
        WANDB_RUN_GROUP='paired_generalization_controlled', WANDB_NAME=job_id,
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1',
    )
    # A failed run must never silently attach to an unrelated existing W&B ID.
    env.pop('WANDB_RUN_ID', None)
    env.pop('WANDB_RESUME', None)
    return env


def _run_process(command, output, phase, job_id):
    with Path(output).open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=launch_environment(job_id),
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        write_json(WORK/'controlled/driver_status.json', dict(phase=phase, job_id=job_id,
            pid=process.pid, driver_pid=os.getpid(), started_unix=time.time(), command=command,
            gpu_uuids=list(GPU_UUIDS)))
        code = process.wait()
    if code:
        write_json(WORK/'controlled/driver_status.json', dict(phase=f'{phase}_failed', job_id=job_id, returncode=code))
        raise RuntimeError(f'{phase} failed; inspect logs/checkpoints before any restart')


def qualify_runtime(plan):
    """Once per exact source/plan, qualify all settings and the real 4-GPU Trainer."""
    source_paths = sorted((ROOT/'RewardModel/reward_model').rglob('*.py'))
    source = Path(__file__).resolve()
    source_paths += [source, source.with_name('control_plan.py'),
                     source.with_name('qualify_controls.py'), source.with_name('run.py'),
                     source.with_name('retain_controlled.py'), source.with_name('realization_protocol.py'),
                     source.with_name('realization_predictions.py'), source.with_name('export_predictions.py')]
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    signature = digest_json(dict(sources=sources, plan_sha256=sha256(WORK/'controlled/plan.json')))
    directory = WORK/'controlled/qualifications'/signature[:16]
    complete = directory/'complete.json'
    if complete.exists():
        record = json.loads(complete.read_text())
        if record['signature'] != signature:
            raise ValueError('Runtime qualification belongs to different source/plan')
        for path, digest in record['evidence_sha256'].items():
            if sha256(directory/path) != digest:
                raise ValueError('Runtime qualification evidence changed')
        return record
    if directory.exists():
        raise RuntimeError(f'Incomplete qualification requires inspection: {directory}')
    directory.mkdir(parents=True)
    command_prefix = [str(ROOT/'.venv/bin/python'), '-m', 'torch.distributed.run',
                      '--standalone', '--nproc_per_node=4']
    cases_path = directory/'control_grid.json'
    _run_process(command_prefix + ['RewardModel/experiments/paired_generalization/qualify_controls.py',
        '--config', str(ROOT/'RewardModel/configs/reward_train_paired_generalization_baseline.yaml'),
        '--control-grid', '--training-noise-rms', str(plan['training_noise_rms']['B']), '--output', str(cases_path)],
        directory/'control_grid.log', 'qualifying_controls', 'control_grid_runtime')
    cases = json.loads(cases_path.read_text())
    if cases['world_size'] != 4 or {case['case'] for case in cases['cases']} != {arm['name'] for arm in plan['arms']}:
        raise ValueError('Runtime qualification did not cover every planned setting on four GPUs')
    if not cases['real_config'] or cases['visible_devices'].split(',') != list(GPU_UUIDS):
        raise ValueError('Runtime qualification did not use the requested real encoders/devices')
    smoke = _base_config(WORK/'smoke_config.yaml')
    output = directory/'trainer'
    smoke['data']['sample_all_assays'] = True
    smoke['training'].update(output_dir=str(output), num_train_epochs=2, max_steps=None,
        eval_steps=None, per_device_train_batch_size=12, gradient_accumulation_steps=4,
        export_evaluation_predictions=True, evaluate_test=False, fixed_train_eval_assays=16,
        save_total_limit=3, report_to=[], protein_shuffle_sensitivity=False,
        metric_for_best_model='eval_macro_spearman', greater_is_better=True)
    config_path = directory/'trainer.yaml'
    config_path.write_text(yaml.safe_dump(smoke, sort_keys=True))
    write_json(output/'controlled_run.json', dict(format_version=1, allow_checkpoint_retention=True,
        output_dir=str(output), config_sha256=sha256(config_path), purpose='runtime qualification only'))
    _run_process(command_prefix + ['RewardModel/experiments/paired_generalization/run.py', 'train',
        '--config', str(config_path)], directory/'trainer.log', 'qualifying_trainer', 'trainer_runtime')
    retain(output, prune=True)
    summary = json.loads((output/'experiment_summary.json').read_text())
    if summary['test_evaluated'] or summary['test_metrics']:
        raise ValueError('Tuning qualification unexpectedly scored test examples')
    paths = [cases_path, output/'experiment_summary.json', output/'retained/retention.json',
             *(output/f'rank_{rank}_complete.json' for rank in range(4))]
    record = dict(signature=signature, sources=sources, world_size=4, purpose='runtime only; synthetic/smoke inputs',
                  evidence_sha256={str(path.relative_to(directory)): sha256(path) for path in paths})
    write_json(complete, record)
    return record


def run_next():
    import fcntl
    if ROOT != Path('/home/atabey/Prot2Mol'):
        raise RuntimeError('Controlled training is restricted to Atum Prot2Mol')
    directory = WORK/'controlled'
    lock = (directory/'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((directory/'plan.json').read_text())
    ready = json.loads((directory/'preparation_complete.json').read_text())
    if sha256(directory/'plan.json') != ready['plan_sha256']:
        raise ValueError('Prepared plan changed')
    if plan['gpu_uuids'] != list(GPU_UUIDS) or plan['spec_sha256'] != sha256(ROOT/'Paired input ranking generalization.md'):
        raise ValueError('Specification or authorized GPU set changed')
    gate = baseline_gate(WORK, plan)
    if not gate['ready']:
        return dict(phase='waiting_for_cheap_baselines', missing=gate['missing'])
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    busy = busy_authorized_devices(processes)
    if busy:
        return dict(phase='waiting_for_authorized_gpus', live_processes=busy)
    free = shutil.disk_usage(directory).free
    if free < 30 * 1024**3:
        return dict(phase='waiting_for_storage', bytes_free=free, minimum_bytes=30*1024**3)
    job = next_job(directory, plan)
    if job is None:
        return dict(phase='control_grid_training_complete_other_experiments_pending',
                    remaining='Postselection test scoring, whole-unit/leak designs, additional realization assessment, Tier 4 and method catalog')
    if sha256(Path(job['config_path'])) != job['config_sha256']:
        raise ValueError('Prepared configuration changed')
    output = Path(job['output_dir'])
    if output.exists():
        raise RuntimeError(f'Incomplete or existing run requires inspection; refusing automatic relaunch: {output}')
    verify_training_data(plan, job['regime'])
    if job['regime'] == 'B':
        if __package__:
            from .realization_protocol import load_contract
        else:
            from realization_protocol import load_contract
        # The assignments and row views must be frozen before scientific fits.
        load_contract()
    qualification = qualify_runtime(plan)
    busy = busy_authorized_devices(subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True))
    if busy:
        return dict(phase='waiting_for_authorized_gpus', live_processes=busy)
    output.mkdir(parents=True)
    write_json(output/'controlled_run.json', dict(format_version=1, allow_checkpoint_retention=True, **job))
    write_json(output/'baseline_gate.json', gate)
    write_json(output/'runtime_qualification.json', qualification)
    command = [str(ROOT/'.venv/bin/python'), '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=4', 'RewardModel/experiments/paired_generalization/run.py',
               'train', '--config', job['config_path']]
    _run_process(command, output/'train.log', 'training', job['id'])
    if job['regime'] == 'B':
        command = [str(ROOT/'.venv/bin/python'),
                   'RewardModel/experiments/paired_generalization/realization_predictions.py',
                   '--run', str(output), '--workers', '4']
        _run_process(command, output/'realization_exports.log', 'exporting_allocation_predictions', job['id'])
    result = retain(output, prune=True)
    evidence = ['experiment_summary.json', 'trainer_state.json', 'retained/retention.json',
                *(f'rank_{rank}_complete.json' for rank in range(4))]
    done = dict(config_sha256=job['config_sha256'], plan_sha256=job['plan_sha256'],
                job_id=job['id'], phase='training_complete_test_not_scored',
                selections=result['selections'], finished_unix=time.time(),
                evidence_sha256={name: sha256(output/name) for name in evidence})
    write_json(output/'controlled_training_complete.json', done)
    return done


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = run_next()
    write_json(WORK/'controlled/driver_status.json', result)
    print(json.dumps(result, indent=2))
