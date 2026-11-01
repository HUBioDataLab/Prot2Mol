"""Export every B checkpoint, then select using each allocation's validation.

No test metrics are computed here. The original training selector is unchanged.
Four independent single-GPU workers share checkpoints, not training updates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256
    from .control_plan import GPU_UUIDS, _write_immutable, token_cache_hashes
    from .realization_protocol import IDENTITY, LABELS, align_rows, load_contract
    from .export_predictions import score_dataset
    from .baseline_common import RankingSelection
else:
    from run import ROOT, WORK, sha256
    from control_plan import GPU_UUIDS, _write_immutable, token_cache_hashes
    from realization_protocol import IDENTITY, LABELS, align_rows, load_contract
    from export_predictions import score_dataset
    from baseline_common import RankingSelection


def immutable_json(path, record):
    _write_immutable(path, json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n')


def destination(run):
    return WORK/'realization_predictions'/Path(run).name


def validate_run_registration(run, launch):
    """Allow registered B full-data or whole-component-curve fits only."""
    run = Path(run).resolve()
    if run==(WORK/'baseline_scaffold_seed42_15ep').resolve():
        if launch['config']['data'].get('train_view_manifest'):
            raise ValueError('The protected reference cannot acquire a training subset')
        return dict(family='protected_reference')
    for family in ('controlled','learning_curves'):
        directory = WORK/family
        if run.parent!=(directory/'runs').resolve():
            continue
        path = directory/'jobs'/f'{run.name}.json'
        metadata = json.loads(path.read_text())
        plan_path = directory/'plan.json'
        if (metadata['id']!=run.name or metadata['regime']!='B' or
            Path(metadata['output_dir']).resolve()!=run or
            metadata['plan_sha256']!=sha256(plan_path) or
            metadata['config_sha256']!=launch['config_sha256'] or
            sha256(Path(metadata['config_path']))!=metadata['config_sha256']):
            raise ValueError('Allocation export has mismatched managed-run registration')
        data = launch['config']['data']
        if family=='learning_curves':
            plan = json.loads(plan_path.read_text())
            if plan['primary_plan_sha256']!=sha256(WORK/'controlled/plan.json'):
                raise ValueError('Curve parent differs from the allocation parent')
            view = plan['views']['B'][f"fraction{int(metadata['fraction']*100):03d}"]
            if (Path(data['train_view_manifest']).resolve()!=Path(view['path']).resolve() or
                data['train_view_sha256']!=view['sha256'] or sha256(Path(view['path']))!=view['sha256']):
                raise ValueError('Curve training subset differs from its frozen view')
        elif data.get('train_view_manifest'):
            raise ValueError('The primary full-data grid cannot acquire a training subset')
        return dict(family=family,job_metadata_sha256=sha256(path),plan_sha256=sha256(plan_path))
    raise ValueError('Run must be the protected reference or a registered B controlled/curve run')


def prepare_jobs(run):
    run = Path(run).resolve()
    launch = json.loads((run/'launch_provenance.json').read_text())
    registration = validate_run_registration(run,launch)
    contract, _, _ = load_contract()
    plan = json.loads(Path(contract['control_plan_path']).read_text())
    if launch['world_size'] != 4:
        raise ValueError('Scientific run must have four completed ranks')
    for rank in range(4):
        done = json.loads((run/f'rank_{rank}_complete.json').read_text())
        if done['rank'] != rank or done['world_size'] != 4 or done.get('config_sha256', launch['config_sha256']) != launch['config_sha256']:
            raise ValueError('Training rank completion mismatch')
    def resolve_data(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (ROOT/'RewardModel/configs'/path).resolve()
    for key in ('train_parquet_path', 'val_parquet_path', 'test_parquet_path', 'tokenized_dataset_dir'):
        if resolve_data(launch['config']['data'][key]) != resolve_data(plan['bases']['B']['data'][key]):
            raise ValueError('Run does not use the frozen B source/cache partitions')
    for split in ('val','test'):
        cache = Path(contract['tokenized_dataset_dir'])/f'{split}_examples'
        if token_cache_hashes(cache) != plan['tokenized_data_sha256']['B'][split]:
            raise ValueError('Held-out token cache changed before inference')
    states = sorted(run.glob('checkpoint-*/trainer_state.json'), key=lambda p: int(p.parent.name.split('-')[1]))
    if len(states) != 15 or float(launch['config']['training']['num_train_epochs']) != 15:
        raise ValueError('All 15 scientific epoch checkpoints are required')
    jobs, sources = [], []
    for epoch, path in enumerate(states, 1):
        state = json.loads(path.read_text())
        step = int(path.parent.name.split('-')[1])
        if not math.isclose(state['epoch'], epoch, abs_tol=1e-6) or state['global_step'] != step:
            raise ValueError('Epoch checkpoint is incomplete or out of order')
        if (path.parent/'config.json').is_symlink() or path.parent.is_symlink():
            raise ValueError('Checkpoint must belong directly to this run')
        if json.loads((path.parent/'config.json').read_text()) != json.loads((run/'config.json').read_text()):
            raise ValueError('Checkpoint configurations differ')
        weights = path.parent/'pytorch_model.bin'
        digest = sha256(weights)
        sources.append(dict(step=step, epoch=epoch, path=str(weights.relative_to(run)), sha256=digest))
        jobs.append(dict(epoch=epoch, step=step, checkpoint=str(path.parent), weights_sha256=digest,
                         model_config_sha256=sha256(path.parent/'config.json')))
    record = dict(format_version=1, run=str(run), config_sha256=launch['config_sha256'],
        launch_provenance_sha256=sha256(run/'launch_provenance.json'),
        contract_path=str(WORK/'controlled/realizations/plan.json'),
        contract_sha256=sha256(WORK/'controlled/realizations/plan.json'), jobs=jobs, source_weights=sources,
        inference=contract['inference'], purpose='Scalar export and validation-only allocation selection; no test scores')
    # Preserve already prepared reference/primary job manifests byte-for-byte.
    if registration['family']=='learning_curves':
        record['run_registration'] = registration
    immutable_json(destination(run)/'jobs.json', record)
    return record


def validate_export(output, expected):
    output = Path(output)
    record = json.loads((output/'complete.json').read_text())
    if record['job'] != expected or sha256(output/'predictions.parquet') != record['predictions_sha256']:
        raise ValueError('Prediction export has different provenance or payload')
    return record


def worker(run, index, workers):
    import torch
    from datasets import load_from_disk
    from reward_model.model import load_reward_model
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUIDS[index]:
        raise RuntimeError('Each inference worker must use its assigned authorized GPU UUID')
    torch.set_num_threads(4)
    run = Path(run).resolve()
    directory = destination(run)
    frozen = json.loads((directory/'jobs.json').read_text())
    contract, rows, _ = load_contract(Path(frozen['contract_path']).parent)
    if frozen['run'] != str(run) or sha256(Path(frozen['contract_path'])) != frozen['contract_sha256']:
        raise ValueError('Frozen run or realization contract changed')
    datasets = {}
    for split in ('val','test'):
        ds = load_from_disk(str(Path(contract['tokenized_dataset_dir'])/f'{split}_examples'))
        if ds._fingerprint != contract['dataset_fingerprints'][split]:
            raise ValueError('Inference dataset fingerprint changed')
        expected = rows[rows.original_split == split].reset_index(drop=True)
        locations = align_rows(expected, ds.select_columns(IDENTITY+LABELS).to_pandas())
        if not np.array_equal(locations, np.arange(len(expected))):
            raise ValueError('Inference cache order changed after view preparation')
        datasets[split] = ds
    model = None
    for job in frozen['jobs'][index::workers]:
        checkpoint = Path(job['checkpoint'])
        if sha256(checkpoint/'pytorch_model.bin') != job['weights_sha256'] or sha256(checkpoint/'config.json') != job['model_config_sha256']:
            raise ValueError('Checkpoint changed before inference')
        pending = []
        for split in ('val','test'):
            output = directory/f"epoch{job['epoch']:02d}_{split}"
            expected = dict(**job, split=split, jobs_sha256=sha256(directory/'jobs.json'))
            if (output/'complete.json').exists():
                validate_export(output, expected)
            else:
                pending.append((split, output, expected))
        if not pending:
            continue
        if model is None:
            model = load_reward_model(str(checkpoint), device=torch.device('cpu'))
            model.to('cuda')
        else:
            state = torch.load(checkpoint/'pytorch_model.bin', map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            del state
        for split, output, expected in pending:
            started = time.monotonic()
            summary = score_dataset(model, datasets[split], output,
                batch_size=contract['inference']['batch_size'], save_embeddings=False)
            immutable_json(output/'complete.json', dict(job=expected, **summary,
                inference_seconds=time.monotonic()-started,
                predictions_sha256=sha256(output/'predictions.parquet'),
                dataset_fingerprint=datasets[split]._fingerprint,
                gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES']))


def choose_epochs(records):
    """Input consists exclusively of validation scores, one per epoch."""
    if not records or len({r['epoch'] for r in records}) != len(records):
        raise ValueError('Need one validation record per distinct epoch')
    valid = [r for r in records if r['macro_spearman'] is not None
             and math.isfinite(r['macro_spearman']) and r['defined_fraction'] is not None
             and 0 < r['defined_fraction'] <= 1]
    if not valid:
        raise ValueError('No defined validation checkpoint')
    return dict(max(valid, key=lambda r: (r['macro_spearman'], -r['epoch'])))


def validation_trajectories(rows, views, epoch_scores):
    selectors = {name[:-4]: (indices, RankingSelection(rows.iloc[indices].reset_index(drop=True)))
                 for name, indices in views.items() if name.endswith('_val')}
    records = {name: [] for name in selectors}
    for epoch, step, scores in epoch_scores:
        if np.shape(scores) != (len(rows),) or not np.isfinite(scores).all():
            raise ValueError('Incomplete held-out scalar predictions')
        for name, (indices, selector) in selectors.items():
            records[name].append(dict(epoch=epoch, step=step, **selector(np.asarray(scores)[indices])))
    return {name: dict(selected=choose_epochs(history), validation_history=history)
            for name, history in records.items()}


def select_validation(run):
    run = Path(run).resolve()
    directory = destination(run)
    frozen = json.loads((directory/'jobs.json').read_text())
    contract, rows, views = load_contract()
    if frozen['run'] != str(run) or sha256(Path(frozen['contract_path'])) != frozen['contract_sha256']:
        raise ValueError('Frozen validation export contract changed')
    evidence = []
    def epoch_scores():
        for job in frozen['jobs']:
            parts = []
            for split in ('val','test'):
                output = directory/f"epoch{job['epoch']:02d}_{split}"
                expected = dict(**job, split=split, jobs_sha256=sha256(directory/'jobs.json'))
                record = validate_export(output, expected)
                frame = pd.read_parquet(output/'predictions.parquet')
                reference = rows[rows.original_split == split].reset_index(drop=True)
                locations = align_rows(reference, frame)
                if not np.array_equal(locations, np.arange(len(reference))) or not np.array_equal(frame.row_index, np.arange(len(frame))):
                    raise ValueError('Exported row order differs from frozen layout')
                if record['rows'] != len(frame):
                    raise ValueError('Export row count differs from completion')
                parts.append(frame.ranking_score.to_numpy())
                evidence.append(dict(path=str((output/'complete.json').relative_to(directory)),
                                     sha256=sha256(output/'complete.json')))
            yield job['epoch'], job['step'], np.concatenate(parts)
    selections = validation_trajectories(rows, views, epoch_scores())
    record = dict(format_version=1, run=str(run), config_sha256=frozen['config_sha256'],
        jobs_sha256=sha256(directory/'jobs.json'), contract_sha256=frozen['contract_sha256'],
        selections=selections, source_weights=frozen['source_weights'], export_manifests=evidence,
        selected_steps=sorted({s['selected']['step'] for s in selections.values()}),
        test_metrics_computed=False, original_training_selection_unchanged=True,
        interpretation=contract['limitations'],
        remaining='Allocation-specific LR selection and selected-checkpoint test evaluation; aggregate-loss and averaging comparisons need full loss evaluation')
    immutable_json(directory/'validation_complete.json', record)
    return record


def verify_validation(run):
    """Verify frozen choices and every scalar payload before any checkpoint pruning."""
    run = Path(run).resolve()
    directory = destination(run)
    result = json.loads((directory/'validation_complete.json').read_text())
    frozen = json.loads((directory/'jobs.json').read_text())
    contract, _, _ = load_contract(Path(frozen['contract_path']).parent)
    if result['run'] != str(run) or frozen['run'] != str(run) or result['test_metrics_computed'] is not False:
        raise ValueError('Validation-only completion belongs to another run or exposes test metrics')
    if result['jobs_sha256'] != sha256(directory/'jobs.json') or result['source_weights'] != frozen['source_weights']:
        raise ValueError('Frozen checkpoint provenance changed')
    if sha256(run/'launch_provenance.json') != frozen['launch_provenance_sha256']:
        raise ValueError('Training provenance changed after allocation inference')
    if result['contract_sha256'] != sha256(Path(frozen['contract_path'])):
        raise ValueError('Allocation contract changed')
    if result['config_sha256'] != frozen['config_sha256'] or set(result['selections']) != set(contract['realizations']):
        raise ValueError('Validation configuration or allocation coverage changed')
    epochs = [(job['epoch'], job['step']) for job in frozen['jobs']]
    if len(epochs) != 15 or [epoch for epoch, _ in epochs] != list(range(1,16)):
        raise ValueError('Full 15-epoch inference trajectory is required')
    if any([(row['epoch'], row['step']) for row in item['validation_history']] != epochs for item in result['selections'].values()):
        raise ValueError('Allocation selection omitted epoch candidates')
    expected_steps = sorted({choose_epochs(item['validation_history'])['step'] for item in result['selections'].values()})
    if result['selected_steps'] != expected_steps or any(item['selected'] != choose_epochs(item['validation_history']) for item in result['selections'].values()):
        raise ValueError('Allocation selection differs from validation history')
    expected_paths = {f"epoch{job['epoch']:02d}_{split}/complete.json" for job in frozen['jobs'] for split in ('val','test')}
    if {item['path'] for item in result['export_manifests']} != expected_paths:
        raise ValueError('Every epoch/partition export is required before cleanup')
    for item in result['export_manifests']:
        path = directory/item['path']
        if sha256(path) != item['sha256']:
            raise ValueError('Export evidence changed after selection')
        record = json.loads(path.read_text())
        if sha256(path.parent/'predictions.parquet') != record['predictions_sha256']:
            raise ValueError('Export payload changed after selection')
    return result


def run_workers(run, workers=4):
    import fcntl
    run = Path(run).resolve()
    directory = destination(run)
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory/'export.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (directory/'validation_complete.json').exists():
        return verify_validation(run)
    prepare_jobs(run)
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    if any(line.split(',')[0].strip() in GPU_UUIDS[:workers] for line in busy.splitlines()):
        raise RuntimeError('Authorized inference GPU is busy; no export workers launched')
    processes, logs = [], []
    try:
        for index in range(workers):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = GPU_UUIDS[index]
            log = (directory/f'worker{index}.log').open('a')
            logs.append(log)
            command = [sys.executable, str(Path(__file__).resolve()), '--run', str(run),
                       '--workers', str(workers), '--worker', str(index)]
            processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                              stdout=log, stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError(f'Allocation inference failed; completed exports retained, return codes {codes}')
    finally:
        # A parent launch failure must not orphan a half-started worker group.
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for log in logs:
            log.close()
    select_validation(run)
    return verify_validation(run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4, choices=range(1,5))
    parser.add_argument('--worker', type=int)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if ROOT != Path('/home/atabey/Prot2Mol'):
        raise RuntimeError('Runtime is restricted to Atum Prot2Mol')
    if args.worker is not None:
        if not 0 <= args.worker < args.workers:
            raise ValueError('Invalid worker index')
        worker(args.run, args.worker, args.workers)
    elif args.prepare_only:
        print(json.dumps(prepare_jobs(args.run), indent=2))
    else:
        result = run_workers(args.run, args.workers)
        print(json.dumps(dict(selected_steps=result['selected_steps'], test_metrics_computed=False), indent=2))
