"""Preserve controlled-run evidence before pruning redundant checkpoint payloads.

Only an explicitly managed, completed run can be pruned. The original fresh
reference has no managed-run marker and is never eligible. Default invocation
stages and verifies artifacts without deleting anything.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import torch

if __package__:
    from .run import sha256, write_json
else:
    from run import sha256, write_json


def selection_rules(records, patience=3):
    """Actual checkpoints on a fixed full-horizon schedule; no test inputs."""
    if patience < 1 or not records:
        raise ValueError("Selection needs observations and positive patience")
    selected = {}
    for name, key, sign in (("macro", "eval_macro_spearman", 1), ("loss", "eval_loss", -1)):
        valid = [r for r in records if r.get(key) is not None and math.isfinite(r[key])]
        if not valid:
            raise ValueError(f"No finite selection metric: {key}")
        best = max(valid, key=lambda r: (sign*r[key], -r['step']))
        selected[f'best_{name}'] = dict(step=best['step'], epoch=best['epoch'], metric=best[key])
        incumbent, stale = None, 0
        stop = records[-1]
        for record in records:
            value = record.get(key)
            if value is not None and math.isfinite(value) and (incumbent is None or sign*value > sign*incumbent[key]):
                incumbent, stale = record, 0
            else:
                stale += 1
            if stale >= patience:
                stop = record
                break
        if incumbent is None:
            raise ValueError(f"Early stopping reached before any finite {key}")
        selected[f'patience{patience}_{name}'] = dict(
            step=incumbent['step'], epoch=incumbent['epoch'], metric=incumbent[key],
            stop_step=stop['step'], stop_epoch=stop['epoch'],
            interpretation='counterfactual prefix of the fixed full-horizon schedule; no compute was saved')
    return selected


def average_weights(checkpoints, target):
    """Uniform native-parameter average; integer buffers must agree exactly."""
    if not checkpoints:
        raise ValueError('Cannot average an empty window')
    total, dtypes = {}, {}
    for number, checkpoint in enumerate(checkpoints):
        current = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if number and set(current) != set(total):
            raise ValueError('Averaging checkpoint keys differ')
        for key, tensor in current.items():
            if tensor.is_complex() or not torch.isfinite(tensor).all():
                raise ValueError(f'Invalid averaging tensor: {key}')
            if not number:
                dtypes[key] = tensor.dtype
                total[key] = tensor.to(torch.float64) if tensor.is_floating_point() else tensor.clone()
            elif tensor.shape != total[key].shape or tensor.dtype != dtypes[key]:
                raise ValueError(f'Averaging tensor shape/dtype differs: {key}')
            elif tensor.is_floating_point():
                total[key].add_(tensor)
            elif not torch.equal(total[key], tensor):
                raise ValueError(f'Nonfloating buffer changed across averaging window: {key}')
    for key in total:
        if total[key].is_floating_point():
            total[key] = (total[key] / len(checkpoints)).to(dtypes[key])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.bin.tmp')
    torch.save(total, temporary)
    restored = torch.load(temporary, map_location='cpu', weights_only=True)
    if any(not torch.equal(total[key], restored[key]) for key in total):
        raise RuntimeError('Saved average failed tensor verification')
    temporary.replace(target)


def _verify_manifest(run, path):
    record = json.loads(path.read_text())
    payload = path.parent / record['predictions_file']
    if payload.parent.resolve() != path.parent.resolve() or sha256(payload) != record['predictions_sha256']:
        raise RuntimeError(f'Invalid prediction payload: {path.relative_to(run)}')
    return record


def retain(run, *, prune=False):
    run = Path(run).resolve()
    marker = json.loads((run / 'controlled_run.json').read_text())
    if marker.get('format_version') != 1 or marker.get('allow_checkpoint_retention') is not True:
        raise ValueError('Run is not managed for controlled checkpoint retention')
    if Path(marker['output_dir']).resolve() != run:
        raise ValueError('Managed-run marker belongs to a different directory')
    summary = json.loads((run / 'experiment_summary.json').read_text())
    launch = json.loads((run / 'launch_provenance.json').read_text())
    config_hash = marker['config_sha256']
    if any(record['config_sha256'] != config_hash for record in (summary, launch)):
        raise ValueError('Run configuration provenance differs')
    training = launch['config']['training']
    if training.get('export_evaluation_predictions') is not True or training.get('fixed_train_eval_assays', 0) <= 0:
        raise ValueError('Retention requires validation and fixed-training trajectories')
    horizon = int(training['num_train_epochs'])
    if float(training['num_train_epochs']) != horizon or horizon < 1:
        raise ValueError('Retention requires a whole-epoch horizon')
    for rank in range(launch['world_size']):
        done = json.loads((run / f'rank_{rank}_complete.json').read_text())
        if done['rank'] != rank or done['world_size'] != launch['world_size'] or done['config_sha256'] != config_hash:
            raise ValueError('Missing or mismatched rank completion')
    final_state = json.loads((run / 'trainer_state.json').read_text())
    if not math.isclose(final_state['epoch'], horizon, abs_tol=1e-6):
        raise ValueError('Training did not finish the declared horizon')
    records = [r for r in final_state['log_history'] if 'eval_macro_spearman' in r and 'eval_loss' in r]
    if len(records) != horizon or [round(r['epoch'], 6) for r in records] != list(range(1, horizon+1)):
        raise ValueError('Expected exactly one validation record per completed epoch')
    if records[-1]['step'] != final_state['global_step']:
        raise ValueError('Final checkpoint step differs from validation trajectory')
    selections = selection_rules(records)
    if training['metric_for_best_model'] != 'eval_macro_spearman' or not training['greater_is_better']:
        raise ValueError('Controlled root model must use macro-Spearman selection')
    if Path(final_state['best_model_checkpoint']).name != f"checkpoint-{selections['best_macro']['step']}":
        raise ValueError('Trainer selection differs from independently reconstructed selection')
    prediction_manifests = []
    checkpoints = []
    for record in records:
        checkpoint = run / f"checkpoint-{record['step']}"
        if checkpoint.is_symlink() or checkpoint.resolve().parent != run:
            raise ValueError('Checkpoint must be directly owned by this run')
        state = json.loads((checkpoint / 'trainer_state.json').read_text())
        if state['global_step'] != record['step'] or not math.isclose(state['epoch'], record['epoch'], abs_tol=1e-6):
            raise ValueError('Checkpoint epoch/step differs from trajectory')
        if json.loads((checkpoint / 'config.json').read_text()) != json.loads((run / 'config.json').read_text()):
            raise ValueError('Checkpoint model configurations differ')
        checkpoints.append(checkpoint)
        for prefix in ('eval', 'train_eval'):
            path = run / 'prediction_exports/trajectory' / f"{prefix}_step{record['step']:08d}.json"
            manifest = _verify_manifest(run, path)
            if manifest['step'] != record['step'] or not math.isclose(manifest['epoch'], record['epoch'], abs_tol=1e-6):
                raise ValueError('Export epoch/step differs from trajectory')
            if prefix == 'eval':
                for key in ('eval_loss', 'eval_macro_spearman'):
                    expected = record[key] if record[key] is not None and math.isfinite(record[key]) else None
                    if manifest['metrics'][key] != expected:
                        raise ValueError('Export metrics differ from Trainer history')
            prediction_manifests.append(dict(path=str(path.relative_to(run)), sha256=sha256(path)))
    selected_path = run / 'prediction_exports/selected' / f"eval_step{final_state['global_step']:08d}.json"
    selected_export = _verify_manifest(run, selected_path)
    if selected_export['metrics']['eval_macro_spearman'] != summary['eval_metrics']['eval_macro_spearman']:
        raise ValueError('Selected prediction export differs from run summary')
    prediction_manifests.append(dict(path=str(selected_path.relative_to(run)), sha256=sha256(selected_path)))

    additional, additional_steps = None, set()
    if marker.get('regime') == 'B':
        if __package__:
            from .realization_predictions import verify_validation, destination
        else:
            from realization_predictions import verify_validation, destination
        allocation = verify_validation(run)
        if allocation['config_sha256'] != config_hash:
            raise ValueError('Additional allocation exports belong to another training configuration')
        if [(r['epoch'], r['step']) for r in allocation['source_weights']] != [(r['epoch'], r['step']) for r in records]:
            raise ValueError('Additional allocation exports omit epoch checkpoints')
        # Scalar exports suffice for macro-rho selection, but not for the
        # allocation-specific contrastive/aggregate losses and weight averages.
        # Protect every B epoch until those remaining analyses have their own
        # verified release protocol; do not make them impossible to finish.
        additional_steps = {item['step'] for item in allocation['source_weights']}
        allocation_path = destination(run)/'validation_complete.json'
        additional = dict(path=str(allocation_path), sha256=sha256(allocation_path),
                          selected_steps=allocation['selected_steps'], protected_steps=sorted(additional_steps),
                          protection_reason='All B epoch weights retained pending allocation-specific full-loss/averaging analyses',
                          source_weights=allocation['source_weights'])

    retained = run / 'retained'
    manifest_path = retained / 'retention.json'
    if manifest_path.exists():
        result = json.loads(manifest_path.read_text())
        if result['config_sha256'] != config_hash or result['selections'] != selections or result['prediction_manifests'] != prediction_manifests:
            raise ValueError('Retention provenance changed')
        if result.get('additional_realizations') != additional:
            raise ValueError('Additional allocation retention evidence changed')
        for item in result['preserved_weights']:
            if sha256(run / item['path']) != item['sha256']:
                raise RuntimeError('Preserved model payload failed verification')
    else:
        sources = [dict(path=str((p/'pytorch_model.bin').relative_to(run)), sha256=sha256(p/'pytorch_model.bin')) for p in checkpoints]
        if additional is not None and sources != [{key: item[key] for key in ('path','sha256')} for item in additional['source_weights']]:
            raise ValueError('Exported allocation checkpoints differ from source weights')
        macro_path = run / f"checkpoint-{selections['best_macro']['step']}" / 'pytorch_model.bin'
        root_weights = torch.load(run/'pytorch_model.bin', map_location='cpu', weights_only=True)
        macro_weights = torch.load(macro_path, map_location='cpu', weights_only=True)
        if set(root_weights) != set(macro_weights) or any(not torch.equal(root_weights[k], macro_weights[k]) for k in root_weights):
            raise ValueError('Root model is not the selected macro checkpoint')
        del root_weights, macro_weights
        start_epoch = int(selections['best_loss']['epoch'])
        window = checkpoints[start_epoch-1:]
        average = retained / 'average_from_loss_minimum' / 'pytorch_model.bin'
        average_weights([p/'pytorch_model.bin' for p in window], average)
        shutil.copyfile(run/'config.json', average.parent/'config.json')
        retained_steps = {item['step'] for item in selections.values()} | additional_steps
        preserved = [run/'pytorch_model.bin', average] + [p/'pytorch_model.bin' for p in checkpoints if int(p.name.split('-')[1]) in retained_steps]
        result = dict(format_version=1, config_sha256=config_hash, selections=selections,
                      averaging=dict(rule='uniform native parameters from minimum validation loss through final epoch, inclusive',
                                     epochs=list(range(start_epoch, horizon+1)), singleton=len(window)==1),
                      prediction_manifests=prediction_manifests, source_weights=sources,
                      preserved_weights=[dict(path=str(p.relative_to(run)), sha256=sha256(p)) for p in preserved],
                      additional_realizations=additional, pruned=False)
        write_json(manifest_path, result)
    if prune and not result['pruned']:
        for item in result['source_weights']:
            path = run / item['path']
            if path.exists() and sha256(path) != item['sha256']:
                raise RuntimeError('Source checkpoint changed after retention staging')
        keep = {str((run / item['path']).resolve()) for item in result['preserved_weights']}
        removed = []
        for checkpoint in checkpoints:
            candidates = [checkpoint/'pytorch_model.bin', checkpoint/'optimizer.pt', checkpoint/'scheduler.pt',
                          checkpoint/'training_args.bin', *checkpoint.glob('rng_state*.pth')]
            for path in candidates:
                if path.is_symlink():
                    raise ValueError('Refusing to prune a symlinked payload')
                if path.exists() and str(path.resolve()) not in keep:
                    # Only known training payloads of this completed managed run.
                    path.unlink()
                    removed.append(str(path.relative_to(run)))
        result.update(pruned=True, removed_files=removed, optimizer_resume_available=False)
        write_json(manifest_path, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--prune', action='store_true')
    args = parser.parse_args()
    print(json.dumps(retain(args.run, prune=args.prune), indent=2))
