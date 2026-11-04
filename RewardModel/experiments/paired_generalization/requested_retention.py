"""Retain the requested best-macro model and native curves, without extra inference.

The historic selection-rule scalars remain for native evaluation compatibility.
No averaged model, allocation predictions, or allocation epoch selection is made.
Only redundant payloads of the exact completed managed fit may be pruned.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import torch
from .run import sha256, write_json
from .retain_controlled import selection_rules, _verify_manifest


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
    if launch['world_size'] != 4 or training['num_train_epochs'] != 15:
        raise ValueError('Requested retention requires four ranks and fifteen epochs')
    if summary['test_evaluated'] or summary['test_metrics']:
        raise ValueError('Training must not score test examples')
    if marker.get('seed') != 42 or marker.get('learning_rate_multiplier') != 1.:
        raise ValueError('Requested retention requires fixed seed42 and base learning rate')
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

    retained = run / 'retained'; manifest_path = retained / 'retention.json'
    if manifest_path.exists():
        result = json.loads(manifest_path.read_text())
        if (result['policy'] != 'requested_best_macro_and_final_only' or
            result['config_sha256'] != config_hash or result['selections'] != selections or
            result['prediction_manifests'] != prediction_manifests or
            result['averaging'] is not None or result['additional_realizations'] is not None):
            raise ValueError('Scoped retention provenance changed')
        for item in result['preserved_weights']:
            if sha256(run / item['path']) != item['sha256']:
                raise ValueError('Preserved model payload changed')
    else:
        sources = [dict(path=str((p/'pytorch_model.bin').relative_to(run)),
                        sha256=sha256(p/'pytorch_model.bin')) for p in checkpoints]
        macro = run / f"checkpoint-{selections['best_macro']['step']}" / 'pytorch_model.bin'
        root_weights = torch.load(run/'pytorch_model.bin', map_location='cpu', weights_only=True)
        selected_weights = torch.load(macro, map_location='cpu', weights_only=True)
        if set(root_weights) != set(selected_weights) or any(
                not torch.equal(root_weights[k], selected_weights[k]) for k in root_weights):
            raise ValueError('Root model is not the selected macro checkpoint')
        preserved = list(dict.fromkeys([run/'pytorch_model.bin', macro, checkpoints[-1]/'pytorch_model.bin']))
        result = dict(format_version=1, policy='requested_best_macro_and_final_only',
            config_sha256=config_hash, selections=selections, averaging=None,
            prediction_manifests=prediction_manifests, source_weights=sources,
            preserved_weights=[dict(path=str(p.relative_to(run)), sha256=sha256(p)) for p in preserved],
            additional_realizations=None, inference_performed=False, pruned=False,
            selection_note='Only best_macro is evaluated. Other scalar rules are reconstructed from the existing native history for compatibility; no extra models are selected or scored.')
        write_json(manifest_path, result)
    if prune and not result['pruned']:
        # Check every source before removing any redundant payload.
        for item in result['source_weights']:
            path = run / item['path']
            if not path.exists() or sha256(path) != item['sha256']:
                raise ValueError('Source checkpoint changed before retention cleanup')
        keep = {str((run/item['path']).resolve()) for item in result['preserved_weights']}
        candidates = [p for checkpoint in checkpoints for p in
            [checkpoint/'pytorch_model.bin', checkpoint/'optimizer.pt', checkpoint/'scheduler.pt',
             checkpoint/'training_args.bin', *checkpoint.glob('rng_state*.pth')]]
        if any(p.is_symlink() for p in candidates):
            raise ValueError('Refusing to prune a symlinked training payload')
        removed = []
        for path in candidates:
            if path.exists() and str(path.resolve()) not in keep:
                path.unlink(); removed.append(str(path.relative_to(run)))
        result.update(pruned=True, removed_files=removed, optimizer_resume_available=False)
        write_json(manifest_path, result)
    return result
