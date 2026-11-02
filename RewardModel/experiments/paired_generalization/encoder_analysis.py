"""E34-E37/Q09: existing encoder settings, selected scores and native trajectories.

No model-family winner or extra fit is selected here. Each setting keeps its
already frozen B-validation LR, and every reported trajectory spans 15 epochs.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path

import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256
    from . import controlled_summary, control_plan
else:
    from run import ROOT, WORK, sha256
    import controlled_summary, control_plan

read = controlled_summary.read
immutable = controlled_summary.immutable
ARMS = {
    'full': {},
    'frozen': dict(freeze_protein_encoder=True, freeze_molecule_encoder=True),
    'last_two_blocks': dict(protein_unfrozen_last_layers=2, molecule_unfrozen_last_layers=2),
    **{f'lora{r}': dict(encoder_lora_rank=r, encoder_lora_alpha=2.*r) for r in (4, 16, 64)},
}
LOSSES = ('loss', 'ranking_loss', 'classification_loss', 'contrastive_loss')
TRAJECTORY_METRICS = (*LOSSES, 'macro_spearman', 'roc_auc',
    'spearman_num_groups', 'spearman_num_defined_groups', 'spearman_defined_fraction')


def software():
    result = controlled_summary.software()
    path = Path(__file__)
    result['source_sha256'][str(path.relative_to(ROOT))] = sha256(path)
    return result


def validate_grid(primary):
    declared = [a for a in primary['arms'] if a['family'] == 'encoder_freedom']
    if (len(declared) != 6 or {a['name'] for a in declared} != set(ARMS) or
        primary['gpu_uuids'] != list(control_plan.GPU_UUIDS)):
        raise ValueError('Encoder settings or authorized devices changed')
    for arm in declared:
        if arm['model'] != ARMS[arm['name']] or arm['training']:
            raise ValueError('Encoder setting changes more than declared encoder freedom')
    jobs = [j for j in primary['tuning_jobs'] if j['arm'] in ARMS]
    expected = {control_plan.job_id(a, r, 42, lr) for a in ARMS for r in ('A', 'B')
                for lr in control_plan.LEARNING_RATE_MULTIPLIERS}
    if len(jobs) != 36 or {j['id'] for j in jobs} != expected:
        raise ValueError('Need exactly 36 existing encoder tuning fits')
    references = {(j['regime'], j['learning_rate_multiplier']): j['config'] for j in jobs if j['arm'] == 'full'}
    for job in jobs:
        config = job['config']; training = config['training']
        reference = references[job['regime'], job['learning_rate_multiplier']]
        model = reference['model']
        if (model['freeze_protein_encoder'] or model['freeze_molecule_encoder'] or
            model['protein_unfrozen_last_layers'] is not None or model['molecule_unfrozen_last_layers'] is not None or
            model['encoder_lora_rank'] != 0 or model['ranking_readout'] != 'ranking_head'):
            raise ValueError('Full reference is not the original full-finetuning ranker')
        if (job['seed'] != 42 or job['phase'] != 'tuning' or
            job['id'] != control_plan.job_id(job['arm'], job['regime'], 42, job['learning_rate_multiplier']) or
            (training['num_train_epochs'], training['per_device_train_batch_size'], training['gradient_accumulation_steps']) != (15, 12, 4)):
            raise ValueError('Encoder fit identity or budget changed')
        common = copy.deepcopy(reference)
        common['model'].update(ARMS[job['arm']])
        common['training']['output_dir'] = training['output_dir']
        if config != common:
            raise ValueError('Encoder comparison changes another training or data parameter')
    return jobs


def describe_curve(rows):
    """Descriptive extrema and adjacent increases, without an overfitting test."""
    if [r['epoch'] for r in rows] != list(range(1, 16)):
        raise ValueError('Need the complete 15-epoch trajectory')
    minimum = {key: min(rows, key=lambda r: (r['val'][key], r['epoch']))['epoch'] for key in LOSSES}
    increases = {key: next(([a['epoch'], b['epoch']] for a, b in zip(rows, rows[1:])
        if b['val'][key] > a['val'][key]), None) for key in LOSSES}
    defined = [r for r in rows if r['val']['macro_spearman'] is not None]
    peak = max(defined, key=lambda r: (r['val']['macro_spearman'], -r['epoch'])) if defined else None
    final = rows[-1]
    probe, val = final['train_probe']['macro_spearman'], final['val']['macro_spearman']
    return dict(loss_minimum_epochs=minimum, first_adjacent_loss_increases=increases,
        validation_macro_peak_epoch=peak['epoch'] if peak else None,
        validation_macro_peak=peak['val']['macro_spearman'] if peak else None,
        final_validation_macro=val, final_probe_macro=probe,
        final_probe_minus_validation=probe-val if probe is not None and val is not None else None,
        final_minus_peak_macro=val-peak['val']['macro_spearman'] if peak and val is not None else None,
        interpretation='Earliest extrema and first strictly positive adjacent loss changes in the full native history. A single increase is not a sustained rise or a statistical overfitting test. Fixed training probes are not full-training metrics.')


def trajectory(job):
    """Read hash-bound native train/validation exports of a completed managed fit."""
    run = Path(job['output_dir']); state_path = run/'trainer_state.json'; state = read(state_path)
    records = [r for r in state['log_history'] if 'eval_macro_spearman' in r and 'eval_loss' in r]
    if (state['epoch'] != 15 or [r['epoch'] for r in records] != list(range(1, 16)) or
        state['global_step'] != records[-1]['step'] or
        any(a['step'] >= b['step'] for a, b in zip(records, records[1:]))):
        raise ValueError('Native training history is incomplete or out of order')
    evidence = {str(state_path): sha256(state_path)}; cohorts = {}; rows = []
    for record in records:
        row = dict(epoch=record['epoch'], step=record['step'])
        for prefix, role in (('eval', 'val'), ('train_eval', 'train_probe')):
            path = run/'prediction_exports/trajectory'/f"{prefix}_step{record['step']:08d}.json"
            manifest = read(path); payload = path.with_suffix('.npz')
            if (manifest['format_version'] != 1 or manifest['phase'] != 'trajectory' or
                manifest['metric_prefix'] != prefix or manifest['ranking_readout'] != 'ranking_head' or
                (manifest['epoch'], manifest['step']) != (record['epoch'], record['step']) or
                manifest['predictions_file'] != payload.name or sha256(payload) != manifest['predictions_sha256']):
                raise ValueError('Native trajectory export changed or has wrong provenance')
            cohort = dict(dataset_fingerprint=manifest['dataset_fingerprint'], rows=manifest['rows'])
            if not cohort['dataset_fingerprint'] or cohort['rows'] <= 0 or cohorts.setdefault(role, cohort) != cohort:
                raise ValueError('Trajectory dataset fingerprint or row count changed')
            values = {key: manifest['metrics'][f'{prefix}_{key}'] for key in TRAJECTORY_METRICS}
            if any(v is not None and not math.isfinite(v) for v in values.values()) or any(values[k] is None for k in LOSSES):
                raise ValueError('Nonfinite trajectory loss or metric')
            if role == 'val' and any(values[k] != record[f'eval_{k}'] for k in TRAJECTORY_METRICS):
                raise ValueError('Validation export disagrees with native Trainer history')
            row[role] = values
            evidence[str(path)] = sha256(path); evidence[str(payload)] = manifest['predictions_sha256']
        rows.append(row)
    return dict(job_id=job['id'], cohorts=cohorts, rows=rows, **describe_curve(rows),
        evidence_sha256=evidence, test_predictions_read=False,
        verification_scope='Native reported metrics and full prediction-payload hashes, not an independent recomputation of trajectory metrics.')


def verify_summary(path):
    path = Path(path); result = read(path); arm = result['summary']['arm']
    if arm not in ARMS:
        raise ValueError('Unregistered encoder summary')
    analyses = result['source']['analysis_sha256']
    for name, digest in analyses.items():
        if sha256(Path(name)) != digest:
            raise ValueError('Primary analysis changed')
    canonical = controlled_summary.summarize(arm, sorted(analyses))
    if canonical.resolve() != path.resolve() or read(canonical) != result:
        raise ValueError('Use the verified canonical primary summary')
    records = [(Path(p), read(Path(p))) for p in analyses]
    return result, {r['job']['id']: (p, r) for p, r in records}


def analyze(paths):
    primary_path = WORK/'controlled/plan.json'; primary = read(primary_path); validate_grid(primary)
    paths = [Path(p) for p in paths]
    if len(paths) != 6 or {read(p)['summary']['arm'] for p in paths} != set(ARMS):
        raise ValueError('Need all six complete encoder primary summaries')
    summaries = {}; analyses = {}; qualification = None; protocols = set(); runtime_evidence = {}
    for path in paths:
        result, records = verify_summary(path); summary = result['summary']; arm = summary['arm']
        if (result['source']['primary_plan_sha256'] != sha256(primary_path) or
            result['source']['software'] != controlled_summary.software() or summary['ranking_readout'] != 'ranking_head'):
            raise ValueError('Encoder primary plan, metrics or readout changed')
        if set(records) != set(controlled_summary.expected_jobs(arm, summary['selected_learning_rate_multiplier'])):
            raise ValueError('Encoder report omits tuning or confirmation fits')
        summaries[arm] = result; analyses[arm] = records
        for _, record in records.values():
            protocols.add(control_plan.digest_json({k: record['source'][k] for k in ('software', 'design_sha256', 'noise_summary_sha256')}))
            runtime_path = Path(record['job']['output_dir'])/'runtime_qualification.json'; runtime = read(runtime_path)
            if runtime['world_size'] != 4 or runtime['signature'] != control_plan.digest_json(dict(sources=runtime['sources'], plan_sha256=sha256(primary_path))):
                raise ValueError('Invalid training qualification')
            if qualification is None:
                qualification = runtime
                for name, digest in runtime['sources'].items():
                    if not (ROOT/name).resolve().is_relative_to(ROOT.resolve()) or sha256(ROOT/name) != digest:
                        raise ValueError('Qualified training source changed')
            elif runtime != qualification:
                raise ValueError('Training implementations differ across encoders')
            runtime_evidence[str(runtime_path)] = sha256(runtime_path)
    if len(protocols) != 1:
        raise ValueError('Metric, composition or noise protocols differ across encoders')
    source = dict(primary_plan_sha256=sha256(primary_path), summary_sha256={str(p): sha256(p) for p in paths},
        runtime_qualification_sha256=runtime_evidence, software=software())
    target = WORK/'encoder_analysis'/control_plan.digest_json(source)[:16]/'complete.json'
    if target.exists():
        saved = read(target); payload = {k: v for k, v in saved.items() if k not in ('source', 'payload_sha256')}
        if saved['source'] != source or saved['payload_sha256'] != control_plan.digest_json(payload):
            raise ValueError('Completed encoder report changed')
        for name, digest in saved['artifact_sha256'].items():
            if sha256(Path(name)) != digest:
                raise ValueError('Encoder comparison artifact changed')
        return target
    settings = {arm: result['summary'] for arm, result in summaries.items()}
    curves = {}; cohorts = {}; hashes = {}; comparisons = {}; spreads = {}; cache = {}
    for arm, records in analyses.items():
        for identity, (_, record) in records.items():
            job = record['job']; curve = trajectory(job); curves[identity] = curve
            key = (job['regime'], job['seed'])
            if cohorts.setdefault(key, curve['cohorts']) != curve['cohorts']:
                raise ValueError('Matched encoder fits use different validation or fixed-probe cohorts')
            hashes.update(curve['evidence_sha256'])
    for arm in list(ARMS)[1:]:
        comparisons[arm] = {}; spreads[arm] = {}
        for rule in controlled_summary.RULES:
            comparisons[arm][rule] = {}; spreads[arm][rule] = {}
            for split in ('val', 'test'):
                comparisons[arm][rule][split] = {}; values = {}
                for regime in ('A', 'B'):
                    comparisons[arm][rule][split][regime] = {}; values[regime] = {}
                    for seed in controlled_summary.SEEDS:
                        tables = []; inputs = []
                        for setting in ('full', arm):
                            lr = settings[setting]['selected_learning_rate_multiplier']
                            identity = control_plan.job_id(setting, regime, seed, lr)
                            path, record = analyses[setting][identity]; checkpoint = record['rules'][rule]['checkpoint']
                            table = path.parent/checkpoint/split/'groups_ranking_score.parquet'
                            tables.append(table); hashes[str(table)] = sha256(table)
                            inputs.append(dict(job_id=identity, checkpoint=checkpoint, learning_rate_multiplier=lr))
                        key = tuple(map(str, tables))
                        if key not in cache:
                            cache[key] = controlled_summary.paired_group_change(*(pd.read_parquet(p) for p in tables))
                        change = dict(reference=inputs[0], alternative=inputs[1], **cache[key])
                        comparisons[arm][rule][split][regime][str(seed)] = change
                        values[regime][str(seed)] = dict(mean_change=change['mean_change'])
                spreads[arm][rule][split] = {r: controlled_summary.seed_spread(v) for r, v in values.items()}
                spreads[arm][rule][split]['A_minus_B'] = controlled_summary.seed_spread({str(s):
                    controlled_summary.subtract(values['A'][str(s)], values['B'][str(s)]) for s in controlled_summary.SEEDS})
    payload = dict(settings=settings, native_trajectories=curves, encoder_minus_full=comparisons,
        paired_change_seed_spread=spreads, artifact_sha256=hashes, test_used_for_selection=False,
        new_fits_created=0, study_complete=False,
        accounting=dict(settings=6, unique_tuning_fits=36, unique_confirmation_fits=24,
            unique_total_fits=60, shared_full_reference_fits=10, additional_encoder_fits=50),
        interpretation='All six registered settings, five checkpoint rules and three diagnostic scorers remain visible. Each setting retains its B-validation-selected LR; paired encoder-minus-full changes therefore compare selected procedures, not a fixed-LR causal encoder effect. Same-LR seed42 results and all60 native trajectories remain available. Whole-unit intervals use mutually defined eligible assays and are conditional on fitted models; seed42 tuning exposure and seed43/44 confirmation spread are separate. A-minus-B is descriptive across different populations. Curve extrema/first increases do not establish a sustained rise, encoder drift or its cause. No family winner is chosen, no test score selects a setting and no new fit is created.')
    for name, digest in {str(primary_path): source['primary_plan_sha256'], **source['summary_sha256'], **runtime_evidence, **hashes}.items():
        if sha256(Path(name)) != digest:
            raise ValueError('Encoder evidence changed during reporting')
    immutable(target, dict(source=source, payload_sha256=control_plan.digest_json(payload), **payload))
    return target


def preflight():
    jobs = validate_grid(read(WORK/'controlled/plan.json'))
    return dict(arms=list(ARMS), tuning_jobs=[j['id'] for j in jobs],
        summary_candidates={a: [str(p) for p in sorted((WORK/'controlled_summaries'/a).glob('*/complete.json'))] for a in ARMS},
        new_fits_created=0, scientific_comparison_created=False,
        interpretation='Read-only inventory; report paths alone do not prove scientific completion.')


if __name__ == '__main__':
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use CPU-only Atum Prot2Mol')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('analyze', 'preflight')); parser.add_argument('--summaries', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps(preflight(), indent=2) if args.action == 'preflight' else analyze(args.summaries))
