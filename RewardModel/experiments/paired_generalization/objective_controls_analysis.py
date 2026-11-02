"""E39/E42-E47: matched reports for existing weight and objective controls.

Compare the registered systems at their frozen B-selected learning rates. No
additional hyperparameter or training run is introduced here. The weight report
links existing validation-only family choices without selecting again on test.
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
    from . import controlled_summary, control_plan, weight_selection
else:
    from run import ROOT, WORK, sha256
    import controlled_summary, control_plan, weight_selection

read = controlled_summary.read
immutable = controlled_summary.immutable
CHANGES = {
    'full': {},
    'bce_low': dict(classification_loss_weight=.0625),
    'bce_high': dict(classification_loss_weight=1.),
    'contrastive_low': dict(contrastive_loss_weight=.25),
    'contrastive_high': dict(contrastive_loss_weight=4.),
    'unmasked': dict(contrastive_mask_policy='none'),
    'training_known_mask': dict(contrastive_mask_policy='training_known'),
    'fixed_scale': dict(learn_contrastive_scale=False),
    'detach_contrastive': dict(contrastive_detach_encoders=True),
    'listwise_normalized': dict(ranking_normalize_per_group=True),
    'assay_balanced': dict(ranking_normalize_per_group=True, ranking_balance_assays=True),
    'pairwise_top': dict(ranking_objective='pairwise_logistic', ranking_position_weighting='top'),
    'pairwise_uniform': dict(ranking_objective='pairwise_logistic', ranking_position_weighting='uniform'),
    'pairwise_hard_zero_margin': dict(ranking_objective='pairwise_logistic', ranking_position_weighting='uniform', ranking_affinity_margin=0.),
    'pairwise_noise_soft': dict(ranking_objective='pairwise_logistic', ranking_position_weighting='uniform', ranking_affinity_margin=0., ranking_pair_target_temperature='training_replicate_rms'),
}
FAMILIES = {
    'weights': dict(registry='static_weight_sweep', arms=tuple(weight_selection.WEIGHTS),
        contrasts={
            'low_bce_minus_reference': ('full', 'bce_low'),
            'high_bce_minus_reference': ('full', 'bce_high'),
            'low_contrastive_minus_reference': ('full', 'contrastive_low'),
            'high_contrastive_minus_reference': ('full', 'contrastive_high'),
        }),
    'contrastive': dict(registry='contrastive_controls', arms=('full', 'unmasked', 'training_known_mask', 'fixed_scale', 'detach_contrastive'),
        contrasts={
            'reference_masks_minus_unmasked': ('unmasked', 'full'),
            'training_known_minus_reference_masks': ('full', 'training_known_mask'),
            'fixed_minus_learned_scale': ('full', 'fixed_scale'),
            'detached_minus_encoder_gradients': ('full', 'detach_contrastive'),
        }),
    'ranking': dict(registry='ranking_controls', arms=('full', 'listwise_normalized', 'assay_balanced', 'pairwise_top', 'pairwise_uniform', 'pairwise_hard_zero_margin', 'pairwise_noise_soft'),
        contrasts={
            'normalized_minus_original_listwise': ('full', 'listwise_normalized'),
            'assay_balanced_minus_list_normalized': ('listwise_normalized', 'assay_balanced'),
            'top_pairwise_minus_original_listwise': ('full', 'pairwise_top'),
            'uniform_minus_top_pair_weights': ('pairwise_top', 'pairwise_uniform'),
            'zero_minus_original_pair_margin': ('pairwise_uniform', 'pairwise_hard_zero_margin'),
            'noise_soft_minus_zero_margin_hard_targets': ('pairwise_hard_zero_margin', 'pairwise_noise_soft'),
        }),
}


def software():
    result = controlled_summary.software(); path = Path(__file__)
    result['source_sha256'][str(path.relative_to(ROOT))] = sha256(path)
    path = Path(weight_selection.__file__)
    result['source_sha256'][str(path.relative_to(ROOT))] = sha256(path)
    return result


def validate_grid(primary, family):
    if family not in FAMILIES:
        raise ValueError('Unknown objective-control family')
    spec = FAMILIES[family]; arms = spec['arms']
    declared = [a for a in primary['arms'] if a['family'] == spec['registry']]
    if (len(declared) != len(arms)-1 or {a['name'] for a in declared} != set(arms)-{'full'} or
        sum(a['name'] == 'full' for a in primary['arms']) != 1 or primary['gpu_uuids'] != list(control_plan.GPU_UUIDS)):
        raise ValueError('Objective-control settings or devices changed')
    for arm in declared:
        if arm['model'] != CHANGES[arm['name']] or arm['training']:
            raise ValueError('Objective setting differs from the registered intervention')
    jobs = [j for j in primary['tuning_jobs'] if j['arm'] in arms]
    expected = {control_plan.job_id(a, r, 42, lr) for a in arms for r in ('A', 'B') for lr in control_plan.LEARNING_RATE_MULTIPLIERS}
    if len(jobs) != 6*len(arms) or {j['id'] for j in jobs} != expected:
        raise ValueError('Need all existing matched tuning configurations')
    if family == 'weights':
        weight_selection.family_jobs(primary)
    references = {(j['regime'], j['learning_rate_multiplier']): j['config'] for j in jobs if j['arm'] == 'full'}
    baseline = dict(contrastive_mask_policy='reference', learn_contrastive_scale=True, contrastive_detach_encoders=False,
        ranking_objective='ligunity_listwise', ranking_normalize_per_group=False, ranking_balance_assays=False,
        ranking_position_weighting='top', ranking_pair_target_temperature=None, ranking_readout='ranking_head')
    for job in jobs:
        config = job['config']; training = config['training']; reference = references[job['regime'], job['learning_rate_multiplier']]
        if any(reference['model'][k] != v for k, v in baseline.items()):
            raise ValueError('Full reference objective or readout changed')
        if (job['seed'] != 42 or job['phase'] != 'tuning' or
            job['id'] != control_plan.job_id(job['arm'], job['regime'], 42, job['learning_rate_multiplier']) or
            (training['num_train_epochs'], training['per_device_train_batch_size'], training['gradient_accumulation_steps']) != (15, 12, 4)):
            raise ValueError('Objective-control identity or training budget changed')
        common = copy.deepcopy(reference); changes = dict(CHANGES[job['arm']])
        if changes.get('ranking_pair_target_temperature') == 'training_replicate_rms':
            noise = primary['training_noise_rms'][job['regime']]
            if not math.isfinite(noise) or noise <= 0:
                raise ValueError('Soft targets require positive finite training-only noise')
            changes['ranking_pair_target_temperature'] = math.sqrt(3.)*noise/math.pi
        common['model'].update(changes); common['training']['output_dir'] = training['output_dir']
        if common != config:
            raise ValueError('Objective control changes another model, data or training parameter')
    return jobs


def verify_summary(path, family):
    path = Path(path); result = read(path); arm = result['summary']['arm']
    if arm not in FAMILIES[family]['arms']:
        raise ValueError('Summary belongs to another family')
    analyses = result['source']['analysis_sha256']
    for name, digest in analyses.items():
        if sha256(Path(name)) != digest:
            raise ValueError('Primary objective-control analysis changed')
    canonical = controlled_summary.summarize(arm, sorted(analyses))
    if canonical.resolve() != path.resolve() or read(canonical) != result:
        raise ValueError('Use the verified canonical primary summary')
    records = [(Path(p), read(p)) for p in analyses]
    return result, {r['job']['id']: (p, r) for p, r in records}


def analyze(family, paths):
    primary_path = WORK/'controlled/plan.json'; primary = read(primary_path); validate_grid(primary, family)
    spec = FAMILIES[family]; paths = [Path(p) for p in paths]
    if len(paths) != len(spec['arms']) or {read(p)['summary']['arm'] for p in paths} != set(spec['arms']):
        raise ValueError('Need every completed primary summary in this family')
    summaries = {}; analyses = {}; qualification = None; protocols = set(); runtime_evidence = {}; analysis_evidence = {}
    for path in paths:
        result, records = verify_summary(path, family); summary = result['summary']; arm = summary['arm']
        if (result['source']['primary_plan_sha256'] != sha256(primary_path) or
            result['source']['software'] != controlled_summary.software() or summary['ranking_readout'] != 'ranking_head' or
            set(records) != set(controlled_summary.expected_jobs(arm, summary['selected_learning_rate_multiplier']))):
            raise ValueError('Objective-control summary differs in plan, scorer, protocol or fit coverage')
        summaries[arm] = summary; analyses[arm] = records
        analysis_evidence.update(result['source']['analysis_sha256'])
        for _, record in records.values():
            protocols.add(control_plan.digest_json({k: record['source'][k] for k in ('software', 'design_sha256', 'noise_summary_sha256')}))
            path = Path(record['job']['output_dir'])/'runtime_qualification.json'; runtime = read(path)
            if runtime['world_size'] != 4 or runtime['signature'] != control_plan.digest_json(dict(sources=runtime['sources'], plan_sha256=sha256(primary_path))):
                raise ValueError('Invalid training qualification')
            if qualification is None:
                qualification = runtime
                for name, digest in runtime['sources'].items():
                    if not (ROOT/name).resolve().is_relative_to(ROOT.resolve()) or sha256(ROOT/name) != digest:
                        raise ValueError('Qualified training source changed')
            elif runtime != qualification:
                raise ValueError('Objective-control training qualifications differ')
            runtime_evidence[str(path)] = sha256(path)
    if len(protocols) != 1:
        raise ValueError('Metrics, composition or noise definitions differ')
    weight_comparison = None; selection_evidence = {}
    if family == 'weights':
        decision = weight_selection.verify()
        if decision['primary_plan_sha256'] != sha256(primary_path):
            raise ValueError('Weight selection belongs to another primary plan')
        for arm, summary in summaries.items():
            if summary['selected_learning_rate_multiplier'] != decision['within_setting']['B'][arm]['learning_rate_multiplier']:
                raise ValueError('Weight report must use each setting\'s frozen B-selected LR')
        selection_evidence[str(weight_selection.destination())] = sha256(weight_selection.destination())
        weight_comparison = {k: decision[k] for k in ('choices', 'within_setting', 'all_candidates',
            'different_weights', 'different_learning_rates', 'family_tie_order', 'accounting', 'historical_limit')}
        weight_comparison['interpretation'] = ('A and B choices are computed from validation only. '
            'The repeated-seed test comparisons below use each setting\'s B-selected LR. '
            'They do not evaluate an A-selected LR when it differs from that setting\'s B-selected LR. '
            'The A-selected procedure is a diagnostic comparator, not an independently validated deployment procedure: '
            'training and selection exposure can cross partition roles between the A and B regimes. '
            'No cross-regime independence is asserted. The B-selected family choice is identified without selecting on test scores; '
            'all five settings remain visible. This axis sweep does not establish globally optimal weights or reconstruct historical tuning.')
    source = dict(family=family, primary_plan_sha256=sha256(primary_path), summary_sha256={str(p): sha256(p) for p in paths},
        analysis_sha256=analysis_evidence, runtime_qualification_sha256=runtime_evidence,
        weight_selection_sha256=selection_evidence, software=software())
    target = WORK/'objective_controls_analysis'/family/control_plan.digest_json(source)[:16]/'complete.json'
    if target.exists():
        saved = read(target); payload = {k: v for k, v in saved.items() if k not in ('source', 'payload_sha256')}
        if saved['source'] != source or saved['payload_sha256'] != control_plan.digest_json(payload):
            raise ValueError('Completed objective-control report changed')
        for name, digest in saved['group_tables_sha256'].items():
            if sha256(Path(name)) != digest:
                raise ValueError('Compared group table changed')
        return target
    comparisons = {}; spreads = {}; cache = {}; hashes = {}
    for name, (reference, alternative) in spec['contrasts'].items():
        comparisons[name] = {}; spreads[name] = {}
        for rule in controlled_summary.RULES:
            comparisons[name][rule] = {}; spreads[name][rule] = {}
            for split in ('val', 'test'):
                comparisons[name][rule][split] = {}; values = {}
                for regime in ('A', 'B'):
                    comparisons[name][rule][split][regime] = {}; values[regime] = {}
                    for seed in controlled_summary.SEEDS:
                        tables = []; inputs = []
                        for arm in (reference, alternative):
                            lr = summaries[arm]['selected_learning_rate_multiplier']; identity = control_plan.job_id(arm, regime, seed, lr)
                            path, record = analyses[arm][identity]; checkpoint = record['rules'][rule]['checkpoint']
                            table = path.parent/checkpoint/split/'groups_ranking_score.parquet'; tables.append(table); hashes[str(table)] = sha256(table)
                            inputs.append(dict(arm=arm, job_id=identity, checkpoint=checkpoint, learning_rate_multiplier=lr))
                        key = tuple(map(str, tables))
                        if key not in cache:
                            cache[key] = controlled_summary.paired_group_change(*(pd.read_parquet(p) for p in tables))
                        change = dict(reference=inputs[0], alternative=inputs[1], **cache[key])
                        comparisons[name][rule][split][regime][str(seed)] = change
                        values[regime][str(seed)] = dict(mean_change=change['mean_change'])
                spreads[name][rule][split] = {r: controlled_summary.seed_spread(v) for r, v in values.items()}
                spreads[name][rule][split]['A_minus_B'] = controlled_summary.seed_spread({str(s):
                    controlled_summary.subtract(values['A'][str(s)], values['B'][str(s)]) for s in controlled_summary.SEEDS})
    n = len(spec['arms'])
    payload = dict(family=family, settings=summaries, contrast_definitions=spec['contrasts'], paired_changes=comparisons,
        weight_selection_comparison=weight_comparison,
        paired_change_seed_spread=spreads, group_tables_sha256=hashes, test_used_for_selection=False, new_fits_created=0, study_complete=False,
        accounting=dict(settings=n, unique_tuning_fits=6*n, unique_confirmation_fits=4*n, unique_total_fits=10*n,
            shared_full_reference_fits=10, additional_control_fits=10*(n-1),
            all_families_unique_total_fits=10*len({a for f in FAMILIES.values() for a in f['arms']}),
            interpretation='All three family totals share the same ten full-reference fits. Their union contains only existing primary-plan fits; this reporter adds none.'),
        interpretation='All registered settings, three diagnostic scorers, five checkpoint rules and same-LR seed42 results remain visible. Contrasts are alternative minus reference at each setting\'s B-selected LR; they compare selected procedures, not a fixed-LR causal effect. Uniform versus top weights uses pairwise models on both sides. Soft versus hard targets uses zero-margin pairwise models on both sides, with a separate margin-removal contrast. Assay balancing is compared against list-normalization, not conflated with it. Whole-unit intervals condition on fitted models and mutually defined eligible assays; tuning-exposed seed42 and confirmation-only seeds43/44 remain separate. Different contrasts may use different defined cohorts. A-minus-B is descriptive across distinct populations. These ranking-score effects do not themselves establish false-negative causality, gradient alignment or calibration mechanisms. No family winner, new fit or test-driven selection is created.')
    for name, digest in {str(primary_path): source['primary_plan_sha256'], **source['summary_sha256'], **analysis_evidence, **runtime_evidence, **selection_evidence, **hashes}.items():
        if sha256(Path(name)) != digest:
            raise ValueError('Objective-control evidence changed during reporting')
    immutable(target, dict(source=source, payload_sha256=control_plan.digest_json(payload), **payload))
    return target


def preflight(family):
    jobs = validate_grid(read(WORK/'controlled/plan.json'), family); spec = FAMILIES[family]
    return dict(family=family, arms=list(spec['arms']), contrasts=spec['contrasts'], tuning_jobs=[j['id'] for j in jobs],
        summary_candidates={a: [str(p) for p in sorted((WORK/'controlled_summaries'/a).glob('*/complete.json'))] for a in spec['arms']},
        new_fits_created=0, scientific_comparison_created=False,
        interpretation='Read-only inventory; report paths alone do not prove completion.')


if __name__ == '__main__':
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use CPU-only Atum Prot2Mol')
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('analyze', 'preflight'))
    parser.add_argument('--family', choices=tuple(FAMILIES), required=True); parser.add_argument('--summaries', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps(preflight(args.family), indent=2) if args.action == 'preflight' else analyze(args.family, args.summaries))
