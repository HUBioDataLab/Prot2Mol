"""E41: CPU analysis and slopes for the registered whole-unit learning curves.

Thirty smaller-fraction fits plus the ten shared primary endpoint fits are
required. This module creates no training, LR selection or inference.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256
    from . import (learning_curve_evaluation as evaluation, learning_curve_plan as curve_plan,
                   selected_analysis, controlled_summary, control_plan, variant_relation)
else:
    from run import ROOT, WORK, sha256
    import learning_curve_evaluation as evaluation
    import learning_curve_plan as curve_plan
    import selected_analysis, controlled_summary, control_plan, variant_relation

read = selected_analysis.read
immutable = selected_analysis.immutable
finite = selected_analysis.finite_json
FRACTIONS = curve_plan.FRACTIONS
MACRO = 'original_eligible_macro_spearman'
METRICS = (*controlled_summary.METRICS, *(f'eligible_ndcg_at_{k}' for k in (1, 3, 5)))


def software():
    from importlib.metadata import version
    result = selected_analysis.software()
    for name in ('learning_curve_analysis.py', 'learning_curve_evaluation.py', 'learning_curve_plan.py',
                 'learning_curve_driver.py', 'controlled_summary.py', 'control_plan.py', 'variant_relation.py'):
        path = Path(__file__).with_name(name)
        result['source_sha256'][str(path.relative_to(ROOT))] = sha256(path)
    result['curve_plot_matplotlib'] = version('matplotlib')
    return result


def core_diagnostics(evaluations, refs, regime):
    """Use fixed held-out labels/units, never full-training similarity strata."""
    if regime not in ('A', 'B'):
        raise ValueError('Expected A or B')
    units = selected_analysis.unit_map_for(refs)
    partitions, tables = {}, {}
    for split in ('val', 'test'):
        prefix = 'eval' if split == 'val' else 'test'
        manifest = Path(evaluations[split]['prediction_manifest'])
        candidate, record = selected_analysis.read_native_predictions(manifest)
        if record['phase'] != 'fixed_checkpoint' or record['ranking_readout'] != 'ranking_head':
            raise ValueError('Curve needs fixed-checkpoint ranking-head predictions')
        locations = selected_analysis.align_rows(refs[split]['rows'], candidate)
        expected = refs[split]['rows'].iloc[locations].reset_index(drop=True)
        frame, record = selected_analysis.native_analysis.native_frame(
            manifest, expected, refs[split]['fingerprint'], prefix)
        metrics, groups = selected_analysis.rescore(frame, units)
        for scorer, table in groups.items():
            eligible = table.original_ranking_eligible
            metrics[scorer].update(eligible_groups=int(eligible.sum()),
                eligible_defined_fraction=float(table.loc[eligible, 'spearman'].notna().mean()) if eligible.any() else None,
                matched_eligible_ndcg={str(k): float(table.loc[eligible, f'ndcg_at_{k}_linear'].mean())
                    if eligible.any() else None for k in (1, 3, 5)})
            tables[f'{split}/groups_{scorer}'] = table
        headline = metrics['ranking_score']
        for computed, native in ((MACRO, 'macro_spearman'), ('eligible_defined_fraction', 'spearman_defined_fraction')):
            selected_analysis.native_analysis.check_native_metric(headline[computed], record['metrics'][prefix+'_'+native], native)
        if headline['eligible_groups'] != int(record['metrics'][prefix+'_spearman_num_groups']):
            raise ValueError('Native eligible cohort changed')
        partitions[split] = dict(rows=len(frame), metrics=metrics, native_metrics=record['metrics'])
    return finite(dict(regime=regime, ranking_readout='ranking_head', partitions=partitions,
        label_precision=selected_analysis.POLICY,
        unit_definition='assay' if regime == 'A' else 'scaffold-assay component',
        interpretation='Fixed held-out metric cohorts. Full-training scaffold support and composition strata are not reused as subset-training support.')), tables


def analysis_inputs(identity):
    if Path(identity).name != identity:
        raise ValueError('Expected a registered curve job ID')
    directory = WORK/'learning_curve_evaluation'/identity
    plan, results = evaluation.verify_complete(directory)
    config = read(Path(plan['job']['output_dir'])/'config.json')
    if config['ranking_min_pchembl_span'] != .5 or plan['job']['fraction'] not in FRACTIONS[:-1]:
        raise ValueError('Unexpected curve eligibility or fraction')
    design = WORK/'tier1/similarity_composition/design'
    refs = selected_analysis.references(plan, design)
    primary = read(WORK/'controlled/plan.json')
    noise = WORK/'tier1/replicates/noise_summary.json'
    if sha256(noise) != primary['noise_summary_sha256']:
        raise ValueError('Shared metric provenance changed')
    source = dict(evaluation_plan_sha256=sha256(directory/'plan.json'),
        evaluation_complete_sha256=sha256(directory/'complete.json'),
        design_sha256=sha256(design/'complete.json'), noise_summary_sha256=sha256(noise),
        software=software(), core_metric_software=selected_analysis.software(),
        token_payloads_used=False, training_support_strata_computed=False)
    return plan, results, refs, source


def verify_record(path, plan, source):
    path = Path(path); record = read(path)
    if record['phase'] != 'selected_prediction_analysis_complete' or record['source'] != source:
        raise ValueError('Curve analysis source or phase changed')
    for key in ('job', 'rules', 'seed_role', 'selected_learning_rate', 'selection_sha256', 'training_view'):
        if record[key] != plan[key]:
            raise ValueError('Curve analysis differs from completed native evaluation')
    if record['payload_sha256'] != control_plan.digest_json({k: v for k, v in record.items() if k != 'payload_sha256'}):
        raise ValueError('Curve analysis payload changed')
    selected_analysis.verify_artifacts(path.parent, record)
    checkpoints = {c['id']: c for c in plan['checkpoints']}
    if set(record['checkpoints']) != set(checkpoints):
        raise ValueError('Curve analysis checkpoint coverage changed')
    for name, checkpoint in checkpoints.items():
        child = read(path.parent/name/'complete.json')
        if (child['source'] != source or child['checkpoint'] != checkpoint or
            child['statistics'] != record['checkpoints'][name]):
            raise ValueError('Curve checkpoint analysis changed')
        required = {f'{s}/groups_{c}.parquet' for s in ('val', 'test') for c in selected_analysis.SCORERS}
        if set(child['artifacts_sha256']) != required or f'{name}/complete.json' not in record['artifacts_sha256']:
            raise ValueError('Curve group tables are incomplete')
        selected_analysis.verify_artifacts(path.parent/name, child)
    return record


def analyze_job(identity):
    import fcntl
    plan, evaluations, refs, source = analysis_inputs(identity)
    output = WORK/'learning_curve_analysis/jobs'/identity/control_plan.digest_json(source)[:16]
    output.mkdir(parents=True, exist_ok=True)
    with (output/'analysis.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        immutable(output/'analysis_plan.json', source)
        if (output/'complete.json').exists():
            verify_record(output/'complete.json', plan, source)
            return output/'complete.json'
        checkpoints = {}
        for checkpoint in plan['checkpoints']:
            destination = output/checkpoint['id']
            if (destination/'complete.json').exists():
                child = read(destination/'complete.json')
                if child['source'] != source or child['checkpoint'] != checkpoint:
                    raise ValueError('Existing checkpoint analysis changed')
                selected_analysis.verify_artifacts(destination, child)
            else:
                if destination.exists():
                    raise RuntimeError('Partial checkpoint analysis requires inspection')
                statistics, tables = core_diagnostics(evaluations[checkpoint['id']], refs, plan['job']['regime'])
                destination.mkdir()
                for name, table in tables.items():
                    path = destination/(name+'.parquet'); path.parent.mkdir(exist_ok=True)
                    table.to_parquet(path, index=False)
                child = dict(source=source, checkpoint=checkpoint, statistics=statistics,
                    artifacts_sha256={str(p.relative_to(destination)): sha256(p) for p in destination.rglob('*.parquet')})
                immutable(destination/'complete.json', child)
            checkpoints[checkpoint['id']] = child['statistics']
        result = dict(source=source, **{k: plan[k] for k in ('job', 'rules', 'seed_role',
            'selected_learning_rate', 'selection_sha256', 'training_view')}, checkpoints=checkpoints,
            phase='selected_prediction_analysis_complete',
            artifacts_sha256={str(p.relative_to(output)): sha256(p) for p in output.rglob('*')
                if p.is_file() and p.name not in ('analysis.lock', 'analysis_plan.json')})
        result['payload_sha256'] = control_plan.digest_json(finite(result))
        immutable(output/'complete.json', result)
        verify_record(output/'complete.json', plan, source)
    return output/'complete.json'


def segment(first, second, first_size, second_size):
    """Adjacent changes per actual doubling; repeated sizes have no slope."""
    if set(first) != set(second):
        raise ValueError('Curve metrics differ')
    denominators = {}
    for axis in ('rows', 'units'):
        a, b = first_size[axis], second_size[axis]
        if not isinstance(a, int) or not isinstance(b, int) or a <= 0 or b < a:
            raise ValueError('Whole-unit curve sizes must be positive nondecreasing integers')
        denominators[axis] = math.log2(b/a)
    changes = controlled_summary.subtract(second, first)
    for mapping in (first, second):
        if any(v is not None and not math.isfinite(v) for v in mapping.values()):
            raise ValueError('Undefined scores must be null, not nonfinite')
    return dict(score_change=changes, log2_size_change=denominators,
        per_doubling={axis: {k: v/d if v is not None and d > 0 else None for k, v in changes.items()}
            for axis, d in denominators.items()})


def size_contract(curve, primary, regime, fraction):
    view = evaluation.view_contract(curve, primary, regime, fraction)
    manifest = read(view['provenance']['manifest_path'])
    units = pd.read_parquet(manifest['source_metadata']['path'], columns=['unit']).unit
    indices = np.load(manifest['indices']['path'], allow_pickle=False)
    rows, selected_units = len(indices), int(units.iloc[indices].nunique())
    expected = dict(rows=rows, units=selected_units, row_fraction=rows/len(units),
        unit_fraction=selected_units/units.nunique())
    if any(view['provenance']['statistics'][key] != value for key, value in expected.items()):
        raise ValueError('Reported curve axes differ from actual selected rows or units')
    return view


def series(points):
    if [p['fraction'] for p in points] != list(FRACTIONS):
        raise ValueError('Need all four curve fractions in registered order')
    return dict(points=points, adjacent=[dict(from_fraction=a['fraction'], to_fraction=b['fraction'],
        **segment(a['metrics'], b['metrics'], a['training_statistics'], b['training_statistics']))
        for a, b in zip(points, points[1:])])


def scalar_curves(summaries, statistics):
    if set(summaries) != set(FRACTIONS) or set(statistics) != {'A', 'B'}:
        raise ValueError('Need four summaries and both actual-size axes')
    selected, same_lr = {}, {}
    def point(fraction, regime, cell, lr):
        return dict(fraction=fraction, metrics=cell['metrics'], coverage=cell['coverage'],
            job_id=cell['job_id'], checkpoint=cell['checkpoint'], learning_rate_multiplier=lr,
            training_statistics=statistics[regime][curve_plan.fraction_name(fraction)])
    for rule in controlled_summary.RULES:
        selected[rule] = {}
        for split in ('val', 'test'):
            selected[rule][split] = {}
            for scorer in selected_analysis.SCORERS:
                selected[rule][split][scorer] = {}
                for regime in ('A', 'B'):
                    views = {str(seed): series([point(f, regime,
                        summaries[f]['selected'][rule][split][scorer][regime]['points'][str(seed)],
                        summaries[f]['selected_learning_rate_multiplier']) for f in FRACTIONS])
                        for seed in controlled_summary.SEEDS}
                    means = []
                    for a, b in zip(views['43']['points'], views['44']['points']):
                        means.append(dict(fraction=a['fraction'], training_statistics=a['training_statistics'],
                            learning_rate_multiplier=a['learning_rate_multiplier'],
                            metrics={k: (a['metrics'][k]+b['metrics'][k])/2
                                if a['metrics'][k] is not None and b['metrics'][k] is not None else None for k in METRICS},
                            coverage_by_seed={'43': a['coverage'], '44': b['coverage']}))
                    spreads = {axis: [controlled_summary.seed_spread({s: views[s]['adjacent'][i]['per_doubling'][axis]
                        for s in views}) for i in range(3)] for axis in ('rows', 'units')}
                    selected[rule][split][scorer][regime] = dict(individual=views,
                        confirmation_mean=series(means), slope_seed_spread=spreads)
    for lr in control_plan.LEARNING_RATE_MULTIPLIERS:
        same_lr[str(lr)] = {split: {scorer: {regime: series([point(f, regime,
            summaries[f]['tuning_candidates'][str(lr)][split][scorer][regime], lr) for f in FRACTIONS])
            for regime in ('A', 'B')} for scorer in selected_analysis.SCORERS} for split in ('val', 'test')}
    return finite(dict(selected=selected, same_lr_seed42=same_lr))


def matched_curve(tables, sizes):
    """One common eligible cohort across all four points; keep every unit."""
    if set(tables) != set(FRACTIONS) or set(sizes) != set(FRACTIONS):
        raise ValueError('Matched slopes need all four registered fractions')
    common = variant_relation.common_cohort(tables)
    ordered = {f: tables[f].sort_values('group_id').reset_index(drop=True).copy() for f in FRACTIONS}
    mask = ordered[FRACTIONS[0]].original_ranking_eligible.to_numpy(dtype=bool, copy=True)
    for frame in ordered.values():
        mask &= np.isfinite(frame.spearman.to_numpy(float))
    for frame in ordered.values():
        frame['spearman'] = frame.spearman.where(mask)
    adjacent = []
    for a, b in zip(FRACTIONS, FRACTIONS[1:]):
        paired = controlled_summary.paired_group_change(ordered[a], ordered[b])
        slopes = segment({MACRO: common['means'][a]}, {MACRO: common['means'][b]}, sizes[a], sizes[b])
        interval = paired['unit_bootstrap']['interval']
        slopes['conditional_unit_slope_interval'] = {axis: [v/d for v in interval]
            if interval is not None and d > 0 else None for axis, d in slopes['log2_size_change'].items()}
        adjacent.append(dict(from_fraction=a, to_fraction=b, paired_change=paired, **slopes))
    return finite(dict(common_cohort=common, adjacent=adjacent,
        interpretation='Same eligible and finite assays across all four fractions, within one regime/seed/rule. Complete held-out units include zero-eligible units. Intervals condition on fitted models and assignments; no seed, subset-redraw or cross-split uncertainty.'))


def draw(curves, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for i, regime in enumerate(('A', 'B')):
        view = curves['selected']['best_macro']['test']['ranking_score'][regime]
        for j, axis in enumerate(('rows', 'units')):
            ax = axes[i, j]
            for seed, color in (('42', '#999999'), ('43', '#5c83b4'), ('44', '#bc784b')):
                points = view['individual'][seed]['points']
                ax.plot([p['training_statistics'][axis] for p in points],
                    [np.nan if p['metrics'][MACRO] is None else p['metrics'][MACRO] for p in points],
                    marker='o', linestyle='--' if seed == '42' else '-', alpha=.6,
                    color=color, label=f'Seed {seed}'+(' (tuning)' if seed == '42' else ''))
            points = view['confirmation_mean']['points']
            ax.plot([p['training_statistics'][axis] for p in points],
                [np.nan if p['metrics'][MACRO] is None else p['metrics'][MACRO] for p in points],
                color='#111111', linewidth=2, marker='s', label='Seeds 43/44 mean')
            ax.set_xscale('log', base=2)
            counts = [p['training_statistics'][axis] for p in points]
            ax.set_xticks(np.geomspace(min(counts), max(counts), 3) if min(counts) < max(counts) else [counts[0]])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:,.0f}'))
            ax.minorticks_off()
            ax.set_xlabel('Actual training '+axis+' (log2 axis)')
            ax.set_title(f'Split {regime}'); ax.grid(alpha=.2)
            if j == 0:
                ax.set_ylabel('Test eligible macro Spearman')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)
    fig.suptitle('Whole-unit learning curves: fixed 15 epochs, each fraction uses its B-selected LR')
    fig.tight_layout(rect=(0, .06, 1, .95))
    for extension in ('png', 'svg'):
        fig.savefig(Path(output)/('learning_curves.'+extension), dpi=170)
    plt.close(fig)


def analyze(paths, endpoint_path):
    curve, primary = evaluation.prepared()
    endpoint = evaluation.shared_endpoint(primary)
    endpoint_path = Path(endpoint_path)
    full, locations = variant_relation.verify_summary(endpoint_path, {'full': {}})
    if (full['summary']['ranking_readout'] != 'ranking_head' or
        full['source']['primary_plan_sha256'] != sha256(WORK/'controlled/plan.json') or
        set(locations) != set(endpoint['job_ids'])):
        raise ValueError('Full endpoint differs from the ten shared primary fits')
    summaries = {1.: full['summary']}
    paths = [Path(p) for p in paths]
    if len(paths) != 30 or len({read(p)['job']['id'] for p in paths}) != 30:
        raise ValueError('Need all thirty distinct smaller-fraction fit analyses')
    records = []
    for path in paths:
        plan, _, _, source = analysis_inputs(read(path)['job']['id'])
        record = verify_record(path, plan, source)
        if source['core_metric_software'] != selected_analysis.software():
            raise ValueError('Curve and primary metric implementation differ')
        records.append(record); locations[record['job']['id']] = (path, record)
    for fraction in FRACTIONS[:-1]:
        chosen = [r for r in records if r['job']['fraction'] == fraction]
        path = WORK/'learning_curves/selections'/(curve_plan.fraction_name(fraction)+'.json')
        decision = dict(read(path), arm='unit_curve_'+curve_plan.fraction_name(fraction))
        summaries[fraction] = controlled_summary.summarize_records(chosen, decision, sha256(path))
    # Revalidate actual sizes/labels even for the shared full endpoint.
    views = {r: {f: size_contract(curve, primary, r, f) for f in FRACTIONS} for r in ('A', 'B')}
    statistics = {r: {curve_plan.fraction_name(f): views[r][f]['provenance']['statistics'] for f in FRACTIONS} for r in views}
    curves = scalar_curves(summaries, statistics)
    paired, cache, table_hashes = {}, {}, {}
    for rule in controlled_summary.RULES:
        paired[rule] = {}
        for split in ('val', 'test'):
            paired[rule][split] = {}
            for regime in ('A', 'B'):
                paired[rule][split][regime] = {}
                for seed in controlled_summary.SEEDS:
                    table_paths = {}
                    for f in FRACTIONS:
                        point = summaries[f]['selected'][rule][split]['ranking_score'][regime]['points'][str(seed)]
                        path, record = locations[point['job_id']]
                        table = path.parent/point['checkpoint']/split/'groups_ranking_score.parquet'
                        table_paths[f] = table; table_hashes[str(table)] = sha256(table)
                    key = tuple(str(table_paths[f]) for f in FRACTIONS)
                    if key not in cache:
                        cache[key] = matched_curve({f: pd.read_parquet(p) for f, p in table_paths.items()},
                            {f: statistics[regime][curve_plan.fraction_name(f)] for f in FRACTIONS})
                    paired[rule][split][regime][str(seed)] = cache[key]
    source = dict(software=software(), curve_plan_sha256=sha256(WORK/'learning_curves/plan.json'),
        primary_plan_sha256=sha256(WORK/'controlled/plan.json'), endpoint_summary_sha256=sha256(endpoint_path),
        analysis_sha256={str(p): sha256(p) for p in paths}, group_table_sha256=table_hashes,
        view_inputs_sha256={p: d for regimes in views.values() for v in regimes.values() for p, d in v['input_sha256'].items()})
    output = WORK/'learning_curve_analysis/reports'/control_plan.digest_json(source)[:16]
    payload = finite(dict(curves=curves, common_cohort_ranking=paired, fraction_summaries=summaries,
        training_statistics=statistics, subset_seed=curve_plan.SUBSET_SEED,
        accounting=dict(smaller_fraction_fits=30, shared_endpoint_fits=10, total_fits=40, new_fits_created=0),
        test_used_for_selection=False, study_complete=False,
        interpretation='Descriptive adjacent slopes on actual rows and units, not extrapolation or a monotonicity test. All fractions/rules/scorers remain visible. Each fraction has its own B-selected LR; same-LR seed42 curves are separate tuning-exposed sensitivities. Three seeds and confirmation-only spread are descriptive, not seed intervals. Fifteen epochs hold duration in passes fixed, not optimizer updates or example exposure. One nested unit permutation is used; composition and subset-local labels can change. B fractions near full row coverage do not measure a low-data regime. Primary full-data training-support strata are not used for subset support.'))
    payload = json.loads(json.dumps(payload))  # Fraction keys have one persisted representation.
    if (output/'complete.json').exists():
        saved = read(output/'complete.json')
        if (saved['source'] != source or saved['payload'] != payload or
            saved['payload_sha256'] != control_plan.digest_json(payload)):
            raise ValueError('Completed curve report changed')
        selected_analysis.verify_artifacts(output, saved)
        return output/'complete.json'
    if output.exists():
        raise RuntimeError('Partial curve report requires inspection')
    output.mkdir(parents=True)
    draw(curves, output)
    immutable(output/'complete.json', dict(source=source, payload=payload,
        payload_sha256=control_plan.digest_json(payload),
        artifacts_sha256={p.name: sha256(p) for p in output.iterdir() if p.is_file()}))
    return output/'complete.json'


def preflight():
    curve, primary = evaluation.prepared()
    return dict(fractions=list(FRACTIONS), actual_training_statistics=curve['statistics'],
        curve_analysis_candidates={j['id']: [str(p) for p in sorted(
            (WORK/'learning_curve_analysis/jobs'/j['id']).glob('*/complete.json'))] for j in curve['tuning_jobs']},
        full_endpoint_summary_candidates=[str(p) for p in sorted((WORK/'controlled_summaries/full').glob('*/complete.json'))],
        required_smaller_fraction_fits=30, required_shared_endpoint_fits=10, new_fits_created=0,
        scientific_curve_created=False, interpretation='Read-only preparation inventory. Paths are not completion evidence; scientific reporting requires all forty verified fits and native selected evaluations.')


if __name__ == '__main__':
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use CPU-only analysis in Atum Prot2Mol')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('preflight')
    job = commands.add_parser('job'); job.add_argument('--job-id', required=True)
    report = commands.add_parser('report')
    report.add_argument('--analyses', nargs='+', type=Path, required=True)
    report.add_argument('--endpoint-summary', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(preflight(), indent=2) if args.action == 'preflight' else
          analyze_job(args.job_id) if args.action == 'job' else analyze(args.analyses, args.endpoint_summary))
