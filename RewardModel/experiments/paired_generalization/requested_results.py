"""Single-seed selected metrics, matched changes and the requested A/B plot.

The amended training scope is fixed before test scores are read. Retain every
requested variant; report conditional whole-unit uncertainty, never seed spread.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256
    from . import requested_evaluation as evaluation, selected_analysis as diagnostics
    from . import encoder_analysis, controlled_summary, variant_relation, control_plan
else:
    from run import ROOT, WORK, sha256
    import requested_evaluation as evaluation, selected_analysis as diagnostics
    import encoder_analysis, controlled_summary, variant_relation, control_plan

read = evaluation.read
immutable = diagnostics.immutable
ARMS = ('full', *evaluation.requested.ARMS)
CONTRASTS = tuple(('full', arm) for arm in ARMS[1:]) + (
    ('pairwise_top', 'pairwise_uniform'), ('pairwise_hard_zero_margin', 'pairwise_noise_soft'))


def software():
    result = diagnostics.software()
    for name in ('requested_results.py', 'requested_evaluation.py', 'requested_single_seed.py',
                 'encoder_analysis.py', 'controlled_summary.py', 'variant_relation.py', 'realization_analysis.py'):
        path = Path(__file__).with_name(name)
        result['source_sha256'][str(path.relative_to(ROOT))] = sha256(path)
    from importlib.metadata import version
    result['packages']['matplotlib'] = version('matplotlib')
    return result


def provenance(identity):
    path = evaluation.destination(identity) / 'plan.json'
    plan, records = evaluation.verify_complete(path)
    design = WORK / 'tier1/similarity_composition/design'
    noise = WORK / 'tier1/replicates/noise_summary.json'
    primary = read(WORK / 'controlled/plan.json')
    if sha256(noise) != primary['noise_summary_sha256']:
        raise ValueError('Training-only noise summary changed')
    rms = read(noise)[plan['job']['regime']]['train']['noise_scale']['rms_replicate_difference']
    if rms != primary['training_noise_rms'][plan['job']['regime']]:
        raise ValueError('Training-only noise calibration changed')
    source = dict(evaluation_plan_sha256=sha256(path),
        evaluation_complete_sha256=sha256(path.parent / 'complete.json'),
        scope_plan_sha256=plan['scope_plan_sha256'], design_sha256=sha256(design / 'complete.json'),
        noise_summary_sha256=sha256(noise), software=software())
    output = WORK / 'requested_analysis' / identity / control_plan.digest_json(source)[:16]
    return plan, records, design, rms, source, output


def analyze(identity):
    import fcntl
    from reward_model.training.config import load_reward_training_config
    plan, records, design, rms, source, output = provenance(identity)
    config = load_reward_training_config(plan['job']['config_path'])
    if config.model.ranking_readout != 'ranking_head':
        raise ValueError('Requested comparisons must retain the original ranking head')
    refs = diagnostics.references(plan, design)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'analysis.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        immutable(output / 'analysis_plan.json', source)
        if (output / 'complete.json').exists():
            verify_analysis(output / 'complete.json'); return output / 'complete.json'
        checkpoint_dir = output / plan['checkpoint']['id']
        if checkpoint_dir.exists():
            raise RuntimeError('Partial requested analysis requires inspection')
        trajectory = encoder_analysis.trajectory(plan['job'])
        statistics, tables = diagnostics.diagnose_checkpoint(records, refs,
            regime=plan['job']['regime'], noise_rms=rms, ranking_readout=config.model.ranking_readout)
        for name, table in tables.items():
            path = checkpoint_dir / (name + '.parquet'); path.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(path, index=False)
        immutable(output / 'native_trajectory.json', trajectory)
        if provenance(identity)[4] != source:
            raise ValueError('Selected analysis evidence changed while calculating metrics')
        for path, digest in trajectory['evidence_sha256'].items():
            if sha256(Path(path)) != digest:
                raise ValueError('Native trajectory changed during reporting')
        result = dict(phase='requested_single_seed_analysis_complete', source=source,
            job=plan['job'], rules=plan['rules'], checkpoints={plan['checkpoint']['id']: statistics},
            native_trajectory=trajectory, seed=42, learning_rate_multiplier=1.,
            test_used_for_selection=False, seed_uncertainty_available=False,
            artifacts_sha256={str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
                if p.is_file() and p.name not in ('analysis.lock', 'analysis_plan.json')},
            interpretation='Selected metrics and all15 native validation/probe epochs for one fixed fit. Whole-unit intervals condition on this seed/configuration and empirical composition weights. No fit, new selection, seed variance or completed adaptation/gradient/geometry claim.')
        result['payload_sha256'] = control_plan.digest_json(result)
        immutable(output / 'complete.json', result)
        return output / 'complete.json'


def verify_analysis(path):
    path = Path(path); result = read(path)
    plan, _, design, _, source, canonical = provenance(result['job']['id'])
    if (path != canonical / 'complete.json' or result['source'] != source or
        result['phase'] != 'requested_single_seed_analysis_complete' or result['job'] != plan['job'] or
        result['rules'] != plan['rules'] or set(result['checkpoints']) != {plan['checkpoint']['id']} or
        result['seed'] != 42 or result['learning_rate_multiplier'] != 1. or
        result['seed_uncertainty_available'] or result['test_used_for_selection']):
        raise ValueError('Requested analysis scope or provenance changed')
    if result['payload_sha256'] != control_plan.digest_json({k: v for k, v in result.items() if k != 'payload_sha256'}):
        raise ValueError('Requested analysis payload changed')
    diagnostics.similarity_composition.verify_design(design)
    diagnostics.verify_artifacts(path.parent, result)
    for name, digest in result['native_trajectory']['evidence_sha256'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('Analyzed native trajectory input changed')
    return result


def comparisons(records):
    """All requested arms, on matched within-regime groups; no winner filter."""
    expected = {(a, r) for a in ARMS for r in ('A', 'B')}
    if set(records) != expected:
        raise ValueError('Need exactly all 26 requested fit analyses')
    protocols, cohort_by_regime, tables, metrics = set(), {}, {}, {}
    for (arm, regime), (path, record) in records.items():
        job = record['job']
        if ((job['arm'], job['regime'], job['seed'], job['learning_rate_multiplier']) != (arm, regime, 42, 1.) or
            set(record['rules']) != {'best_macro'}):
            raise ValueError('Comparison is not the fixed requested seed/configuration')
        protocols.add(control_plan.digest_json({k: record['source'][k] for k in
            ('scope_plan_sha256', 'design_sha256', 'noise_summary_sha256', 'software')}))
        cohort = record['native_trajectory']['cohorts']
        if cohort_by_regime.setdefault(regime, cohort) != cohort:
            raise ValueError('Requested variants used different native validation/probe cohorts')
        checkpoint = record['rules']['best_macro']['checkpoint']
        metrics[arm, regime] = {}
        for split in ('val', 'test'):
            metrics[arm, regime][split] = {}
            for scorer in diagnostics.SCORERS:
                metrics[arm, regime][split][scorer] = controlled_summary.metric_point(record, 'best_macro', split, scorer)
                table_path = path.parent / checkpoint / split / f'groups_{scorer}.parquet'
                tables[arm, regime, split, scorer] = pd.read_parquet(table_path)
    if len(protocols) != 1:
        raise ValueError('Compared metrics use different protocols')
    changes = {}
    for first, second in CONTRASTS:
        label = second + '_minus_' + first
        changes[label] = {r: {s: {scorer: controlled_summary.paired_group_change(
            tables[first, r, s, scorer], tables[second, r, s, scorer]) for scorer in diagnostics.SCORERS}
            for s in ('val', 'test')} for r in ('A', 'B')}
    relations, cohorts = {}, {}
    for split in ('val', 'test'):
        cohorts[split] = {regime: variant_relation.common_cohort({a: tables[a, regime, split, 'ranking_score'] for a in ARMS})
            for regime in ('A', 'B')}
        own = [dict(arm=a, **{r: metrics[a, r][split]['ranking_score']['metrics'][variant_relation.MACRO] for r in ('A', 'B')}) for a in ARMS]
        common = [dict(arm=a, **{r: cohorts[split][r]['means'][a] for r in ('A', 'B')}) for a in ARMS]
        relations[split] = dict(own_defined=variant_relation.association(own),
            common_defined=variant_relation.association(common))
    return diagnostics.finite_json(dict(
        metrics={a: {r: metrics[a, r] for r in ('A', 'B')} for a in ARMS},
        matched_changes=changes, variant_relations=relations, common_cohort=cohorts,
        uncertainty='Whole-assay intervals on A and whole-component intervals on B, conditional on seed42/configuration. Paired effects use mutually defined eligible assays. No seed interval, significance test or causal A/B inference.',
        interpretation='All13 requested/reference settings remain visible. Contrastive detachment, encoder freedom and ranking controls use fixed rates. Uniform-vs-top and noise-soft-vs-hard comparisons have their own matched controls. A/B relations describe different populations across this selected variant collection; they do not establish a universal law.'))


def draw(result, destination):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size': 9, 'svg.hashsalt': 'requested-fixed-seed-variants'}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
        for ax, view in zip(axes, ('own_defined', 'common_defined')):
            relation = result['variant_relations']['test'][view]
            for i, point in enumerate(relation['points'], 1):
                if point['A'] is None or point['B'] is None: continue
                ax.scatter(point['A'], point['B'], s=35, marker='*' if point['arm'] == 'full' else 'o')
                ax.annotate(str(i), (point['A'], point['B']), xytext=(4, 3), textcoords='offset points', fontsize=7)
            value = relation['statistics']['spearman']
            label = 'undefined' if value is None else f'{value:.3f}'
            ax.set(title=f'{view.replace("_", " ")} assays; across-variant rho={label}',
                xlabel='Split A test macro Spearman', ylabel='Split B test macro Spearman')
            ax.grid(alpha=.15)
        fig.suptitle('Requested variants: seed42, fixed base LR, best validation-macro checkpoints\nDescriptive association; numbers map to variant_index.json', fontsize=10)
        for suffix in ('.png', '.svg'):
            fig.savefig(destination.with_suffix(suffix), dpi=160, metadata={'Date': None} if suffix == '.svg' else None)
        plt.close(fig)


def summarize(paths):
    paths = [Path(p) for p in paths]
    if len(paths) != 26 or len(set(paths)) != 26:
        raise ValueError('Need exactly 26 distinct requested analysis paths')
    records = {}
    for path in paths:
        record = verify_analysis(path); key = record['job']['arm'], record['job']['regime']
        if key in records: raise ValueError('Duplicate requested fit analysis')
        records[key] = path, record
    source = dict(analyses_sha256={str(p): sha256(p) for p in paths}, software=software())
    output = WORK / 'requested_summary' / control_plan.digest_json(source)[:16]
    if (output / 'complete.json').exists():
        result = read(output / 'complete.json')
        if result['source'] != source: raise ValueError('Requested summary provenance changed')
        diagnostics.verify_artifacts(output, result); return output / 'complete.json'
    if output.exists(): raise RuntimeError('Partial requested summary requires inspection')
    result = comparisons(records); output.mkdir(parents=True)
    immutable(output / 'variant_index.json', {str(i): a for i, a in enumerate(ARMS, 1)})
    draw(result, output / 'A_vs_B')
    for path, digest in source['analyses_sha256'].items():
        if sha256(Path(path)) != digest: raise ValueError('Selected analysis changed during summary')
    immutable(output / 'complete.json', dict(phase='requested_fixed_seed_comparison_complete',
        source=source, result=result, training_fits=26, new_fits=0,
        seed_uncertainty_available=False, test_used_for_selection=False,
        artifacts_sha256={p.name: sha256(p) for p in output.iterdir() if p.is_file()},
        remaining='Adaptation, representation geometry, gradient/transference and allocation experiments remain separate; this report is not full-study completion.'))
    return output / 'complete.json'


if __name__ == '__main__':
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use CPU-only analysis inside Atum Prot2Mol')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('analyze').add_argument('--job-id', required=True)
    sub.add_parser('summarize').add_argument('--analyses', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    print(analyze(args.job_id) if args.action == 'analyze' else summarize(args.analyses))
