"""E58/Q17 primary-grid component: descriptive A/B relations across variants.

Requires all 28 existing primary-arm summaries. E40/E41 are separate unfinished
inputs to the MD's all-Tier-3 plot; this component never declares E58 complete.
No fit, LR choice, method selection, significance test or predictive claim.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

if __package__:
    from .run import ROOT, WORK, sha256
    from . import controlled_summary, control_plan
else:
    from run import ROOT, WORK, sha256
    import controlled_summary, control_plan

read = controlled_summary.read
immutable = controlled_summary.immutable
METRICS = (*controlled_summary.METRICS, *(f'eligible_ndcg_at_{k}' for k in (1, 3, 5)))
MACRO = 'original_eligible_macro_spearman'
VIEWS = ('seed42_tuning_exposed', 'seed43_confirmation', 'seed44_confirmation', 'confirmation_mean')


def software():
    result = controlled_summary.software()
    for name in ('variant_relation.py', 'control_plan.py'):
        path = Path(__file__).with_name(name)
        result['source_sha256'][str(path.relative_to(ROOT))] = sha256(path)
    try:
        result['variant_plot_matplotlib'] = importlib.metadata.version('matplotlib')
    except importlib.metadata.PackageNotFoundError:
        result['variant_plot_matplotlib'] = None
    return result


def validate_grid(primary):
    arms = control_plan.control_arms(); definitions = {a['name']: a for a in arms}
    if (primary['arms'] != arms or primary['gpu_uuids'] != list(control_plan.GPU_UUIDS) or
        primary['learning_rate_multipliers'] != list(control_plan.LEARNING_RATE_MULTIPLIERS) or
        primary['tuning_seed'] != 42 or primary['confirmation_seeds'] != [43, 44]):
        raise ValueError('Primary variants, devices or matched budget changed')
    expected = {control_plan.job_id(a, r, 42, lr) for a in definitions for r in ('A', 'B')
                for lr in control_plan.LEARNING_RATE_MULTIPLIERS}
    jobs = primary['tuning_jobs']
    if len(jobs) != len(expected) or {j['id'] for j in jobs} != expected:
        raise ValueError('Need every registered primary tuning fit exactly once')
    for job in jobs:
        if (job['seed'] != 42 or job['phase'] != 'tuning' or
            job['id'] != control_plan.job_id(job['arm'], job['regime'], 42, job['learning_rate_multiplier'])):
            raise ValueError('Primary fit identity changed')
        expected_config = control_plan.make_config(primary['bases'][job['regime']], definitions[job['arm']],
            regime=job['regime'], seed=42, multiplier=job['learning_rate_multiplier'],
            output=job['config']['training']['output_dir'], training_noise_rms=primary['training_noise_rms'])
        if job['config'] != expected_config:
            raise ValueError('Primary variant changes an unregistered parameter')
    return definitions


def association(points):
    """Equal weight per declared variant; undefined points remain explicit."""
    if len({p['arm'] for p in points}) != len(points):
        raise ValueError('A variant cannot be counted twice')
    finite = lambda v: v is not None and math.isfinite(v)
    used = [p for p in points if finite(p['A']) and finite(p['B'])]
    result = dict(variants=len(points), defined_pairs=len(used),
        excluded_variants=[p['arm'] for p in points if p not in used],
        pearson=None, spearman=None, slope_B_on_A=None, intercept_B_on_A=None, in_sample_r_squared=None)
    if len(used) < 2:
        return dict(points=points, statistics=result)
    x, y = [np.array([p[r] for p in used], dtype=float) for r in ('A', 'B')]
    dx, dy = x-x.mean(), y-y.mean(); xx, yy = float(dx@dx), float(dy@dy)
    if xx > 0:
        slope = float(dx@dy/xx)
        result.update(slope_B_on_A=slope, intercept_B_on_A=float(y.mean()-slope*x.mean()))
    if xx > 0 and yy > 0:
        corr = float(np.clip(dx@dy/math.sqrt(xx*yy), -1, 1))
        rx, ry = rankdata(x), rankdata(y); rx -= rx.mean(); ry -= ry.mean()
        result.update(pearson=corr, spearman=float(np.clip(rx@ry/math.sqrt(float(rx@rx)*float(ry@ry)), -1, 1)),
            in_sample_r_squared=corr*corr)
    return dict(points=points, statistics=result)


def seed_views(points):
    if set(points) != {'42', '43', '44'}:
        raise ValueError('All three seed roles are required')
    arms = [p['arm'] for p in points['42']]
    if any([p['arm'] for p in rows] != arms for rows in points.values()):
        raise ValueError('Seed views must retain every variant in the same order')
    means = []
    for a, b in zip(points['43'], points['44']):
        value = {r: (a[r]+b[r])/2 if a[r] is not None and b[r] is not None else None for r in ('A', 'B')}
        means.append(dict(arm=a['arm'], **value,
            coverage_by_confirmation_seed={str(s): p['coverage'] for s, p in ((43, a), (44, b))}))
    return {**{view: association(points[str(seed)]) for seed, view in zip((42, 43, 44), VIEWS)},
            'confirmation_mean': association(means)}


def scalar_relations(summaries, definitions):
    selected = {}; candidates = {}
    for rule in controlled_summary.RULES:
        selected[rule] = {}
        for split in ('val', 'test'):
            selected[rule][split] = {}
            for metric in METRICS:
                points = {}
                for seed in controlled_summary.SEEDS:
                    rows = []
                    for arm in definitions:
                        cell = summaries[arm]['selected'][rule][split]['ranking_score']
                        pair = {r: cell[r]['points'][str(seed)] for r in ('A', 'B')}
                        rows.append(dict(arm=arm, **{r: pair[r]['metrics'][metric] for r in pair},
                            coverage={r: pair[r]['coverage'] for r in pair}))
                    points[str(seed)] = rows
                selected[rule][split][metric] = seed_views(points)
    for lr in control_plan.LEARNING_RATE_MULTIPLIERS:
        candidates[str(lr)] = {}
        for split in ('val', 'test'):
            candidates[str(lr)][split] = {}
            for metric in METRICS:
                rows = []
                for arm in definitions:
                    cell = summaries[arm]['tuning_candidates'][str(lr)][split]['ranking_score']
                    rows.append(dict(arm=arm, **{r: cell[r]['metrics'][metric] for r in ('A', 'B')},
                        coverage={r: cell[r]['coverage'] for r in ('A', 'B')}))
                candidates[str(lr)][split][metric] = association(rows)
    return dict(selected=selected, same_lr_seed42=candidates)


def common_cohort(tables):
    """Same eligible, defined assays across variants, within one regime/seed."""
    keys = ['group_id', 'component', 'size', 'original_ranking_eligible']; reference = None; arrays = {}
    for arm, frame in tables.items():
        if frame.group_id.duplicated().any():
            raise ValueError('Duplicate assay in variant comparison')
        frame = frame.sort_values('group_id').reset_index(drop=True)
        if reference is None:
            reference = frame[keys]
        elif not reference.equals(frame[keys]):
            raise ValueError('Variant cohorts, units or eligibility differ')
        arrays[arm] = frame.spearman.to_numpy(dtype=float)
    eligible = reference.original_ranking_eligible.to_numpy(dtype=bool)
    common = eligible.copy()
    for values in arrays.values():
        common &= np.isfinite(values)
    return dict(groups=len(reference), units=int(reference.component.nunique()),
        eligible_groups=int(eligible.sum()), common_defined_eligible_groups=int(common.sum()),
        own_defined_eligible_groups={arm: int((eligible & np.isfinite(v)).sum()) for arm, v in arrays.items()},
        means={arm: float(v[common].mean()) if common.any() else None for arm, v in arrays.items()})


def verify_summary(path, definitions):
    path = Path(path); result = read(path); arm = result['summary']['arm']
    if arm not in definitions:
        raise ValueError('Unregistered primary variant')
    for name, digest in result['source']['analysis_sha256'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('Primary analysis changed')
    canonical = controlled_summary.summarize(arm, sorted(result['source']['analysis_sha256']))
    if canonical.resolve() != path.resolve() or read(canonical) != result:
        raise ValueError('Use the complete verified canonical primary summary')
    records = [(Path(p), read(Path(p))) for p in result['source']['analysis_sha256']]
    return result, {r['job']['id']: (p, r) for p, r in records}


def draw(relations, definitions, destination, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    families = list(dict.fromkeys(a['family'] for a in definitions.values()))
    colors = {f: plt.get_cmap('tab10')(i) for i, f in enumerate(families)}
    visible = [p for view in relations.values() for p in view['points']
               if p['A'] is not None and p['B'] is not None]
    limits = {}
    for regime in ('A', 'B'):
        values = [p[regime] for p in visible]
        low, high = (min(values), max(values)) if values else (-1., 1.)
        pad = max(.02, .08*(high-low)); limits[regime] = (low-pad, high+pad)
    with plt.rc_context({'font.size': 9, 'svg.hashsalt': 'paired-generalization-variants'}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout='constrained', sharex=True, sharey=True)
        for ax, view in zip(axes.flat, VIEWS):
            record = relations[view]
            for index, point in enumerate(record['points'], 1):
                if point['A'] is None or point['B'] is None:
                    continue
                ax.scatter(point['A'], point['B'], s=28, color=colors[definitions[point['arm']]['family']])
                ax.annotate(str(index), (point['A'], point['B']), xytext=(3, 3), textcoords='offset points', fontsize=6)
            stats = record['statistics']
            if stats['slope_B_on_A'] is not None:
                x = np.array(limits['A']); ax.plot(x, stats['intercept_B_on_A']+stats['slope_B_on_A']*x, color='0.45', lw=.8)
            label = 'undefined' if stats['spearman'] is None else f"{stats['spearman']:.3f}"
            ax.set_title(view.replace('_', ' ') + f"; n={stats['defined_pairs']}\nAcross-variant Spearman: {label}", fontsize=10)
            ax.set(xlabel='Split A test macro Spearman', ylabel='Split B test macro Spearman',
                xlim=limits['A'], ylim=limits['B']); ax.grid(alpha=.15)
        handles = [plt.Line2D([], [], marker='o', ls='', color=colors[f], label=f.replace('_', ' ')) for f in families]
        fig.legend(handles=handles, loc='outside lower center', ncol=3, fontsize=8)
        fig.suptitle(title+'\nB-selected LR per variant; best-macro checkpoints. Descriptive fits; numbers map to variant_index.json.', fontsize=11)
        for suffix in ('.png', '.svg'):
            fig.savefig(destination.with_suffix(suffix), dpi=160, metadata={'Date': None} if suffix == '.svg' else None)
        plt.close(fig)


def analyze(paths):
    primary_path = WORK/'controlled/plan.json'; primary = read(primary_path); definitions = validate_grid(primary)
    paths = [Path(p) for p in paths]
    if len(paths) != len(definitions) or {read(p)['summary']['arm'] for p in paths} != set(definitions):
        raise ValueError('Need all 28 primary summaries, without filtering by scores')
    code = software(); hashes = {str(primary_path): sha256(primary_path)}; summaries = {}; analyses = {}
    qualification = None; protocols = set()
    for path in paths:
        hashes[str(path)] = sha256(path); result, records = verify_summary(path, definitions)
        summary = result['summary']; arm = summary['arm']
        readout = definitions[arm]['model'].get('ranking_readout', primary['bases']['B']['model']['ranking_readout'])
        if (result['source']['primary_plan_sha256'] != hashes[str(primary_path)] or
            result['source']['software'] != controlled_summary.software() or summary['ranking_readout'] != readout or
            set(records) != set(controlled_summary.expected_jobs(arm, summary['selected_learning_rate_multiplier']))):
            raise ValueError('Primary plan, metric, configured readout or fit coverage changed')
        summaries[arm] = summary; analyses[arm] = records
        hashes.update(result['source']['analysis_sha256'])
        for _, record in records.values():
            protocols.add(control_plan.digest_json({k: record['source'][k] for k in ('software', 'design_sha256', 'noise_summary_sha256')}))
            runtime_path = Path(record['job']['output_dir'])/'runtime_qualification.json'; runtime = read(runtime_path)
            if (runtime['world_size'] != 4 or runtime['signature'] != control_plan.digest_json(
                dict(sources=runtime['sources'], plan_sha256=hashes[str(primary_path)]))):
                raise ValueError('Invalid training qualification')
            if qualification is None:
                qualification = runtime
                for name, digest in runtime['sources'].items():
                    if not (ROOT/name).resolve().is_relative_to(ROOT.resolve()) or sha256(ROOT/name) != digest:
                        raise ValueError('Qualified training source changed')
                    hashes[str(ROOT/name)] = digest
            elif runtime != qualification:
                raise ValueError('Training implementations differ across variants')
            hashes[str(runtime_path)] = sha256(runtime_path)
    if len(protocols) != 1:
        raise ValueError('Metric, composition or noise protocols differ across variants')
    source = dict(input_sha256=hashes, software=code)
    target = WORK/'variant_relation'/control_plan.digest_json(source)[:16]/'complete.json'
    if target.exists():
        saved = read(target)
        if saved['source'] != source or saved['payload_sha256'] != control_plan.digest_json(saved['result']):
            raise ValueError('Completed variant report changed')
        for name, digest in saved['result']['artifact_sha256'].items():
            if sha256(Path(name)) != digest:
                raise ValueError('Variant comparison artifact changed')
        return target
    result = scalar_relations(summaries, definitions); common = {}; table_hashes = {}
    for rule in controlled_summary.RULES:
        common[rule] = {}
        for split in ('val', 'test'):
            points = {}; coverage = {}
            for seed in controlled_summary.SEEDS:
                cohorts = {}
                for regime in ('A', 'B'):
                    tables = {}
                    for arm in definitions:
                        lr = summaries[arm]['selected_learning_rate_multiplier']
                        path, record = analyses[arm][control_plan.job_id(arm, regime, seed, lr)]
                        table = path.parent/record['rules'][rule]['checkpoint']/split/'groups_ranking_score.parquet'
                        table_hashes[str(table)] = sha256(table); tables[arm] = pd.read_parquet(table)
                    cohorts[regime] = common_cohort(tables)
                coverage[str(seed)] = {r: {k: v for k, v in c.items() if k != 'means'} for r, c in cohorts.items()}
                points[str(seed)] = [dict(arm=arm, **{r: cohorts[r]['means'][arm] for r in cohorts},
                    coverage={r: dict(eligible_groups=cohorts[r]['eligible_groups'],
                        common_defined_eligible_groups=cohorts[r]['common_defined_eligible_groups']) for r in cohorts}) for arm in definitions]
            common[rule][split] = dict(relations=seed_views(points), coverage=coverage)
    target.parent.mkdir(parents=True, exist_ok=True)
    index = {str(i): dict(arm=a, family=definitions[a]['family'], ranking_readout=summaries[a]['ranking_readout'],
        selected_learning_rate_multiplier=summaries[a]['selected_learning_rate_multiplier']) for i, a in enumerate(definitions, 1)}
    immutable(target.parent/'variant_index.json', index)
    draw(result['selected']['best_macro']['test'][MACRO], definitions, target.parent/'primary_selected_test', 'Primary variants: original eligible cohorts')
    draw(common['best_macro']['test']['relations'], definitions, target.parent/'primary_common_cohort_test', 'Primary variants: common defined eligible cohorts')
    artifacts = {str(p): sha256(p) for p in target.parent.iterdir() if p.name == 'variant_index.json' or p.suffix in ('.png', '.svg')}
    result.update(common_cohort_sensitivity=common, variant_index=index,
        artifact_sha256={**table_hashes, **artifacts},
        accounting=dict(primary_variants=len(definitions), existing_primary_fits=10*len(definitions), new_fits_created=0),
        missing_all_tier3_families=['E40 leak-in/out', 'E41 learning curves'], e58_complete=False, study_complete=False,
        test_used_for_selection=False, new_fits_created=0,
        interpretation='Primary-grid component only; the MD all-Tier-3 plot also requires E40/E41. All registered primary settings remain visible with their configured readouts and own B-validation-selected LRs. Seed42 is tuning-exposed; seed43/44 and their two-seed means are separate. The three same-LR seed42 relations are separate sensitivity views, not pooled extra observations. Original metrics retain per-fit defined coverage; common-cohort macro sensitivity intersects eligible, defined assays across all variants within each regime/seed/rule/partition and reports losses of coverage. Common cohorts can differ across seeds/rules. A/B populations differ and partition roles can overlap. Variants share data, references and tuning choices; Pearson, rank correlation, OLS slopes and R-squared are descriptive in-sample associations, not independent-sample tests, causal effects, out-of-sample predictions, or proof that tuning on A transfers to B. No association confidence intervals or p-values are claimed. No test score selects variants, readouts, LRs or rules.')
    for name, digest in {**hashes, **result['artifact_sha256']}.items():
        if sha256(Path(name)) != digest:
            raise ValueError('Variant evidence changed during reporting')
    if software() != code:
        raise ValueError('Variant reporting software changed')
    immutable(target, dict(source=source, result=result, payload_sha256=control_plan.digest_json(result)))
    return target


def preflight():
    definitions = validate_grid(read(WORK/'controlled/plan.json'))
    return dict(primary_variants=len(definitions), existing_primary_fits=10*len(definitions),
        summary_candidates={a: [str(p) for p in sorted((WORK/'controlled_summaries'/a).glob('*/complete.json'))] for a in definitions},
        missing_all_tier3_families=['E40 leak-in/out', 'E41 learning curves'],
        matplotlib=software()['variant_plot_matplotlib'], scientific_comparison_created=False,
        new_fits_created=0, e58_complete=False,
        interpretation='Read-only inventory. Paths alone do not prove scientific completion; all 28 canonical summaries and original training/metric evidence are required.')


if __name__ == '__main__':
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use CPU-only reporting in Atum Prot2Mol')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('analyze', 'preflight')); parser.add_argument('--summaries', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps(preflight(), indent=2) if args.action == 'preflight' else analyze(args.summaries))
