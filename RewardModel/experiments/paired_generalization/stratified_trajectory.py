"""Q11: localize scalar validation trajectories in fixed size/scaffold strata.

Saved predictions only; no training, inference, test scoring or model selection.
This audit localizes BCE and ranking metrics, not listwise/contrastive losses.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

if __package__:
    from .run import ROOT, WORK, sha256
    from .control_plan import digest_json
    from .realization_protocol import align_rows
    from .label_precision import trainer_label_frame, POLICY
    from .similarity_composition import verify_design, immutable
    from .native_analysis import native_frame
else:
    from run import ROOT, WORK, sha256
    from control_plan import digest_json
    from realization_protocol import align_rows
    from label_precision import trainer_label_frame, POLICY
    from similarity_composition import verify_design, immutable
    from native_analysis import native_frame

SIZE_BINS = ('1', '2', '3', '4-10', '11-25', '>25')
SIMILARITY = ('all_shared', 'mixed', 'all_unseen')


def read(path):
    return json.loads(Path(path).read_text())


def masks(groups):
    return {'all': np.ones(len(groups), bool),
            **{f'size/{key}': groups.size_bin.eq(key).to_numpy() for key in SIZE_BINS},
            **{f'scaffold/{key}': groups.similarity_class.eq(key).to_numpy() for key in SIMILARITY}}


def score(frame, rows, design_groups):
    """Reconstruct row BCE and average-rank correlations without prior metrics."""
    frame = trainer_label_frame(frame)
    locations = align_rows(rows, frame)
    frame = frame.iloc[np.argsort(locations)].reset_index(drop=True)
    if (not frame.binary_label.isin([0, 1]).all() or
            not np.isfinite(frame[['activity_logits', 'ranking_score']].to_numpy()).all()):
        raise ValueError('Expected finite scores and binary labels')
    logits = frame.activity_logits.to_numpy(float)
    frame['bce'] = np.logaddexp(0., logits) - frame.binary_label.to_numpy(float)*logits
    frame['abs_logit'] = np.abs(logits)
    grouped = frame.groupby('group_id', sort=True)
    ranks = grouped[['pchembl_value', 'ranking_score']].rank(method='average')
    centered = ranks - ranks.groupby(frame.group_id).transform('mean')
    frame['cross'] = centered.pchembl_value * centered.ranking_score
    frame['label_square'] = centered.pchembl_value**2
    frame['score_square'] = centered.ranking_score**2
    observed = frame.groupby('group_id', sort=True).agg(
        size=('pchembl_value', 'size'), low=('pchembl_value', 'min'), high=('pchembl_value', 'max'),
        bce_sum=('bce', 'sum'), abs_logit_sum=('abs_logit', 'sum'),
        cross=('cross', 'sum'), label_square=('label_square', 'sum'), score_square=('score_square', 'sum'))
    groups = design_groups.set_index('group_id').sort_index().copy()
    if not observed.index.equals(groups.index) or not np.array_equal(observed['size'], groups['size']):
        raise ValueError('Fixed assay identities or sizes differ')
    eligible = (observed['size'] >= 3) & (observed.high-observed.low >= .5)
    if not np.array_equal(eligible, groups.eligible):
        raise ValueError('Fixed float32-label eligibility differs')
    denominator = np.sqrt(observed.label_square*observed.score_square)
    groups['spearman'] = (observed.cross/denominator.where(denominator > 0)).where(eligible)
    groups['bce_sum'] = observed.bce_sum
    groups['abs_logit_sum'] = observed.abs_logit_sum
    groups = groups.reset_index()
    summaries = {}
    for key, mask in masks(groups).items():
        chosen = groups.loc[mask]
        selected = frame[frame.group_id.isin(chosen.group_id)]
        defined = chosen.spearman.notna()
        count = int(chosen['size'].sum())
        summaries[key] = dict(rows=count, groups=len(chosen), units=int(chosen.unit_id.nunique()),
            eligible_groups=int(chosen.eligible.sum()), defined_groups=int(defined.sum()),
            row_bce=float(chosen.bce_sum.sum()/count) if count else None,
            assay_macro_bce=float((chosen.bce_sum/chosen['size']).mean()) if len(chosen) else None,
            mean_absolute_logit=float(chosen.abs_logit_sum.sum()/count) if count else None,
            macro_spearman=float(chosen.spearman.mean()) if defined.any() else None,
            roc_auc=float(roc_auc_score(selected.binary_label, selected.activity_logits))
                if selected.binary_label.nunique() == 2 else None)
    return summaries, groups


def paired_changes(first, last, repetitions=2000, seed=42):
    """Same unit draws for both endpoints; undefined ranks use matched groups."""
    first, last = (x.sort_values('group_id').reset_index(drop=True) for x in (first, last))
    fixed = ['group_id', 'size', 'eligible', 'unit_id', 'size_bin', 'similarity_class']
    pd.testing.assert_frame_equal(first[fixed], last[fixed], check_exact=True)
    units = pd.Index(sorted(first.unit_id.unique()))
    if not len(units):
        raise ValueError('No validation units')
    output = {}
    for key, mask in masks(first).items():
        matched = mask & first.spearman.notna().to_numpy() & last.spearman.notna().to_numpy()
        table = pd.DataFrame(dict(unit_id=first.unit_id,
            bce_delta=np.where(mask, last.bce_sum-first.bce_sum, 0.),
            row_count=np.where(mask, first['size'], 0),
            rho_delta=np.where(matched, last.spearman-first.spearman, 0.),
            rho_count=matched.astype(int)))
        by_unit = table.groupby('unit_id').sum().reindex(units, fill_value=0.)
        values = by_unit.to_numpy(float)
        draws = {'row_bce': [], 'matched_macro_spearman': []}
        rng = np.random.default_rng(seed)
        for start in range(0, repetitions, 100):
            indices = rng.integers(len(units), size=(min(100, repetitions-start), len(units)))
            sums = values[indices].sum(axis=1)
            for name, col in [('row_bce', 0), ('matched_macro_spearman', 2)]:
                draws[name].extend(np.divide(sums[:, col], sums[:, col+1],
                    out=np.full(len(sums), np.nan), where=sums[:, col+1] > 0))
        record = dict(rows=int(table.row_count.sum()), matched_groups=int(matched.sum()),
            all_partition_units=len(units), stratum_units=int(first.loc[mask, 'unit_id'].nunique()))
        totals = values.sum(axis=0)
        for name, col in [('row_bce', 0), ('matched_macro_spearman', 2)]:
            finite = np.asarray(draws[name]); finite = finite[np.isfinite(finite)]
            record[name] = dict(change=float(totals[col]/totals[col+1]) if totals[col+1] else None,
                interval=np.quantile(finite, [.025, .975]).tolist() if len(finite) and len(units)>1 else None,
                defined_repetitions=len(finite), repetitions=repetitions, seed=seed)
        output[key] = record
    return output


def summarize(records, tables, horizon=15):
    epochs = [r['epoch'] for r in records]
    if not records or epochs != list(range(1, len(records)+1)) or len(records) > horizon:
        raise ValueError('Expected a contiguous observed epoch prefix')
    minimum = min(records, key=lambda r: r['strata']['all']['row_bce'])['epoch']
    maximum = max(records, key=lambda r: r['strata']['all']['macro_spearman'])['epoch']
    starts = sorted({1, minimum, maximum})
    comparisons = {f'{epoch}_to_{epochs[-1]}': paired_changes(tables[epoch], tables[epochs[-1]]) for epoch in starts}
    extrema = {}
    for key in records[0]['strata']:
        points = [r for r in records if r['strata'][key]['rows']]
        rank = [r for r in points if r['strata'][key]['macro_spearman'] is not None]
        extrema[key] = dict(bce_min_epoch=min(points, key=lambda r: r['strata'][key]['row_bce'])['epoch'] if points else None,
            macro_max_epoch=max(rank, key=lambda r: r['strata'][key]['macro_spearman'])['epoch'] if rank else None)
    return dict(epochs=epochs, horizon=horizon, complete_horizon=len(records)==horizon,
        overall_bce_min_epoch=minimum, overall_macro_max_epoch=maximum,
        stratum_extrema=extrema, paired_endpoint_changes=comparisons, trajectory=records,
        scope='Validation only, scalar BCE and ranking metrics. Native listwise/contrastive loss localization remains pending.',
        interpretation='Descriptive observed-prefix extrema. Paired whole-unit intervals condition on this fit, allocation, fixed strata and observed endpoint choice; they omit seed and endpoint-selection uncertainty and are not adjusted for multiple strata. No causal or final-optimum inference from a prefix. B has no shared-scaffold stratum under the predefined binary relation.')


def run(a_trajectory):
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Run CPU-only in the authorized Atum project')
    base = WORK/'tier1'
    design = base/'similarity_composition/design'
    verify_design(design)
    sources = {}

    def checked(path, expected=None):
        path = Path(path)
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Artifact changed: {path}')
        sources[str(path)] = digest
        return path

    checked(design/'complete.json')
    a_trajectory = Path(a_trajectory)
    if a_trajectory.parent.parent.name != 'full__A__seed42__lr1p0' or not a_trajectory.name.startswith('trajectory_through_epoch'):
        raise ValueError('Use an immutable verified A reference prefix')
    a = read(checked(a_trajectory))
    verification = read(checked(a_trajectory.with_name(a_trajectory.name.replace('trajectory_', 'verification_'))))
    if verification['trajectory_sha256'] != sha256(a_trajectory) or not verification['source_and_artifact_hashes_verified']:
        raise ValueError('A prefix verification differs')
    analysis_plan = read(checked(a_trajectory.parent/'analysis_plan.json', a['analysis_plan_sha256']))
    references = read(checked(a_trajectory.parent/'references/complete.json', analysis_plan['reference_sha256']))
    expected = pd.read_parquet(checked(a_trajectory.parent/'references/eval.parquet', references['references']['eval']['sha256']))
    b_summary = read(checked(base/'baseline_trajectory_summary_v2.json'))
    for name, digest in b_summary['sources_sha256'].items():
        checked(WORK/name, digest)
    candidates = {'A': [], 'B': []}
    for item in a['trajectory']:
        snapshot = item['eval']
        for name, digest in snapshot['source']['provenance']['software']['source_sha256'].items():
            checked(ROOT/name, digest)
        manifest = checked(snapshot['source']['manifest'], snapshot['source']['manifest_sha256'])
        frame, info = native_frame(manifest, expected, references['references']['eval']['fingerprint'], 'eval')
        checked(manifest.parent/info['predictions_file'], snapshot['source']['payload_sha256'])
        candidates['A'].append((int(item['epoch']), frame, info['metrics']['eval_macro_spearman'], info['metrics']['eval_classification_loss']))
    for item in b_summary['rows']:
        epoch = item['epoch']; name = f'epoch{epoch:02d}_val'
        metrics = read(checked(base/'metrics_float32'/f'{name}.json'))
        export = read(checked(base/'predictions'/name/'complete.json'))
        if export['split'] != 'val' or export['checkpoint_sha256'] != metrics['checkpoint_sha256'] or metrics['label_precision'] != POLICY:
            raise ValueError('B validation export provenance differs')
        path = checked(base/'predictions'/name/'predictions.parquet', metrics['predictions_sha256'])
        frame = pd.read_parquet(path)
        if len(frame) != export['rows'] or not np.array_equal(frame.row_index, np.arange(len(frame))):
            raise ValueError('B prediction coverage differs')
        candidates['B'].append((epoch, frame, item['exported_validation_macro'], item['native_validation']['eval_classification_loss']))
    code = {str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__))}
    from importlib.metadata import version
    plan = dict(sources_sha256=sources, code_sha256=code, label_precision=POLICY,
        packages={name: version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn')})
    output = base/'stratified_trajectories'/digest_json(plan)[:16]
    if output.exists():
        raise RuntimeError('Existing audit directory requires inspection; never overwrite evidence')
    output.mkdir(parents=True)
    immutable(output/'plan.json', plan)
    results = {}
    for regime, snapshots in candidates.items():
        rows = pd.read_parquet(design/f'rows_{regime}_val.parquet')
        groups = pd.read_parquet(design/f'groups_{regime}_val.parquet')
        records, tables = [], {}
        for epoch, frame, macro, bce in snapshots:
            stats, table = score(frame, rows, groups)
            np.testing.assert_allclose(stats['all']['macro_spearman'], macro, atol=1e-14, rtol=0)
            # Native torch float32 BCE reduction and export float64 arithmetic can differ slightly.
            np.testing.assert_allclose(stats['all']['row_bce'], bce, atol=1e-5, rtol=0)
            table.to_parquet(output/f'{regime}_epoch{epoch:02d}_groups.parquet', index=False)
            tables[epoch] = table
            records.append(dict(epoch=epoch, strata=stats, native_bce=bce, exported_macro_reference=macro))
            print(json.dumps(dict(regime=regime, epoch=epoch, bce=stats['all']['row_bce'], macro=macro)), flush=True)
        results[regime] = summarize(records, tables)
    immutable(output/'complete.json', dict(plan_sha256=sha256(output/'plan.json'), results=results,
        test_predictions_read=False, model_fitting=False,
        artifacts_sha256={p.name: sha256(p) for p in output.glob('*.parquet')}))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a-trajectory', type=Path, required=True)
    print(run(parser.parse_args().a_trajectory), flush=True)
