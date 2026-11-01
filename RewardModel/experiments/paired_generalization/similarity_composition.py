"""MD E08/E09: binary scaffold strata and descriptive composition weighting.

Preparation reads identities/labels only. Scoring uses saved predictions; it
does not fit models or select settings. The same design can score neural runs
once their held-out predictions are available.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .baseline_common import load_partitions, SOURCE_COLUMNS
    from .control_plan import _write_immutable
    from .metrics import size_bin, group_metrics, aggregate_groups
    from .label_precision import POLICY
else:
    from run import ROOT, WORK, sha256, write_json
    from baseline_common import load_partitions, SOURCE_COLUMNS
    from control_plan import _write_immutable
    from metrics import size_bin, group_metrics, aggregate_groups
    from label_precision import POLICY

DESIGN = dict(
    relation='Maximum binary exact scaffold identity over ALL training molecules, irrespective of protein. Preserve pooled acyclic and private unparseable nodes from the splitter.',
    similarity_values=[0, 1],
    whole_group_strata=['all_unseen', 'mixed', 'all_shared'],
    within_stratum='Recompute each assay on rows in one similarity class; require at least three rows and label span >=0.5 again. Report changed cohort size explicitly.',
    size_bins=['3','4-10','11-25','>25'],
    positive_fraction_bins=['0','(0,0.25]','(0.25,0.5]','(0.5,0.75]','(0.75,1)','1'],
    mean_pchembl_bins=['<5','[5,6)','[6,7)','[7,8)','>=8'],
    weights='On headline-eligible groups, A group weight = (B cell count / B eligible groups) / A cell count. Cells use size bin x positive-fraction bin x mean-pChEMBL bin. No clipping or score-dependent cell choice.',
    support='B cells absent in A are unsupported; report their mass and compare only common-support populations. No extrapolation into absent cells.',
    undefined='Undefined Spearman is excluded with both ordinary and weighted defined coverage reported. It is not replaced by zero.',
    inference='Post-hoc descriptive standardization using held-out labels; not a deployable estimator, model selection criterion, or causal estimate. Fixed coarse cells do not exactly match every continuous descriptor.',
    bootstrap='2000 whole-unit draws, seed42. A uses assays, B uses scaffold-assay components. Intervals condition on fitted models, assignments and fixed composition weights; they omit uncertainty in estimating those weights and training/seed variation. A weighting change uses paired A draws. No cross-split gap confidence interval is claimed.',
    label_precision=POLICY,
)


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    _write_immutable(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def descriptors(frame, training_molecules, scaffolds, units):
    """Global training support; no scores or held-out-neighbour fitting."""
    if not scaffolds.index.is_unique or not units.index.is_unique:
        raise ValueError('Scaffold and unit maps must have unique identities')
    training_scaffolds = scaffolds.reindex(pd.Index(training_molecules))
    values = frame.compound_id.map(scaffolds)
    if training_scaffolds.isna().any() or values.isna().any():
        raise ValueError('Missing scaffold identity')
    rows = frame.copy()
    rows['max_similarity'] = values.isin(set(training_scaffolds)).astype(np.uint8)
    rows['molecule_seen'] = rows.compound_id.isin(set(training_molecules))
    rows['unit_id'] = rows.assay_id.map(units)
    if rows.unit_id.isna().any() or not np.isfinite(rows.pchembl_value).all():
        raise ValueError('Missing unit or invalid labels')
    if not rows.binary_label.isin([0, 1]).all():
        raise ValueError('Expected binary partition-local labels')
    rows['unit_id'] = rows.unit_id.astype(np.int64)
    grouped = rows.groupby('group_id', sort=True)
    if grouped.unit_id.nunique().max() != 1:
        raise ValueError('An assay crosses indivisible units')
    groups = grouped.agg(size=('pchembl_value','size'), low=('pchembl_value','min'),
        high=('pchembl_value','max'), label_mean=('pchembl_value','mean'),
        positive_rate=('binary_label','mean'), unit_id=('unit_id','first'),
        min_similarity=('max_similarity','min'), max_similarity=('max_similarity','max')).reset_index()
    groups['label_span'] = groups.high-groups.low
    groups['eligible'] = (groups['size'] >= 3) & (groups.label_span >= .5)
    groups['size_bin'] = groups['size'].map(size_bin)
    groups['similarity_class'] = np.select([groups.max_similarity == 0, groups.min_similarity == 1],
                                         ['all_unseen','all_shared'], default='mixed')
    rate = groups.positive_rate
    groups['positive_bin'] = np.select([rate == 0, rate <= .25, rate <= .5, rate <= .75, rate < 1],
        DESIGN['positive_fraction_bins'][:-1], default='1')
    groups['mean_bin'] = np.asarray(DESIGN['mean_pchembl_bins'])[np.searchsorted([5.,6.,7.,8.], groups.label_mean, side='right')]
    groups['cell'] = groups.size_bin+'|'+groups.positive_bin+'|'+groups.mean_bin
    counts = {str(value):dict(rows=int((rows.max_similarity == value).sum()),
        unique_molecules=int(rows.loc[rows.max_similarity == value,'compound_id'].nunique())) for value in (0,1)}
    summary = dict(rows=len(rows), unique_molecules=int(rows.compound_id.nunique()), groups=len(groups),
        eligible_groups=int(groups.eligible.sum()), similarity=counts,
        exact_molecule_seen_rows=int(rows.molecule_seen.sum()),
        whole_groups={name:dict(groups=int((groups.similarity_class == name).sum()),
            eligible_groups=int(((groups.similarity_class == name) & groups.eligible).sum()))
            for name in DESIGN['whole_group_strata']})
    return rows, groups, summary


def composition_weights(a, b):
    counts = pd.concat([a.loc[a.eligible].groupby('cell').size().rename('A_groups'),
                        b.loc[b.eligible].groupby('cell').size().rename('B_groups')], axis=1).fillna(0).astype(int)
    if not counts.B_groups.sum() or not counts.A_groups.sum():
        raise ValueError('Both populations need headline-eligible groups')
    counts['target_mass'] = counts.B_groups/counts.B_groups.sum()
    counts['common_support'] = (counts.A_groups > 0) & (counts.B_groups > 0)
    counts['A_group_weight'] = np.where(counts.common_support,
        counts.target_mass/counts.A_groups.clip(lower=1), 0.)
    return counts.reset_index()


def verify_design(base):
    plan = read(base/'complete.json')
    if plan['design'] != DESIGN:
        raise ValueError('Saved descriptive design differs from this implementation')
    for path, digest in plan['source_sha256'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Design source changed')
    for name, digest in plan['artifacts_sha256'].items():
        if sha256(base/name) != digest:
            raise ValueError('Prepared design artifact changed')
    return plan


def prepare(base):
    if (base/'complete.json').exists():
        return verify_design(base)
    if base.exists():
        raise RuntimeError('Partial design requires inspection before reuse')
    control_path = WORK/'controlled/plan.json'
    control = read(control_path)
    if sha256(control_path) != read(WORK/'controlled/preparation_complete.json')['plan_sha256']:
        raise ValueError('Primary frozen split plan changed')
    scaffold_path = WORK/'tier1/data/molecule_scaffolds.parquet'
    unit_path = WORK/'tier1/data/assay_units.parquet'
    scaffolds = pd.read_parquet(scaffold_path).set_index('compound_id').scaffold
    unit_table = pd.read_parquet(unit_path).sort_values('assay_id')
    units = dict(A=pd.Series(np.arange(len(unit_table)),index=unit_table.assay_id),
                 B=unit_table.set_index('assay_id').component)
    sources = {str(p):sha256(p) for p in (control_path,scaffold_path,unit_path)}
    base.mkdir(parents=True)
    summaries, tables = {}, {}
    for regime in ('A','B'):
        frames, hashes = load_partitions(regime)
        if hashes != control['source_data_sha256'][regime]:
            raise ValueError('Source partitions differ from the controlled grid')
        for split, digest in hashes.items():
            sources[control['bases'][regime]['data'][f'{split}_parquet_path']] = digest
        training_molecules = frames['train'].compound_id.unique()
        for split in ('val','test'):
            rows, groups, summary = descriptors(frames[split],training_molecules,scaffolds,units[regime])
            if regime == 'B' and rows.max_similarity.any():
                raise ValueError('Scaffold-disjoint B unexpectedly shares a training scaffold')
            name = f'{regime}_{split}'
            rows.to_parquet(base/f'rows_{name}.parquet',index=False)
            groups.to_parquet(base/f'groups_{name}.parquet',index=False)
            summaries[name],tables[name] = summary,groups
    coverage = {}
    for split in ('val','test'):
        cells = composition_weights(tables[f'A_{split}'],tables[f'B_{split}'])
        cells.to_parquet(base/f'weights_{split}.parquet',index=False)
        coverage[split] = dict(target_groups=int(cells.B_groups.sum()),
            supported_target_groups=int(cells.loc[cells.common_support,'B_groups'].sum()),
            supported_target_mass=float(cells.loc[cells.common_support,'target_mass'].sum()),
            unsupported_cells=cells.loc[(cells.B_groups > 0) & ~cells.common_support,'cell'].tolist())
    result = dict(design=DESIGN, source_sha256=sources, composition=summaries, coverage=coverage,
        artifacts_sha256={p.name:sha256(p) for p in sorted(base.glob('*.parquet'))},
        model_scores_used=False, source_code_sha256=sha256(Path(__file__)))
    immutable(base/'complete.json',result)
    return result


def bootstrap_scores(table, weights, all_units, repetitions=2000, seed=42):
    """Fixed weights, same whole-unit draws for each supplied weighting."""
    units = pd.Index(np.unique(all_units))
    if not units.is_unique or not table.unit_id.isin(units).all():
        raise ValueError('Bootstrap units do not cover score rows')
    score = table.spearman.to_numpy(dtype=float)
    finite = np.isfinite(score)
    totals, counts, point = {}, {}, {}
    for name, raw in weights.items():
        raw = np.asarray(raw, dtype=float)
        if raw.shape != score.shape or not np.isfinite(raw).all() or (raw < 0).any():
            raise ValueError('Invalid fixed composition weights')
        w = np.where(finite,raw,0.)
        frame = pd.DataFrame(dict(unit=table.unit_id.to_numpy(),weight=w,value=w*np.nan_to_num(score)))
        by_unit = frame.groupby('unit')[['value','weight']].sum().reindex(units,fill_value=0.)
        totals[name],counts[name] = by_unit.value.to_numpy(),by_unit.weight.to_numpy()
        point[name] = dict(score=float(frame.value.sum()/w.sum()) if w.sum() else None,
            defined_weight_fraction=float(w.sum()/raw.sum()) if raw.sum() else None,
            effective_groups=float(w.sum()**2/(w@w)) if w.sum() else None)
    rng = np.random.default_rng(seed)
    samples = {name:[] for name in weights}
    for start in range(0,repetitions,100):
        draw = rng.integers(len(units),size=(min(100,repetitions-start),len(units)))
        for name in samples:
            denominator = counts[name][draw].sum(axis=1)
            numerator = totals[name][draw].sum(axis=1)
            samples[name].extend(np.divide(numerator,denominator,out=np.full(len(draw),np.nan),where=denominator>0))
    for name, values in samples.items():
        values = np.asarray(values)
        valid = np.isfinite(values)
        point[name].update(interval=np.quantile(values[valid],[.025,.975]).tolist() if valid.any() and len(units)>1 else None,
            defined_repetitions=int(valid.sum()),units=len(units),repetitions=repetitions,seed=seed)
    if 'common' in samples and 'standardized' in samples:
        delta = np.asarray(samples['standardized'])-samples['common']
        delta = delta[np.isfinite(delta)]
        point['standardization_change'] = dict(
            score=point['standardized']['score']-point['common']['score'] if point['standardized']['score'] is not None and point['common']['score'] is not None else None,
            interval=np.quantile(delta,[.025,.975]).tolist() if len(delta) and len(units)>1 else None,
            defined_repetitions=len(delta),interpretation='Paired A-unit draws; conditional on fixed empirical composition weights')
    return point


def score_groups(frame):
    records = []
    for identity, group in frame.groupby('group_id',sort=True):
        record = group_metrics(group.pchembl_value,group.ranking_score)
        record.update(group_id=identity,unit_id=int(group.unit_id.iloc[0]))
        records.append(record)
    return records


def summarize_stratum(records, all_units):
    if not records:
        return dict(groups=0,eligible_groups=0,eligible_defined_fraction=None,macro_spearman=None,bootstrap=None)
    table = pd.json_normalize(records)
    eligible = table.original_ranking_eligible
    bootstrap = bootstrap_scores(table,{'macro':eligible.to_numpy(float)},all_units)['macro']
    return dict(**aggregate_groups(records),eligible_groups=int(eligible.sum()),
        eligible_defined_fraction=float(table.loc[eligible,'spearman'].notna().mean()) if eligible.any() else None,
        bootstrap=bootstrap)


def score_similarity(rows, groups, predictions):
    if len(predictions) != len(rows):
        raise ValueError('Predictions do not cover the fixed partition')
    # Identity and label order is part of the contract, never an implicit join.
    pd.testing.assert_frame_equal(rows[SOURCE_COLUMNS].reset_index(drop=True),
        predictions[SOURCE_COLUMNS].reset_index(drop=True),check_dtype=False,check_exact=True)
    scores = predictions.ranking_score.to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite prediction')
    frame = rows.assign(ranking_score=scores)
    records = score_groups(frame)
    classes = groups.set_index('group_id').similarity_class
    all_units = groups.unit_id.unique()
    whole = {name:summarize_stratum([r for r in records if classes[r['group_id']] == name],all_units)
             for name in DESIGN['whole_group_strata']}
    within = {str(value):summarize_stratum(score_groups(frame[frame.max_similarity == value]),all_units) for value in (0,1)}
    table = groups.merge(pd.DataFrame([dict(group_id=r['group_id'],spearman=r['spearman']) for r in records]),
                         on='group_id',validate='one_to_one')
    return dict(full=summarize_stratum(records,all_units),whole_groups=whole,within_stratum=within),table


def compare_composition(a, b, cells):
    lookup = cells.set_index('cell')
    wa = a.cell.map(lookup.A_group_weight).fillna(0).to_numpy()*a.eligible.to_numpy()
    common_b = b.cell.map(lookup.common_support).fillna(False).to_numpy() & b.eligible.to_numpy()
    a_scores = bootstrap_scores(a,dict(raw=a.eligible.to_numpy(float),common=(wa>0).astype(float),standardized=wa),a.unit_id.unique())
    b_scores = bootstrap_scores(b,dict(raw=b.eligible.to_numpy(float),common=common_b.astype(float)),b.unit_id.unique())
    def mean(frame, weights, column):
        return float(np.average(frame[column],weights=weights)) if np.sum(weights) else None
    balance = {column:dict(A_raw=mean(a,a.eligible,column), B_raw=mean(b,b.eligible,column),
        A_standardized=mean(a,wa,column),B_common=mean(b,common_b,column)) for column in ('size','positive_rate','label_mean','label_span')}
    def difference(left,right):
        return None if left is None or right is None else left-right
    return dict(A=a_scores,B=b_scores,balance=balance,
        raw_A_minus_B=difference(a_scores['raw']['score'],b_scores['raw']['score']),
        common_A_minus_B=difference(a_scores['common']['score'],b_scores['common']['score']),
        standardized_common_A_minus_B=difference(a_scores['standardized']['score'],b_scores['common']['score']),
        target_supported_mass=float(cells.loc[cells.common_support,'target_mass'].sum()),
        interpretation=DESIGN['inference'],bootstrap_scope=DESIGN['bootstrap'])


def analyze_depth3(base):
    design = base/'design'
    plan = verify_design(design)
    output = base/'lambdamart_depth3'
    sources = {'design/complete.json':sha256(design/'complete.json')}
    predictions = {}
    for regime in ('A','B'):
        source = WORK/'tier2/lambdamart'/regime
        completed = read(source/'complete.json')
        if not completed['no_new_training'] or completed['evaluated_settings'] != [dict(max_depth=3,reg_lambda=1.,min_child_weight=1.)]:
            raise ValueError('Expected the user-selected fixed depth-3 baseline')
        for name,digest in completed['artifacts_sha256'].items():
            if sha256(source/name) != digest:
                raise ValueError('Depth-3 evidence changed')
        sources[str(source/'complete.json')] = sha256(source/'complete.json')
        for split in ('val','test'):
            predictions[f'{regime}_{split}'] = pd.read_parquet(source/split/'predictions.parquet')
    if (output/'complete.json').exists():
        result = read(output/'complete.json')
        if result['sources_sha256'] != sources or result['source_code_sha256'] != sha256(Path(__file__)):
            raise ValueError('Completed analysis provenance changed')
        for name,digest in result['artifacts_sha256'].items():
            if sha256(output/name) != digest:
                raise ValueError('Completed analysis artifact changed')
        return result
    if output.exists():
        raise RuntimeError('Partial analysis requires inspection before reuse')
    output.mkdir()
    summaries, tables = {}, {}
    for name,prediction in predictions.items():
        rows = pd.read_parquet(design/f'rows_{name}.parquet')
        groups = pd.read_parquet(design/f'groups_{name}.parquet')
        summaries[name],tables[name] = score_similarity(rows,groups,prediction)
        regime,split = name.split('_')
        expected = read(WORK/'tier2/lambdamart'/regime/'complete.json')['metrics'][split]['original_eligible_macro_spearman']
        observed = summaries[name]['full']['original_eligible_macro_spearman']
        if not np.isclose(observed,expected,atol=1e-14,rtol=0):
            raise ValueError('Stratification failed to reproduce the original full-partition metric')
        tables[name].to_parquet(output/f'groups_{name}.parquet',index=False)
        print(f'Fixed depth-3 strata scored: {name}',flush=True)
    comparisons = {split:compare_composition(tables[f'A_{split}'],tables[f'B_{split}'],
        pd.read_parquet(design/f'weights_{split}.parquet')) for split in ('val','test')}
    result = dict(sources_sha256=sources,design=plan['design'],similarity=summaries,composition=comparisons,
        artifacts_sha256={p.name:sha256(p) for p in output.glob('*.parquet')},
        source_code_sha256=sha256(Path(__file__)),
        scope='Supplementary fixed depth-3 A/B evidence requested by the user. No new training or tuning. Matched neural strata and reweighting, seeds and additional B allocation analyses remain pending.')
    immutable(output/'complete.json',result)
    return result


if __name__ == '__main__':
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','depth3'])
    args = parser.parse_args()
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Run this CPU-only analysis in the authorized Atum project')
    base = WORK/'tier1/similarity_composition'
    base.mkdir(parents=True,exist_ok=True)
    with (base/'analysis.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'prepare':
            result = prepare(base/'design')
        else:
            result = analyze_depth3(base)
        print(json.dumps({'complete':True,'action':args.action,
                          'coverage':result.get('coverage'),'scope':result.get('scope')},indent=2))
