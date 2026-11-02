"""Statistical aggregation for the frozen E49-E52 query cohorts.

This library neither reads model weights nor selects epochs or starts jobs.
Callers must verify query artifacts, selected epochs, base/model identities and
the frozen views before supplying their assay records and group inventories.
Here we check the statistical cohort contract, retain undefined assays, and
reuse the study's whole-unit bootstrap. This is not a scientific job receipt.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .adaptation_protocol import SEEDS
from .adaptation_query import SCORERS
from .analyze_predictions import unit_bootstrap
from .control_plan import digest_json
from .controlled_summary import paired_group_change, seed_spread
from .selected_analysis import finite_json


def group_inventory(groups):
    columns = ['group_id', 'unit', 'rows', 'ranking_eligible']
    groups = groups[columns].copy().sort_values('group_id').reset_index(drop=True)
    if (groups.group_id.duplicated().any() or groups[columns].isna().any().any()
            or not all(isinstance(v, str) and v for v in groups.group_id)
            or not all(isinstance(v, str) and v for v in groups.unit)
            or not all(type(v) in (bool, np.bool_) for v in groups.ranking_eligible)
            or not all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
                       and v > 0 for v in groups.rows)):
        raise ValueError('Expected unique whole assays with units, sizes and boolean eligibility')
    if ((groups.rows < 3) & groups.ranking_eligible).any():
        raise ValueError('An assay smaller than three cannot be ranking eligible')
    return groups


def assay_tables(records, groups):
    """Convert native query-worker rows without dropping undefined/ineligible assays."""
    groups = group_inventory(groups)
    expected = groups.set_index('group_id')
    indexed = {scorer: {} for scorer in SCORERS}
    for record in records:
        scorer, group = record['scorer'], record['group_id']
        if scorer not in indexed or group not in expected.index or group in indexed[scorer]:
            raise ValueError('Unexpected or duplicate query assay/pathway')
        frozen = expected.loc[group]
        if (record['unit'] != frozen.unit or record['rows'] != frozen.rows
                or type(record['eligible']) is not bool or type(record['defined']) is not bool
                or record['eligible'] != frozen.ranking_eligible
                or (record['defined'] and not record['eligible'])):
            raise ValueError('Query unit, whole-assay size or eligibility changed')
        rho = record['rho']
        if record['defined']:
            if (isinstance(rho, bool) or not isinstance(rho, (int, float))
                    or not math.isfinite(rho) or not -1.000000000001 <= rho <= 1.000000000001):
                raise ValueError('Defined query correlation must be finite and in [-1, 1]')
        elif rho is not None:
            raise ValueError('Undefined query correlation must remain explicit null')
        indexed[scorer][group] = float(rho) if record['defined'] else np.nan
    result = {}
    for scorer, values in indexed.items():
        if set(values) != set(expected.index):
            raise ValueError('Every query assay must appear in every pathway')
        frame = groups.rename(columns=dict(unit='component', rows='size',
                                          ranking_eligible='original_ranking_eligible')).copy()
        frame['spearman'] = frame.group_id.map(values).astype(float)
        result[scorer] = frame
    return result


def point(table):
    eligible = table.original_ranking_eligible
    defined = eligible & np.isfinite(table.spearman)
    return finite_json(dict(groups=len(table), units=int(table.component.nunique()),
        eligible_groups=int(eligible.sum()), defined_groups=int(defined.sum()),
        eligible_fraction=float(eligible.mean()) if len(table) else None,
        eligible_defined_fraction=float(defined.sum()/eligible.sum()) if eligible.sum() else None,
        macro_spearman=table.loc[defined, 'spearman'].mean(),
        unit_bootstrap=unit_bootstrap(table.assign(spearman=table.spearman.where(defined)))))


def paired_points(base, adapted):
    return dict(baseline=point(base), adapted=point(adapted),
                adapted_minus_baseline=paired_group_change(base, adapted))


def compare_budgets(baseline_records, adapted_records, groups_by_k):
    """One regime/seed/role/kind/scope; positive k only, never pool fits or seeds.

    groups_by_k contains the full score-independent available query inventory at
    each supported k. The baseline covers their union. A shared unit must keep
    exactly the same query assays at every k where it is available. For within-
    unit probes, concatenate predictions from separate per-unit fits only after
    the caller verifies their task identities. Their support sets never pool.
    """
    if (not groups_by_k or set(adapted_records) != set(groups_by_k)
            or any(type(k) is not int or k < 1 for k in groups_by_k)):
        raise ValueError('Need exactly the supported positive integer query budgets')
    budgets = sorted(groups_by_k)
    groups = {k:group_inventory(groups_by_k[k]) for k in budgets}
    if any(frame.empty for frame in groups.values()):
        raise ValueError('Unavailable budgets must be reported separately, not as empty scores')
    combined = pd.concat(groups.values(), ignore_index=True)
    if any(part.drop_duplicates().shape[0] != 1 for _,part in combined.groupby('group_id')):
        raise ValueError('Assay identity, unit, size or eligibility changed across budgets')
    union = combined.drop_duplicates().sort_values('group_id').reset_index(drop=True)
    unit_members = {unit:set(frame.group_id) for unit,frame in union.groupby('unit')}
    for frame in groups.values():
        for unit, members in frame.groupby('unit'):
            if set(members.group_id) != unit_members[unit]:
                raise ValueError('A shared unit changed its query assays across budgets')
    base = assay_tables(baseline_records, union)
    adapted = {k:assay_tables(adapted_records[k], groups[k]) for k in budgets}
    common_units = sorted(set.intersection(*(set(frame.unit) for frame in groups.values())))
    common_groups = sorted(union.loc[union.unit.isin(common_units), 'group_id'])
    available, common, common_defined = {}, {}, {}
    for scorer in SCORERS:
        available[scorer], common[scorer], common_defined[scorer] = {}, {}, {}
        for k in budgets:
            ids = set(groups[k].group_id)
            baseline = base[scorer][base[scorer].group_id.isin(ids)].reset_index(drop=True)
            available[scorer][str(k)] = paired_points(baseline, adapted[k][scorer])
        if not common_units:
            common[scorer] = common_defined[scorer] = None
            continue
        baseline = base[scorer][base[scorer].component.isin(common_units)].reset_index(drop=True)
        matched = {k:adapted[k][scorer][adapted[k][scorer].component.isin(common_units)]
                   .sort_values('group_id').reset_index(drop=True) for k in budgets}
        # Mask correlations, not rows: zero-eligible/undefined units remain in resampling.
        mask = baseline.original_ranking_eligible & np.isfinite(baseline.spearman)
        for table in matched.values():
            mask &= np.isfinite(table.spearman)
        fixed_base = baseline.assign(spearman=baseline.spearman.where(mask))
        fixed = {k:table.assign(spearman=table.spearman.where(mask)) for k,table in matched.items()}
        for k in budgets:
            common[scorer][str(k)] = paired_points(baseline, matched[k])
            common_defined[scorer][str(k)] = paired_points(fixed_base, fixed[k])
        common_defined[scorer]['defined_group_ids'] = baseline.loc[mask, 'group_id'].tolist()
    changes = {}
    for lower, upper in zip(budgets, budgets[1:]):
        units = sorted(set(groups[lower].unit) & set(groups[upper].unit))
        changes[f'{lower}->{upper}'] = dict(units=units, pathways={scorer:
            paired_group_change(*(adapted[k][scorer][adapted[k][scorer].component.isin(units)]
                                  .reset_index(drop=True) for k in (lower, upper)))
            for scorer in SCORERS} if units else None)
    return dict(budgets=budgets, query_inventory_sha256=digest_json(union.to_dict('records')),
        available_units={str(k):sorted(set(groups[k].unit)) for k in budgets},
        available_group_ids={str(k):groups[k].group_id.tolist() for k in budgets},
        common_units=common_units, common_group_ids=common_groups,
        available=available, common_units_statistics=common,
        common_defined_statistics=common_defined, adjacent_budget_changes=changes,
        scientific_provenance_verified=False,
        interpretation='Native macro averages assays, not unit means. k=0 is the unchanged base on each exact query cohort. '
        'Available cohorts may differ across k; common-unit results preserve their query identities. '
        'Common-defined sensitivity intersects eligible, defined assays over the base and every k, retaining all units in resampling. '
        'Paired intervals condition on these fitted models, query units and frozen support membership. '
        'They do not measure seed uncertainty, support-redraw uncertainty or a causal across-versus-within effect. '
        'This statistical layer does not verify model provenance, support exposure or epoch selection.')


def summarize_seeds(results):
    """Descriptive seed spread, with seed42 tuning exposure kept explicit."""
    if set(results) != {str(seed) for seed in SEEDS}:
        raise ValueError('Need all three separately aggregated seeds')
    reference = results['42']
    for result in results.values():
        for key in ('budgets', 'query_inventory_sha256', 'available_units', 'available_group_ids',
                    'common_units', 'common_group_ids'):
            if result[key] != reference[key]:
                raise ValueError('Frozen query cohorts differ across seeds')
    summaries = {}
    for cohort in ('available', 'common_units_statistics', 'common_defined_statistics'):
        summaries[cohort] = {}
        for scorer in SCORERS:
            summaries[cohort][scorer] = {}
            for k in reference['budgets']:
                points = {seed:result[cohort][scorer] for seed,result in results.items()}
                if any(value is None for value in points.values()):
                    if not all(value is None for value in points.values()):
                        raise ValueError('Common-unit availability changed across seeds')
                    summaries[cohort][scorer][str(k)] = None
                    continue
                values = {seed:dict(baseline=record[str(k)]['baseline']['macro_spearman'],
                    adapted=record[str(k)]['adapted']['macro_spearman'],
                    paired_change=record[str(k)]['adapted_minus_baseline']['mean_change'])
                    for seed,record in points.items()}
                summaries[cohort][scorer][str(k)] = seed_spread(values)
    return dict(seed_spread=summaries,
        interpretation='Descriptive means, sample SDs and ranges across fixed support/query memberships. '
        'Seed42 is tuning-exposed; seeds43/44 are confirmations. Each seed keeps its own prediction-defined cohort and coverage in the input report. '
        'No seed is treated as an independent query unit; no across-seed bootstrap or model selection is performed.')
