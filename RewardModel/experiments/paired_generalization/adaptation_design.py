"""Freeze the MD's E49-E52 assay memberships without model fitting or scores.

Across-unit probes divide original validation units into adaptation/validation
sets and retain original test units. Within-unit probes use fixed support/query
assays inside validation or test units. Neither procedure splits an assay.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .prepare_random import archive_directory
else:
    from run import ROOT, WORK, sha256, write_json
    from prepare_random import archive_directory

SEED = 42
K_VALUES = (1, 2, 4, 8, 16, 64)
SOURCE_COLUMNS = ['assay_id', 'target_chembl_id', 'compound_id', 'pchembl_value']


def priority(namespace, identity, seed=SEED):
    return hashlib.sha256(json.dumps([seed, namespace, str(identity)], separators=(',', ':')).encode()).hexdigest()


def group_inventory(frame, regime, role, units):
    if regime not in ('A', 'B') or role not in ('train', 'val', 'test'):
        raise ValueError('Unknown source regime or partition')
    if frame.empty or frame[SOURCE_COLUMNS].isna().any().any():
        raise ValueError('Expected complete nonempty source observations')
    if frame.duplicated(['assay_id', 'target_chembl_id', 'compound_id']).any():
        raise ValueError('Repeated curated observation identity')
    if frame.groupby('assay_id').target_chembl_id.nunique().max() != 1:
        raise ValueError('One assay must have one target identity')
    labels = frame.pchembl_value.to_numpy(dtype=np.float32).astype(np.float64)
    if not np.isfinite(labels).all():
        raise ValueError('Nonfinite continuous labels')
    data = frame.assign(value=labels)
    groups = data.groupby('assay_id', sort=True).agg(target_chembl_id=('target_chembl_id', 'first'),
        rows=('value', 'size'), low=('value', 'min'), high=('value', 'max')).reset_index()
    groups['group_id'] = groups.target_chembl_id.astype(str)+'__'+groups.assay_id.astype(str)
    if regime == 'A':
        groups['unit'] = 'assay:'+groups.assay_id.astype(str)
    else:
        mapped = groups.assay_id.map(units)
        if mapped.isna().any():
            raise ValueError('Missing scaffold-assay component')
        groups['unit'] = 'component:'+mapped.astype(np.int64).astype(str)
    groups['ranking_eligible'] = (groups.rows >= 3) & (groups.high-groups.low >= .5)
    groups['source_partition'] = role
    return groups.drop(columns=['low', 'high'])


def counts(groups):
    return dict(rows=int(groups.rows.sum()), groups=len(groups), units=int(groups.unit.nunique()),
                ranking_eligible_groups=int(groups.ranking_eligible.sum()))


def build_design(inventories, regime, *, seed=SEED, k_values=K_VALUES):
    if regime not in ('A', 'B'):
        raise ValueError('Unknown source regime')
    if not k_values or tuple(sorted(set(k_values))) != tuple(k_values) or any(k < 1 or int(k) != k for k in k_values):
        raise ValueError('k must be distinct ascending positive whole group counts')
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if set(inventories[left].group_id) & set(inventories[right].group_id):
            raise ValueError('Original assay partitions overlap')
        if set(inventories[left].unit) & set(inventories[right].unit):
            raise ValueError('Original indivisible units cross partitions')
    validation_units = sorted(inventories['val'].unit.unique(), key=lambda u: priority(regime+'/units', u, seed))
    if len(validation_units) < 2:
        raise ValueError('Need at least two original validation units for disjoint adaptation/validation')
    adaptation_units = set(validation_units[:len(validation_units)//2])
    val = inventories['val'].copy()
    val['probe_role'] = np.where(val.unit.isin(adaptation_units), 'adaptation', 'validation')
    test = inventories['test'].assign(probe_role='test')
    groups = pd.concat([val, test], ignore_index=True).sort_values('group_id').reset_index(drop=True)
    groups['adaptation_order'] = -1
    available = groups[groups.probe_role == 'adaptation']
    order = sorted(available.group_id, key=lambda g: priority(regime+'/across', g, seed))
    locations = groups.set_index('group_id').index.get_indexer(order)
    groups.loc[locations, 'adaptation_order'] = np.arange(len(order))
    across = {}
    for k in k_values:
        selected = groups[(groups.adaptation_order >= 0) & (groups.adaptation_order < k)]
        if len(order) < k:
            selected = selected.iloc[:0]
        across[str(k)] = dict(available=len(order) >= k, available_groups=len(order), **counts(selected),
            selected_group_ids=selected.sort_values('adaptation_order').group_id.tolist())
    # Reserve the query assays once, independent of k. Unused support assays
    # never become query data as k changes. Unit selection also uses no labels.
    within_roles = np.full(len(groups), 'not_used', dtype=object)
    within_order = np.full(len(groups), -1, dtype=np.int64)
    positions = dict(zip(groups.group_id, groups.index))
    row_counts = groups.rows.to_numpy()
    rank_eligible = groups.ranking_eligible.to_numpy()
    unit_records = []
    for (role, unit), group in groups[groups.probe_role != 'adaptation'].groupby(['probe_role', 'unit'], sort=True):
        ordered = sorted(group.group_id, key=lambda g: priority(regime+'/within/'+unit, g, seed))
        capacity = len(ordered)//2
        support_ids, query_ids = ordered[:capacity], ordered[capacity:]
        support = np.array([positions[g] for g in support_ids], dtype=np.int64)
        query = np.array([positions[g] for g in query_ids], dtype=np.int64)
        within_roles[support], within_roles[query] = 'support', 'query'
        within_order[support] = np.arange(capacity)
        unit_records.append(dict(probe_role=role, unit=unit, total_groups=len(group),
            support_capacity=capacity, query_groups=len(query_ids), query_rows=int(row_counts[query].sum()),
            query_ranking_eligible_groups=int(rank_eligible[query].sum())))
    groups['within_role'], groups['within_support_order'] = within_roles, within_order
    unit_table = pd.DataFrame(unit_records)
    within = {}
    for k in k_values:
        result = {}
        for role in ('validation', 'test'):
            eligible = unit_table[(unit_table.probe_role == role) & (unit_table.support_capacity >= k)]
            result[role] = dict(units=len(eligible),
                rank_evaluable_units=int((eligible.query_ranking_eligible_groups > 0).sum()),
                query_groups=int(eligible.query_groups.sum()), query_rows=int(eligible.query_rows.sum()),
                query_ranking_eligible_groups=int(eligible.query_ranking_eligible_groups.sum()))
        result['validation_selection_possible'] = result['validation']['rank_evaluable_units'] > 0
        result['test_rank_evaluation_possible'] = result['test']['rank_evaluable_units'] > 0
        within[str(k)] = result
    summary = dict(regime=regime, assignment_seed=seed, k_values=list(k_values),
        unit_definition='whole assay' if regime == 'A' else 'whole scaffold-assay component',
        source_partitions={role:counts(frame) for role,frame in inventories.items()},
        probe_partitions={role:counts(groups[groups.probe_role == role]) for role in ('adaptation','validation','test')},
        across_unit=across, within_unit=within,
        fixed_test_groups_preserved=set(groups[groups.probe_role == 'test'].group_id) == set(inventories['test'].group_id),
        within_unit_limit=('An A unit contains exactly one assay; disjoint support/query assays within a unit are impossible.'
            if regime == 'A' else 'Report feasible and rank-evaluable unit counts at each k. Larger k can change the eligible unit cohort; compare common cohorts as well as available cohorts. Unsupported validation budgets cannot select an adaptation model.'),
        interpretation='Group memberships only. No tokens, prediction scores, model weights or optimizer are used. No adaptation result or training-ready view is established.')
    return groups, unit_table, summary


def main():
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use CPU-only preparation in authorized Atum Prot2Mol')
    directory = WORK/'tier4/adaptation_design'
    if directory.exists():
        raise RuntimeError('Existing adaptation design requires verification before reuse')
    primary_path = WORK/'controlled/plan.json'
    primary = json.loads(primary_path.read_text())
    if sha256(ROOT/'Paired input ranking generalization.md') != primary['spec_sha256']:
        raise ValueError('The governing MD changed')
    unit_path = WORK/'tier1/data/assay_units.parquet'
    assignment_path = WORK/'tier1/split_realizations/complete.json'
    if sha256(unit_path) != json.loads(assignment_path.read_text())['unit_map_sha256']:
        raise ValueError('Frozen indivisible-unit map changed')
    unit_frame = pd.read_parquet(unit_path)
    if unit_frame.assay_id.duplicated().any():
        raise ValueError('Assay-to-unit identity is ambiguous')
    units = unit_frame.set_index('assay_id').component
    sources = {str(primary_path):sha256(primary_path), str(unit_path):sha256(unit_path),
               str(assignment_path):sha256(assignment_path)}
    outputs = {}
    for regime in ('A', 'B'):
        inventories = {}
        for role in ('train', 'val', 'test'):
            path = Path(primary['bases'][regime]['data'][role+'_parquet_path'])
            sources[str(path)] = sha256(path)
            if sources[str(path)] != primary['source_data_sha256'][regime][role]:
                raise ValueError('Frozen source partition changed')
            frame = pd.read_parquet(path, columns=SOURCE_COLUMNS)
            inventories[role] = group_inventory(frame, regime, role, units)
        outputs[regime] = build_design(inventories, regime)
        print(json.dumps(dict(regime=regime, probe_partitions=outputs[regime][2]['probe_partitions'],
                              within_unit=outputs[regime][2]['within_unit'])), flush=True)
    directory.parent.mkdir(parents=True, exist_ok=True)
    archive_directory(directory)
    artifacts = {}
    for regime, (groups, unit_table, _) in outputs.items():
        for name, table in [(f'groups_{regime}.parquet', groups), (f'within_units_{regime}.parquet', unit_table)]:
            table.to_parquet(directory/name, index=False)
            artifacts[name] = sha256(directory/name)
    write_json(directory/'complete.json', dict(format_version=1, experiment_ids=['E49','E50','E51','E52'],
        spec_sha256=primary['spec_sha256'], primary_plan_sha256=sha256(primary_path),
        source_sha256=sources, code_sha256=sha256(Path(__file__)), artifacts_sha256=artifacts,
        regimes={regime:value[2] for regime,value in outputs.items()},
        selection_policy='Hash identities only: half the original validation units for adaptation, half for validation; original test units unchanged. Adaptation k groups are nested whole-assay prefixes. Within-unit support/query halves are fixed before k selection.',
        label_policy='Future training views must recompute BCE pair medians separately within each selected adaptation subset and fixed evaluation set. Never reuse original validation labels pooled across the new adaptation/validation split; continuous labels remain unchanged.',
        fitting_gate='Only group-membership preparation. Complete Tier 3 comparisons and a separately frozen adaptation training/selection protocol, token/label view qualification and runtime qualification before any fitting.',
        exposure_limit='Original validation units were used for base-model checkpoint/LR selection. These are conditional adaptation probes on the existing benchmark, not a new independent test population.',
        baseline_policy='Include the unchanged base model at k=0 on each identical query cohort; zero adaptation requires no fit.',
        test_predictions_read=False, model_fitting=False, model_inference=False))
    return directory


if __name__ == '__main__':
    print(main())
