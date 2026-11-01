"""Audit alternative B assignments using the original split algorithm and units.

No new similarity relation, graph, training labels or model is introduced.
This records assignment manifests, not model-performance uncertainty.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, DATA, sha256, write_json
else:
    from run import ROOT, WORK, DATA, sha256, write_json

from reward_model.training.scaffold_component_split import _deal_val_test


def assignment_from_units(frame, units, *, seed, restarts=64, randomize_equal_size_order=False):
    if units.assay_id.duplicated().any():
        raise ValueError('Each assay must have exactly one component')
    work = frame[['assay_id', 'target_chembl_id', 'binary_label']].copy()
    work['component'] = work.assay_id.map(units.set_index('assay_id').component)
    if work.component.isna().any():
        raise ValueError('Missing component assignment')
    work['component'] = work.component.astype(int)
    stats = work.groupby('component').agg(rows=('assay_id','size'), positives=('binary_label','sum'))
    giant = int(stats.rows.idxmax())
    targets = set(work.loc[work.component == giant, 'target_chembl_id'])
    rescued = sorted(int(component) for component, group in work[work.component != giant].groupby('component')
                     if not set(group.target_chembl_id).issubset(targets))
    mapping = {component:'train' for component in [giant, *rescued]}
    pool = stats.drop(index=[giant, *rescued], errors='ignore')
    if randomize_equal_size_order:
        # The subsequent descending-size sort retains the same size priority;
        # a label-blind seeded input permutation changes the order within ties.
        # The greedy objective and its 64-restart budget remain unchanged.
        pool = pool.iloc[np.random.default_rng(seed).permutation(len(pool))]
    mapping.update(_deal_val_test(pool, seed=seed, restarts=restarts))
    assignments = stats.reset_index()
    assignments['split'] = assignments.component.map(mapping)
    return assignments, work.component.map(mapping).to_numpy()


def compare_assignments(reference, candidate):
    merged = reference[['component','rows','split']].merge(
        candidate[['component','rows','split']], on=['component','rows'], suffixes=('_reference','_candidate'), validate='one_to_one')
    if len(merged) != len(reference) or len(merged) != len(candidate):
        raise ValueError('Realizations use different indivisible units')
    changed = merged.split_reference != merged.split_candidate
    return dict(training_units_identical=bool(((merged.split_reference == 'train') == (merged.split_candidate == 'train')).all()),
                changed_units=int(changed.sum()), changed_rows=int(merged.loc[changed,'rows'].sum()),
                heldout_rows=int(merged.loc[merged.split_reference != 'train','rows'].sum()))


def main():
    source = ROOT/'dataset/processed/chembl_37/protein_cluster_50_activity_balanced/all.parquet'
    cache = json.loads((WORK/'tier1/data/scaffold_cache.json').read_text())
    if sha256(source) != cache['source_sha256']:
        raise ValueError('Original scaffold-unit cache belongs to different source data')
    frame = pd.read_parquet(source, columns=['assay_id','target_chembl_id','binary_label'])
    units_path = WORK/'tier1/data/assay_units.parquet'
    units = pd.read_parquet(units_path)
    original = {}
    for split in ('train','val','test'):
        groups = pd.read_parquet(DATA/'scaffold_seed42'/f'{split}.parquet', columns=['assay_id'])
        for assay in groups.assay_id.unique():
            if assay in original:
                raise ValueError('Existing B split contains overlapping assays')
            original[assay] = split
    original_rows = frame.assay_id.map(original).to_numpy()
    if pd.isna(original_rows).any():
        raise ValueError('Original split does not cover source assays')
    destination = WORK/'tier1/split_realizations'
    destination.mkdir(parents=True, exist_ok=True)
    reference, results, original_results = None, [], []
    cases = [('original', seed, False) for seed in (42,43,44)] + [('randomized_equal_size_order', seed, True) for seed in (43,44)]
    for policy, seed, randomize in cases:
        assignments, roles = assignment_from_units(frame, units, seed=seed, randomize_equal_size_order=randomize)
        if policy == 'original' and seed == 42:
            if not np.array_equal(roles, original_rows):
                raise ValueError('Cached-unit reconstruction does not reproduce the actual seed-42 split')
            reference = assignments
        else:
            if not np.array_equal(roles == 'train', original_rows == 'train'):
                raise ValueError('Alternative split unexpectedly changed training observations')
        path = destination/f'{policy}_components_seed{seed}.parquet'
        assignments.to_parquet(path,index=False)
        counts = {split: dict(units=int((assignments.split == split).sum()),
                             rows=int(assignments.loc[assignments.split == split, 'rows'].sum()),
                             raw_positive_rate=float(assignments.loc[assignments.split == split,'positives'].sum()/assignments.loc[assignments.split == split,'rows'].sum()))
                  for split in ('train','val','test')}
        # These source-row indices preserve whole units and original source order.
        hashes = {}
        for split in ('val','test'):
            indices = destination/f'{policy}_source_indices_seed{seed}_{split}.npy'
            np.save(indices, np.flatnonzero(roles == split), allow_pickle=False)
            hashes[split] = sha256(indices)
        result = dict(seed=seed, policy=policy, assignment_sha256=sha256(path), source_indices_sha256=hashes,
                      partitions=counts, versus_seed42=compare_assignments(reference, assignments))
        if policy == 'original':
            original_results.append(result)
            if seed == 44:
                write_json(destination/'original_seed_audit.json', dict(source_sha256=cache['source_sha256'], results=original_results,
                    identical_assignments=len({item['assignment_sha256'] for item in original_results}) == 1,
                    interpretation='Changing only the seed can leave the deterministic sorted candidate as the winner; identical assignments are not distinct realizations.'))
        if seed == 42 or randomize:
            results.append(result)
        print(json.dumps(result),flush=True)
    if len({result['assignment_sha256'] for result in results}) != 3:
        raise ValueError('The planned tie-breaking variants did not yield three distinct realizations')
    write_json(destination/'complete.json', dict(source_sha256=cache['source_sha256'],
        unit_map_sha256=sha256(units_path), realizations=results, original_seed_audit=original_results,
        splitter_sha256=sha256(ROOT/'RewardModel/reward_model/training/scaffold_component_split.py'),
        audit_code_sha256=sha256(ROOT/'RewardModel/experiments/paired_generalization/split_realization_audit.py'),
        training_identical=True,
        interpretation='The splitter fixes training membership before validation/test allocation. Additional realizations use seeded input ordering before the same descending-size sort and 64-restart balance objective; only the priority among equal-size components changes. Training fits may be shared if all training settings are identical; each realization must perform its own validation-only checkpoint and LR selection. These are dependent held-out reallocations, not variation in training support or split geometry.',
        remaining='All-checkpoint predictions on the held-out union and selection/evaluation per realization; multi-seed controlled models'))


if __name__ == '__main__':
    main()
