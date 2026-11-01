"""Freeze row-exact views for the three dependent B held-out allocations.

Training support is identical. Only validation/test roles change. These maps do
not fit or select a model and do not make the allocations independent samples.
The original immutable control grid is supplemented, never rewritten.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256
    from .control_plan import _write_immutable, token_cache_hashes
else:
    from run import ROOT, WORK, sha256
    from control_plan import _write_immutable, token_cache_hashes

IDENTITY = ['group_id', 'target_chembl_id', 'compound_id']
LABELS = ['pchembl_value', 'binary_label']
RAW_COLUMNS = ['assay_id', 'target_chembl_id', 'compound_id', *LABELS]


def example_frame(frame):
    result = frame.copy()
    if 'group_id' not in result:
        result['group_id'] = result.target_chembl_id.astype(str) + '__' + result.assay_id.astype(str)
    for name in IDENTITY:
        result[name] = result[name].astype(str)
    for name in LABELS:
        result[name] = result[name].astype(np.float32)
        if not np.isfinite(result[name]).all():
            raise ValueError('Nonfinite source labels')
    if result.duplicated(IDENTITY).any():
        raise ValueError('Ambiguous duplicate example identity')
    return result.reset_index(drop=True)


def align_rows(reference, candidate, *, check_binary=True):
    """Map candidate order to reference without assuming raw/cache row ordering."""
    reference, candidate = example_frame(reference), example_frame(candidate)
    locations = pd.MultiIndex.from_frame(reference[IDENTITY]).get_indexer(
        pd.MultiIndex.from_frame(candidate[IDENTITY]))
    if len(locations) != len(reference) or np.any(locations < 0) or len(np.unique(locations)) != len(reference):
        raise ValueError('Example identities are not a complete permutation of the reference')
    for name in LABELS if check_binary else ['pchembl_value']:
        if not np.array_equal(reference[name].to_numpy()[locations], candidate[name].to_numpy()):
            raise ValueError(f'Example labels disagree: {name}')
    return locations


def make_views(source, raw_partitions, token_frames, source_indices, unit_map):
    """Pure row alignment and whole-unit checks; no model scores are inputs."""
    source = example_frame(source)
    unit_map = unit_map[['assay_id', 'component']].copy()
    if unit_map.assay_id.duplicated().any():
        raise ValueError('Each assay needs exactly one indivisible unit')
    layouts = []
    for split in ('val', 'test'):
        original_indices = np.asarray(source_indices['original_seed42'][split])
        if original_indices.dtype.kind not in 'iu' or np.any(original_indices < 0) or np.any(original_indices >= len(source)):
            raise ValueError('Invalid original source indices')
        reference = source.iloc[original_indices].reset_index(drop=True)
        raw = example_frame(raw_partitions[split])
        tokens = example_frame(token_frames[split])
        # B's pair-median binary labels need not equal raw observation labels.
        # Continuous labels and identities must still match the original source.
        align_rows(reference, raw, check_binary=False)
        align_rows(raw, tokens)
        location = align_rows(reference, tokens, check_binary=False)
        tokens = tokens[IDENTITY + LABELS].copy()
        tokens['assay_id'] = reference.assay_id.to_numpy()[location]
        tokens['source_index'] = original_indices[location]
        tokens['original_split'] = split
        tokens['original_row_index'] = np.arange(len(tokens))
        tokens['component'] = tokens.assay_id.map(unit_map.set_index('assay_id').component)
        if tokens.component.isna().any():
            raise ValueError('Unmapped held-out component')
        tokens['component'] = tokens.component.astype(np.int64)
        layouts.append(tokens)
    rows = pd.concat(layouts, ignore_index=True)
    if rows.source_index.duplicated().any() or rows.duplicated(IDENTITY).any():
        raise ValueError('Original validation and test rows overlap')
    universe = set(rows.source_index.tolist())
    views, counts = {}, {}
    for realization, indices in source_indices.items():
        if set(indices) != {'val', 'test'}:
            raise ValueError('Each realization needs validation and test roles')
        role = np.full(len(rows), '', dtype='<U4')
        covered = set()
        counts[realization] = {}
        for split in ('val', 'test'):
            values = np.asarray(indices[split])
            if values.dtype.kind not in 'iu' or len(np.unique(values)) != len(values):
                raise ValueError('Source indices must be distinct integers')
            members = set(values.tolist())
            if covered & members or not members <= universe:
                raise ValueError('Alternative allocation overlaps roles or changes held-out support')
            covered |= members
            positions = np.flatnonzero(rows.source_index.isin(members))
            role[positions] = split
            views[f'{realization}_{split}'] = positions
            counts[realization][split] = dict(rows=len(positions),
                units=int(rows.iloc[positions].component.nunique()),
                groups=int(rows.iloc[positions].group_id.nunique()))
        if covered != universe:
            raise ValueError('Alternative allocation omits held-out observations')
        check = rows[['component', 'group_id']].assign(role=role)
        if (check.groupby('component').role.nunique() != 1).any() or (check.groupby('group_id').role.nunique() != 1).any():
            raise ValueError('Alternative allocation splits an indivisible unit or group')
    if len({tuple(views[f'{name}_val']) for name in source_indices}) != len(source_indices):
        raise ValueError('Realizations are not distinct')
    return rows, views, counts


def load_contract(directory=None):
    directory = Path(directory or WORK/'controlled/realizations')
    contract_path = directory/'plan.json'
    contract = json.loads(contract_path.read_text())
    for name, digest in contract['artifacts_sha256'].items():
        if Path(name).name != name or sha256(directory/name) != digest:
            raise ValueError('Realization layout artifact changed')
    if sha256(Path(contract['control_plan_path'])) != contract['control_plan_sha256']:
        raise ValueError('Parent control plan changed')
    if sha256(Path(contract['assignment_audit_path'])) != contract['assignment_audit_sha256']:
        raise ValueError('Assignment audit changed after maps were frozen')
    rows = pd.read_parquet(directory/'rows.parquet')
    with np.load(directory/'views.npz', allow_pickle=False) as handle:
        views = {name: handle[name] for name in handle.files}
    return contract, rows, views


def prepare():
    from datasets import load_from_disk
    if __package__:
        from .prepare_random import archive_directory
    else:
        from prepare_random import archive_directory
    if ROOT != Path('/home/atabey/Prot2Mol'):
        raise RuntimeError('Realization preparation is restricted to Atum Prot2Mol')
    archive_directory(WORK/'realization_predictions')
    directory = WORK/'controlled/realizations'
    if (directory/'plan.json').exists():
        return load_contract(directory)[0]
    plan_path = WORK/'controlled/plan.json'
    plan = json.loads(plan_path.read_text())
    audit_path = WORK/'tier1/split_realizations/complete.json'
    audit = json.loads(audit_path.read_text())
    if audit['training_identical'] is not True or len(audit['realizations']) != 3:
        raise ValueError('Three allocations with unchanged training support are required')
    source_path = ROOT/'dataset/processed/chembl_37/protein_cluster_50_activity_balanced/all.parquet'
    if sha256(source_path) != audit['source_sha256']:
        raise ValueError('Scaffold assignments belong to different source data')
    units_path = WORK/'tier1/data/assay_units.parquet'
    if sha256(units_path) != audit['unit_map_sha256']:
        raise ValueError('Indivisible units changed')
    source_indices, assignments = {}, {}
    for item in audit['realizations']:
        stem = f"{item['policy']}_source_indices_seed{item['seed']}"
        name = f"{'original' if item['policy'] == 'original' else 'equal_size'}_seed{item['seed']}"
        source_indices[name] = {}
        component_path = audit_path.parent/f"{item['policy']}_components_seed{item['seed']}.parquet"
        if sha256(component_path) != item['assignment_sha256']:
            raise ValueError('Component assignment changed')
        assignments[name] = item
        for split in ('val', 'test'):
            path = audit_path.parent/f'{stem}_{split}.npy'
            if sha256(path) != item['source_indices_sha256'][split]:
                raise ValueError('Assignment source indices changed')
            source_indices[name][split] = np.load(path, allow_pickle=False)
    raw, tokens, fingerprints = {}, {}, {}
    data = plan['bases']['B']['data']
    for split in ('val', 'test'):
        path = Path(data[f'{split}_parquet_path'])
        if sha256(path) != plan['source_data_sha256']['B'][split]:
            raise ValueError('B source partition changed')
        raw[split] = pd.read_parquet(path, columns=RAW_COLUMNS)
        cache = Path(data['tokenized_dataset_dir'])/f'{split}_examples'
        if token_cache_hashes(cache) != plan['tokenized_data_sha256']['B'][split]:
            raise ValueError('B tokenized partition changed')
        ds = load_from_disk(str(cache))
        fingerprints[split] = ds._fingerprint
        tokens[split] = ds.select_columns(IDENTITY + LABELS).to_pandas()
    rows, views, counts = make_views(pd.read_parquet(source_path, columns=RAW_COLUMNS), raw, tokens,
                                     source_indices, pd.read_parquet(units_path))
    for name in counts:
        for split in ('val', 'test'):
            expected = assignments[name]['partitions'][split]
            if any(counts[name][split][key] != expected[key] for key in ('rows', 'units')):
                raise ValueError('Verified views disagree with assignment counts')
    directory.mkdir(parents=True, exist_ok=True)
    # These payloads are not authoritative until the immutable plan is written.
    rows.to_parquet(directory/'rows.parquet', index=False)
    np.savez_compressed(directory/'views.npz', **views)
    contract = dict(format_version=1, control_plan_path=str(plan_path), control_plan_sha256=sha256(plan_path),
        assignment_audit_path=str(audit_path), assignment_audit_sha256=sha256(audit_path),
        source_sha256=audit['source_sha256'], units_sha256=audit['unit_map_sha256'],
        artifacts_sha256={name: sha256(directory/name) for name in ('rows.parquet', 'views.npz')},
        training_identical=True, realizations=counts, assignments=assignments,
        tokenized_dataset_dir=data['tokenized_dataset_dir'], dataset_fingerprints=fingerprints,
        inference=dict(batch_size=32, autocast='bfloat16', scalar_exports_only=True),
        checkpoint_selection='Each allocation uses only its own validation macro Spearman; earliest epoch breaks ties; defined coverage is explicit. Original primary selection stays the Trainer selection. Uniform-inference allocation analysis is reported separately.',
        hyperparameter_selection='Each allocation selects its own LR from the same three candidates. Confirmation must include the union of the winning LRs; the 280-fit primary plan is not assumed to cover this union.',
        test_policy='Export scalar predictions without test metrics; freeze allocation-specific checkpoint and LR choices before scoring its test roles.',
        limitations='Dependent held-out reallocations with identical training support and fixed scaffold geometry; not independent split draws or estimates of training-support variation.',
        coverage='Macro-Spearman selection across allocations. Allocation-specific aggregate-loss/averaging comparisons still need their own full loss evaluation.',
        preparation_code_sha256=sha256(Path(__file__)))
    _write_immutable(directory/'plan.json', json.dumps(contract, indent=2, sort_keys=True)+'\n')
    return load_contract(directory)[0]


if __name__ == '__main__':
    result = prepare()
    print(json.dumps({key: result[key] for key in ('realizations', 'training_identical', 'limitations')}, indent=2))
