"""Prepare nested whole-unit curves, with subset-local binary supervision."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .control_plan import _write_immutable, make_config, materialize_job, token_cache_hashes
    from .realization_protocol import IDENTITY, LABELS, align_rows
    from .prepare_random import archive_directory
else:
    from run import ROOT, WORK, sha256, write_json
    from control_plan import _write_immutable, make_config, materialize_job, token_cache_hashes
    from realization_protocol import IDENTITY, LABELS, align_rows
    from prepare_random import archive_directory

from reward_model.training.train_views import check_whole_units, subset_pair_labels, indexed_label_view

FRACTIONS = (.1, .25, .5, 1.)
SUBSET_SEED = 42


def fraction_name(value):
    if value not in FRACTIONS:
        raise ValueError('Fraction is outside the registered learning curve')
    return f'fraction{int(value*100):03d}'


def nested_unit_indices(unit_ids, *, fractions=FRACTIONS, seed=SUBSET_SEED):
    units = np.asarray(unit_ids)
    unique = np.unique(units)
    if not len(unique) or tuple(sorted(set(fractions))) != tuple(fractions) or not all(0 < f <= 1 for f in fractions):
        raise ValueError('Nested fractions need nonempty units and distinct ascending fractions in (0,1]')
    order = np.random.default_rng(seed).permutation(unique)
    result = {}
    for fraction in fractions:
        count = max(1, int(math.floor(fraction*len(order))))
        indices = np.flatnonzero(np.isin(units, order[:count]))
        check_whole_units(units, indices)
        result[fraction] = indices
    return result, order


def subset_statistics(frame, indices, labels, heldout_groups):
    selected = frame.iloc[indices]
    units = frame.groupby('unit', sort=True).size()
    training_groups = selected[['group_id','target_chembl_id']].drop_duplicates()
    target_counts = training_groups.groupby('target_chembl_id').size()
    groups = selected.groupby('group_id').pchembl_value.agg(['size','min','max'])
    eligible = (groups['size'] >= 3) & ((groups['max'].astype('float32')-groups['min'].astype('float32')) >= .5)
    return dict(rows=len(selected), row_fraction=len(selected)/len(frame),
        units=int(selected.unit.nunique()), unit_fraction=selected.unit.nunique()/frame.unit.nunique(),
        groups=len(groups), rank_eligible_groups=int(eligible.sum()), targets=int(selected.target_chembl_id.nunique()),
        molecules=int(selected.compound_id.nunique()), positive_rate=float(np.mean(labels)),
        labels_changed_from_full_training=int(np.count_nonzero(labels != selected.binary_label.to_numpy())),
        largest_training_unit_rows=int(units.max()), largest_training_unit_fraction=float(units.max()/len(frame)),
        largest_training_unit_included=bool(units.idxmax() in set(selected.unit)),
        assay_size_quantiles=dict(zip(('min','median','p90','max'), np.quantile(groups['size'], [0,.5,.9,1]).tolist())),
        replicate_mean=float(selected.measurement_count.mean()),
        heldout_target_coverage={role: dict(
            groups_with_seen_target_fraction=float(group.target_chembl_id.isin(target_counts.index).mean()),
            targets_seen=int(group.target_chembl_id.drop_duplicates().isin(target_counts.index).sum()),
            targets_total=int(group.target_chembl_id.nunique()),
            median_training_groups_per_target=float(group.target_chembl_id.map(target_counts).fillna(0).median()))
            for role, group in heldout_groups.items()})


def prepare():
    from datasets import load_from_disk
    from reward_model.training.data import RewardAssayListDataset
    if ROOT != Path('/home/atabey/Prot2Mol'):
        raise RuntimeError('Learning-curve preparation is restricted to Atum Prot2Mol')
    directory = WORK/'learning_curves'
    archive_directory(directory)
    primary_path = WORK/'controlled/plan.json'
    primary = json.loads(primary_path.read_text())
    if (directory/'plan.json').exists():
        previous = json.loads((directory/'plan.json').read_text())
        if previous['primary_plan_sha256'] != sha256(primary_path):
            raise ValueError('Parent control plan changed')
        for regimes in previous['views'].values():
            for view in regimes.values():
                if sha256(Path(view['path'])) != view['sha256']:
                    raise ValueError('Prepared training view changed')
                record = json.loads(Path(view['path']).read_text())
                for name in ('indices','labels','source_metadata'):
                    if sha256(Path(record[name]['path'])) != record[name]['sha256']:
                        raise ValueError('Prepared training view payload changed')
        for job in previous['tuning_jobs']:
            materialize_job(job, directory, sha256(directory/'plan.json'))
        write_json(directory/'preparation_complete.json', dict(plan_sha256=sha256(directory/'plan.json'),
            configurations=len(previous['tuning_jobs']),action='prepared only; no training started'))
        return previous
    units_path = WORK/'tier1/data/assay_units.parquet'
    units = pd.read_parquet(units_path).set_index('assay_id').component
    views, stats, artifacts = {}, {}, {}
    for regime in ('A','B'):
        data = primary['bases'][regime]['data']
        source = Path(data['train_parquet_path'])
        if sha256(source) != primary['source_data_sha256'][regime]['train']:
            raise ValueError('Full training partition changed')
        cache = Path(data['tokenized_dataset_dir'])/'train_examples'
        if token_cache_hashes(cache) != primary['tokenized_data_sha256'][regime]['train']:
            raise ValueError('Full training token cache changed')
        raw = pd.read_parquet(source, columns=['assay_id','target_chembl_id','compound_id',*LABELS,'measurement_count'])
        raw['group_id'] = raw.target_chembl_id.astype(str)+'__'+raw.assay_id.astype(str)
        ds = load_from_disk(str(cache))
        locations = align_rows(raw, ds.select_columns(IDENTITY+LABELS).to_pandas())
        frame = raw.iloc[locations].reset_index(drop=True)
        frame['unit'] = frame.assay_id if regime == 'A' else frame.assay_id.map(units)
        if frame.unit.isna().any():
            raise ValueError('Missing indivisible unit')
        if regime == 'B':
            frame['unit'] = frame.unit.astype(np.int64)
        base = directory/'views'/regime
        base.mkdir(parents=True, exist_ok=True)
        metadata_path = base/'source_metadata.parquet'
        frame.to_parquet(metadata_path, index=False)
        subsets, order = nested_unit_indices(frame.unit.to_numpy())
        order_path = base/'unit_order.npy'
        np.save(order_path, order.astype(str) if order.dtype.kind == 'O' else order, allow_pickle=False)
        source_record = dict(parquet_path=str(source), parquet_sha256=sha256(source),
            cache_dir=str(cache), cache_files_sha256=primary['tokenized_data_sha256'][regime]['train'],
            fingerprint=ds._fingerprint, rows=len(ds))
        heldout = {}
        for role in ('val','test'):
            path = Path(data[f'{role}_parquet_path'])
            if sha256(path) != primary['source_data_sha256'][regime][role]:
                raise ValueError('Fixed held-out partition changed')
            heldout[role] = pd.read_parquet(path, columns=['assay_id','target_chembl_id']).drop_duplicates()
        views[regime], stats[regime] = {}, {}
        for fraction, indices in subsets.items():
            name = fraction_name(fraction)
            labels = subset_pair_labels(frame, indices)
            statistics = subset_statistics(frame, indices, labels, heldout)
            # Check the actual production sampler's realized dataset-item count.
            view = indexed_label_view(ds, indices, labels, f'{regime}_{name}')
            sampler = RewardAssayListDataset(view, seed=42, ranking_max_ligands=16,
                ranking_opportunity_divisor=32, ranking_min_pchembl_span=.5,
                max_classification_only_per_item=16, include_all_assays_for_contrastive=True,
                item_count_multiple=48)
            statistics.update(dataset_items_per_epoch=len(sampler),
                microbatches_per_rank_per_epoch=len(sampler)//48,
                expected_optimizer_steps_per_epoch=math.ceil(len(sampler)/192))
            del sampler, view
            if fraction == 1. and (statistics['labels_changed_from_full_training'] != 0 or not np.array_equal(indices, np.arange(len(ds)))):
                raise ValueError('Full-data endpoint is not equivalent to the shared controlled reference')
            index_path, labels_path = base/f'{name}_indices.npy', base/f'{name}_labels.npy'
            np.save(index_path, indices, allow_pickle=False)
            np.save(labels_path, labels, allow_pickle=False)
            record = dict(format_version=1, regime=regime, fraction_requested=fraction, subset_seed=SUBSET_SEED,
                unit_definition='whole assay' if regime == 'A' else 'whole scaffold-assay component',
                source=source_record, label_policy='within_selected_training_pair_median', activity_threshold=6.,
                source_metadata=dict(path=str(metadata_path),sha256=sha256(metadata_path)),
                indices=dict(path=str(index_path),sha256=sha256(index_path)),
                labels=dict(path=str(labels_path),sha256=sha256(labels_path)), statistics=statistics,
                unit_order_sha256=sha256(order_path),
                sampling='Seeded uniform permutation of units; nested prefixes of floor(fraction * unit_count), minimum one; no label-based choice or forced giant inclusion')
            path = base/f'{name}.json'
            _write_immutable(path, json.dumps(record, indent=2, sort_keys=True)+'\n')
            views[regime][name] = dict(path=str(path),sha256=sha256(path))
            stats[regime][name] = statistics
            print(json.dumps(dict(regime=regime,fraction=fraction,**statistics)),flush=True)
        artifacts[regime] = dict(source_metadata_sha256=sha256(metadata_path),unit_order_sha256=sha256(order_path))
        del ds, frame, raw
    arm = next(item for item in primary['arms'] if item['name'] == 'full')
    jobs = []
    for fraction in FRACTIONS[:-1]:
        name = fraction_name(fraction)
        for multiplier in (1., .3, 3.):
            for regime in ('A','B'):
                identity = f"unit_curve_{name}__{regime}__seed42__lr{str(multiplier).replace('.', 'p')}"
                config = make_config(primary['bases'][regime], arm, regime=regime, seed=42, multiplier=multiplier,
                    output=directory/'runs'/identity, training_noise_rms=primary['training_noise_rms'])
                config['data'].update(train_view_manifest=views[regime][name]['path'],train_view_sha256=views[regime][name]['sha256'])
                jobs.append(dict(id=identity, arm=f'unit_curve_{name}', regime=regime, seed=42,
                    phase='tuning', learning_rate_multiplier=multiplier, fraction=fraction, config=config))
    plan = dict(format_version=1, primary_plan_sha256=sha256(primary_path), spec_sha256=primary['spec_sha256'],
        unit_map_sha256=sha256(units_path), subset_seed=SUBSET_SEED, fractions=list(FRACTIONS),
        views=views, statistics=stats, artifacts=artifacts, tuning_jobs=jobs,
        full_data_endpoint='Reuse exact full-arm primary-grid jobs after verifying row/label identity; no duplicate fit',
        accounting=dict(new_tuning_runs=18,new_confirmation_runs=12,new_total_runs=30,shared_full_endpoint_runs=10),
        selection='Each fraction gets the same three LRs on A/B at seed42. B validation selects a common LR for matched A/B confirmations at seeds43/44. Fixed subset seed42 isolates optimizer-seed spread from subset draw variation.',
        compute='15 epochs and effective batch192 at every fraction; optimizer steps and total observations differ with realized subset size. These are data-size curves at equal epoch count, not equal-compute curves.',
        interpretation='Report unit fractions, actual row fractions, target coverage, label changes and whole-unit uncertainty. A giant component can make a flat unit curve uninformative about broad data saturation.',
        remaining='Runtime qualification, all training/validation-only selection and test scoring; additional B allocation analysis if affordable',
        preparation_code_sha256=sha256(Path(__file__)))
    path = directory/'plan.json'
    _write_immutable(path, json.dumps(plan,indent=2,sort_keys=True)+'\n')
    for job in jobs:
        materialize_job(job, directory, sha256(path))
    write_json(directory/'preparation_complete.json', dict(plan_sha256=sha256(path),configurations=len(jobs),action='prepared only; no training started'))
    return plan


if __name__ == '__main__':
    result = prepare()
    print(json.dumps(result['accounting'],indent=2))
