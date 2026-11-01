"""Compact, subset-local data views for the frozen E49-E52 memberships.

This prepares data only. It neither chooses an adaptation training protocol nor
opens the Tier-3 completion gate. Within-unit tasks are separate model fits;
their support sets must never be pooled. Every k=0 task uses its exact query
view with unchanged base weights. Token buffers stay in the existing caches.
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
    from .control_plan import token_cache_hashes
    from .realization_protocol import IDENTITY, LABELS, align_rows
    from .adaptation_design import K_VALUES, group_inventory
    from .prepare_random import archive_directory
else:
    from run import ROOT, WORK, sha256, write_json
    from control_plan import token_cache_hashes
    from realization_protocol import IDENTITY, LABELS, align_rows
    from adaptation_design import K_VALUES, group_inventory
    from prepare_random import archive_directory

from reward_model.training.train_views import indexed_label_view, subset_pair_labels

LABEL_POLICY = 'pair_median_separately_within_each_support_or_query_view'
ACTIVITY_THRESHOLD = 6.


def view_requests(groups, *, k_values=K_VALUES):
    """Translate frozen membership into tasks; never choose groups from labels."""
    if not k_values or tuple(sorted(set(k_values))) != tuple(k_values) or any(
            isinstance(k, bool) or int(k) != k or k < 1 for k in k_values):
        raise ValueError('Expected distinct ascending positive integer budgets')
    if groups.empty or groups.group_id.duplicated().any():
        raise ValueError('Expected unique nonempty frozen groups')
    if set(groups.probe_role) != {'adaptation', 'validation', 'test'}:
        raise ValueError('Expected the three frozen adaptation roles')
    if groups.groupby('unit').probe_role.nunique().max() != 1:
        raise ValueError('A unit crosses adaptation roles')
    expected_source = groups.probe_role.map(dict(adaptation='val', validation='val', test='test'))
    if not groups.source_partition.eq(expected_source).all():
        raise ValueError('Frozen roles disagree with original partitions')
    requests, tasks = {}, {}

    def request(name, selected):
        if selected.empty or selected.source_partition.nunique() != 1:
            raise ValueError('Each view needs complete assays from one source partition')
        requests[name] = dict(source_partition=selected.source_partition.iloc[0],
                              group_ids=sorted(selected.group_id.tolist()))
        return name

    validation = request('across_validation', groups[groups.probe_role == 'validation'])
    test = request('across_test', groups[groups.probe_role == 'test'])
    tasks['across_k0'] = dict(kind='across', k=0, support_view=None,
                             validation_view=validation, query_view=test)
    adaptation = groups[groups.probe_role == 'adaptation'].sort_values('adaptation_order')
    if adaptation.adaptation_order.tolist() != list(range(len(adaptation))):
        raise ValueError('Frozen across-unit support order is not a complete permutation')
    across_availability = {}
    for k in k_values:
        across_availability[str(k)] = len(adaptation) >= k
        if len(adaptation) >= k:
            support = request(f'across_support_k{k}', adaptation.iloc[:k])
            tasks[f'across_k{k}'] = dict(kind='across', k=k, support_view=support,
                validation_view=validation, query_view=test, baseline_task='across_k0')

    # One query view per unit, reused by every budget and its unchanged baseline.
    # Units with no support are explicitly counted but cannot form an adaptation task.
    unavailable, seen_ids = [], set()
    for (role, unit), members in groups[groups.probe_role != 'adaptation'].groupby(
            ['probe_role', 'unit'], sort=True):
        support = members[members.within_role == 'support'].sort_values('within_support_order')
        query = members[members.within_role == 'query']
        if (len(support) != len(members)//2 or len(query) + len(support) != len(members)
                or support.within_support_order.tolist() != list(range(len(support)))
                or not query.within_support_order.eq(-1).all()):
            raise ValueError('Invalid fixed within-unit support/query membership')
        if len(support) < min(k_values):
            unavailable.append(dict(probe_role=role, unit=unit, support_capacity=len(support)))
            continue
        identity = role+'_'+hashlib.sha256(unit.encode()).hexdigest()[:20]
        if identity in seen_ids:
            raise ValueError('Within-unit task identity collision')
        seen_ids.add(identity)
        query_view = request('within_query_'+identity, query)
        baseline = 'within_'+identity+'_k0'
        common = dict(kind='within', unit=unit, probe_role=role, query_view=query_view,
                      query_ranking_eligible_groups=int(query.ranking_eligible.sum()))
        tasks[baseline] = dict(**common, k=0, support_view=None)
        for k in k_values:
            if len(support) >= k:
                support_view = request(f'within_support_{identity}_k{k}', support.iloc[:k])
                tasks[f'within_{identity}_k{k}'] = dict(**common, k=k,
                    support_view=support_view, baseline_task=baseline)
    cohorts = {role: {str(k): common_unit_cohort(tasks, role, [k]) for k in k_values}
               for role in ('validation', 'test')}
    rank_cohorts = {role: {str(k): sorted(task['unit'] for task in tasks.values()
        if task['kind'] == 'within' and task['probe_role'] == role and task['k'] == k
        and task['query_ranking_eligible_groups'] > 0) for k in k_values}
        for role in ('validation', 'test')}
    return dict(views=requests, tasks=tasks, across_available=across_availability,
                within_cohorts=cohorts, within_rank_evaluable_cohorts=rank_cohorts,
                within_validation_selection_possible={str(k):bool(rank_cohorts['validation'][str(k)]) for k in k_values},
                within_unavailable=unavailable)


def common_unit_cohort(tasks, role, budgets):
    """Identity intersection for a paired k comparison, including k=0 if asked.

    Rank-ineligible queries stay in the available cohort; report their count and
    missing macro rho explicitly rather than choosing membership from labels.
    """
    if role not in ('validation', 'test') or not budgets:
        raise ValueError('A within-unit cohort needs a role and budgets')
    by_budget = [{task['unit'] for task in tasks.values() if task['kind'] == 'within'
                  and task['probe_role'] == role and task['k'] == k} for k in budgets]
    return sorted(set.intersection(*by_budget))


def align_metadata(raw, dataset, groups):
    """Retain raw label precision for medians, but check native float32 labels."""
    raw = raw.copy()
    raw['group_id'] = raw.target_chembl_id.astype(str)+'__'+raw.assay_id.astype(str)
    locations = align_rows(raw, dataset.select_columns(IDENTITY+LABELS).to_pandas())
    metadata = raw.iloc[locations].reset_index(drop=True)
    for name in IDENTITY:
        metadata[name] = metadata[name].astype(str)
    inventory = metadata.groupby('group_id', sort=True).agg(
        rows=('compound_id', 'size'), target_chembl_id=('target_chembl_id', 'first'))
    frozen = groups.set_index('group_id').sort_index()
    if (not inventory.index.equals(frozen.index)
            or not np.array_equal(inventory.rows, frozen.rows)
            or not np.array_equal(inventory.target_chembl_id.astype(str), frozen.target_chembl_id.astype(str))):
        raise ValueError('Cache observations do not match frozen assay memberships')
    metadata['unit'] = metadata.group_id.map(frozen.unit)
    return metadata


def pack_views(metadata, requests, groups):
    """Pack small indices/labels only; token payloads are never serialized."""
    vectors, label_vectors, records, offset = [], [], {}, 0
    by_group = {role: frame.groupby('group_id', sort=False).indices for role, frame in metadata.items()}
    frozen = groups.set_index('group_id')
    for name, request in requests.items():
        role, ids = request['source_partition'], request['group_ids']
        mapping = by_group[role]
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(mapping):
            raise ValueError('View refers to invalid assay membership')
        indices = np.sort(np.concatenate([mapping[g] for g in ids])).astype(np.int64)
        labels = subset_pair_labels(metadata[role], indices, ACTIVITY_THRESHOLD)
        selected = metadata[role].iloc[indices]
        stats = dict(rows=len(indices), groups=len(ids), units=int(selected.unit.nunique()),
            ranking_eligible_groups=int(frozen.loc[ids].ranking_eligible.sum()),
            positive_rate=float(labels.mean()),
            binary_labels_changed_from_source=int(np.count_nonzero(labels != selected.binary_label.to_numpy())))
        records[name] = dict(**request, offset=offset, length=len(indices), statistics=stats)
        vectors.append(indices)
        label_vectors.append(labels)
        offset += len(indices)
    return np.concatenate(vectors), np.concatenate(label_vectors), records


def checked_artifact(reference):
    path = Path(reference['path'])
    if sha256(path) != reference['sha256']:
        raise ValueError(f'Adaptation artifact changed: {path.name}')
    return path


class AdaptationViews:
    """Verify a frozen bundle once, and materialize zero-copy views on demand.

    The bundle is a data contract, not authorization for fitting. A future
    runner still needs the separately frozen protocol and Tier-3/runtime gates.
    """

    def __init__(self, manifest_path, expected_sha256):
        from datasets import load_from_disk
        self.digest = expected_sha256
        path = checked_artifact(dict(path=str(manifest_path), sha256=expected_sha256))
        self.record = record = json.loads(path.read_text())
        if (record['format_version'] != 1 or record['label_policy'] != LABEL_POLICY
                or record['activity_threshold'] != ACTIVITY_THRESHOLD):
            raise ValueError('Unsupported adaptation view contract')
        design_path = checked_artifact(record['design'])
        design = json.loads(design_path.read_text())
        if (sha256(checked_artifact(record['primary_plan'])) != design['primary_plan_sha256']
                or record['spec_sha256'] != design['spec_sha256']):
            raise ValueError('Adaptation views belong to another frozen design')
        groups_path = checked_artifact(record['groups'])
        if sha256(groups_path) != design['artifacts_sha256'][f"groups_{record['regime']}.parquet"]:
            raise ValueError('Adaptation groups differ from the frozen design')
        self.groups = pd.read_parquet(groups_path)
        self.requests = view_requests(self.groups, k_values=tuple(record['k_values']))
        if (any(record[key] != value for key, value in self.requests.items() if key != 'views')
                or set(record['views']) != set(self.requests['views'])):
            raise ValueError('Adaptation tasks differ from the frozen membership policy')
        if list(record['k_values']) != design['regimes'][record['regime']]['k_values']:
            raise ValueError('Adaptation budgets differ from the frozen design')
        self.indices = np.load(checked_artifact(record['indices']), allow_pickle=False)
        self.labels = np.load(checked_artifact(record['labels']), allow_pickle=False)
        if (self.indices.ndim != 1 or self.indices.dtype != np.int64
                or self.labels.shape != self.indices.shape or self.labels.dtype != np.uint8
                or not np.isin(self.labels, [0, 1]).all()):
            raise ValueError('Invalid packed adaptation vectors')
        self.datasets, self.metadata = {}, {}
        primary = json.loads(checked_artifact(record['primary_plan']).read_text())
        for role in ('val', 'test'):
            source = record['sources'][role]
            raw_path = checked_artifact(source['parquet'])
            cache = Path(source['cache_dir'])
            data = primary['bases'][record['regime']]['data']
            if (raw_path.resolve() != Path(data[role+'_parquet_path']).resolve()
                    or cache.resolve() != (Path(data['tokenized_dataset_dir'])/(role+'_examples')).resolve()
                    or source['parquet']['sha256'] != primary['source_data_sha256'][record['regime']][role]
                    or source['cache_files_sha256'] != primary['tokenized_data_sha256'][record['regime']][role]
                    or token_cache_hashes(cache) != source['cache_files_sha256']):
                raise ValueError('Adaptation source differs from the frozen primary cache')
            dataset = load_from_disk(str(cache))
            if dataset._fingerprint != source['fingerprint'] or len(dataset) != source['rows']:
                raise ValueError('Adaptation source identity or row count changed')
            raw = pd.read_parquet(raw_path, columns=['assay_id', 'target_chembl_id', 'compound_id', *LABELS])
            expected = align_metadata(raw, dataset, self.groups[self.groups.source_partition == role])
            actual = pd.read_parquet(checked_artifact(source['metadata']))
            pd.testing.assert_frame_equal(actual, expected)
            self.datasets[role], self.metadata[role] = dataset, actual
        # Rebuild every vector from the pinned raw observations, not cached
        # pooled binary labels. This also rejects rehashed offset/label edits.
        indices, labels, views = pack_views(self.metadata, self.requests['views'], self.groups)
        if (not np.array_equal(indices, self.indices) or not np.array_equal(labels, self.labels)
                or record['views'] != views):
            raise ValueError('Adaptation view indices or labels violate frozen subset semantics')

    def load(self, view_id):
        view = self.record['views'][view_id]
        start, end = view['offset'], view['offset']+view['length']
        fingerprint = hashlib.sha256((self.digest+':'+view_id).encode()).hexdigest()
        return indexed_label_view(self.datasets[view['source_partition']],
            self.indices[start:end], self.labels[start:end], fingerprint)


def prepare():
    from datasets import load_from_disk
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use CPU-only preparation in authorized Atum Prot2Mol')
    directory = WORK/'tier4/adaptation_views'
    if directory.exists():
        raise RuntimeError('Existing adaptation views require verification before reuse')
    design_path, primary_path = WORK/'tier4/adaptation_design/complete.json', WORK/'controlled/plan.json'
    design, primary = json.loads(design_path.read_text()), json.loads(primary_path.read_text())
    if sha256(primary_path) != design['primary_plan_sha256'] or sha256(ROOT/'Paired input ranking generalization.md') != design['spec_sha256']:
        raise ValueError('Frozen adaptation parent or governing MD changed')
    for path, digest in design['source_sha256'].items():
        checked_artifact(dict(path=path, sha256=digest))
    for name, digest in design['artifacts_sha256'].items():
        checked_artifact(dict(path=str(design_path.parent/name), sha256=digest))
    units = pd.read_parquet(WORK/'tier1/data/assay_units.parquet').set_index('assay_id').component
    archive_directory(directory)
    manifests = {}
    for regime in ('A', 'B'):
        groups_path = design_path.parent/f'groups_{regime}.parquet'
        groups = pd.read_parquet(groups_path)
        requests = view_requests(groups, k_values=tuple(design['regimes'][regime]['k_values']))
        metadata, sources = {}, {}
        dest = directory/regime
        dest.mkdir(parents=True)
        for role in ('val', 'test'):
            data = primary['bases'][regime]['data']
            raw_path = Path(data[role+'_parquet_path'])
            cache = Path(data['tokenized_dataset_dir'])/(role+'_examples')
            hashes = token_cache_hashes(cache)
            if hashes != primary['tokenized_data_sha256'][regime][role]:
                raise ValueError('Frozen adaptation token payload changed')
            raw = pd.read_parquet(raw_path, columns=['assay_id', 'target_chembl_id', 'compound_id', *LABELS])
            # Inventory eligibility follows the Trainer's float32 labels.
            inventory = group_inventory(raw, regime, role, units).sort_values('group_id').reset_index(drop=True)
            frozen = groups[groups.source_partition == role][inventory.columns].sort_values('group_id').reset_index(drop=True)
            pd.testing.assert_frame_equal(inventory, frozen)
            dataset = load_from_disk(str(cache))
            metadata[role] = align_metadata(raw, dataset, frozen)
            metadata_path = dest/f'{role}_metadata.parquet'
            metadata[role].to_parquet(metadata_path, index=False)
            sources[role] = dict(parquet=dict(path=str(raw_path), sha256=sha256(raw_path)),
                cache_dir=str(cache), cache_files_sha256=hashes, fingerprint=dataset._fingerprint,
                rows=len(dataset), metadata=dict(path=str(metadata_path), sha256=sha256(metadata_path)))
        indices, labels, views = pack_views(metadata, requests['views'], groups)
        index_path, label_path = dest/'indices.npy', dest/'labels.npy'
        np.save(index_path, indices, allow_pickle=False)
        np.save(label_path, labels, allow_pickle=False)
        record = dict(format_version=1, regime=regime, label_policy=LABEL_POLICY,
            activity_threshold=ACTIVITY_THRESHOLD,
            spec_sha256=design['spec_sha256'], k_values=design['regimes'][regime]['k_values'],
            design=dict(path=str(design_path), sha256=sha256(design_path)),
            primary_plan=dict(path=str(primary_path), sha256=sha256(primary_path)),
            groups=dict(path=str(groups_path), sha256=sha256(groups_path)), sources=sources,
            indices=dict(path=str(index_path), sha256=sha256(index_path)),
            labels=dict(path=str(label_path), sha256=sha256(label_path)), views=views,
            **{key:value for key,value in requests.items() if key != 'views'},
            common_cohort_policy='Compare identical query views on the intersection of unit identities across budgets; retain rank-ineligible units in coverage counts. Each nonzero task has an unchanged k=0 baseline on that exact query view.',
            fitting_gate=design['fitting_gate'], exposure_limit=design['exposure_limit'],
            code_sha256=sha256(Path(__file__)), token_payloads_copied=False,
            model_fitting=False, model_inference=False, production_runtime_qualified=False)
        manifest_path = dest/'views.json'
        write_json(manifest_path, record)
        AdaptationViews(manifest_path, sha256(manifest_path))
        manifests[regime] = dict(path=str(manifest_path), sha256=sha256(manifest_path))
        print(json.dumps(dict(regime=regime, views=len(views), tasks=len(requests['tasks']),
                              packed_bytes=indices.nbytes+labels.nbytes)), flush=True)
    write_json(directory/'complete.json', dict(manifests=manifests,
        model_fitting=False, model_inference=False, token_payloads_copied=False,
        production_runtime_qualified=False, fitting_gate=design['fitting_gate']))
    return directory


if __name__ == '__main__':
    print(prepare())
