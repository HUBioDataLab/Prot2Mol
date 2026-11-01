import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset, load_from_disk

from experiments.paired_generalization import adaptation_design as design
from experiments.paired_generalization import adaptation_views as views
from experiments.paired_generalization.control_plan import token_cache_hashes
from experiments.paired_generalization.run import sha256, write_json
from reward_model.training.data import RewardAssayListDataset, RewardEvaluationDataset
from reward_model.training.train_views import subset_pair_labels


def fixture(regime='B'):
    raw, unit_map = {}, {}
    for role, sizes in [('train', [2]), ('val', [2, 4, 8, 16]), ('test', [1, 2, 4, 8])]:
        rows = []
        for u, count in enumerate(sizes):
            for a in range(count):
                assay = f'{role}_{u}_{a}'
                unit_map[assay] = {'train':0, 'val':100, 'test':200}[role]+u
                for m, value in enumerate([5., 8., 9.]):
                    rows.append(dict(assay_id=assay, target_chembl_id='p',
                        compound_id=f'{role}_{u}_shared' if m == 0 else f'{assay}_{m}',
                        pchembl_value=value))
        raw[role] = pd.DataFrame(rows)
    units = pd.Series(unit_map)
    inv = {role:design.group_inventory(frame, regime, role, units) for role,frame in raw.items()}
    groups, _, summary = design.build_design(inv, regime, k_values=(1, 2, 4, 64))
    # Same target/molecule has support measurement 5 and query measurement 7.
    # Whole-partition pair medians contaminate some support labels.
    for role in ('val', 'test'):
        query = set(groups.loc[groups.within_role == 'query', 'assay_id'])
        mask = raw[role].assay_id.isin(query) & raw[role].compound_id.str.endswith('shared')
        raw[role].loc[mask, 'pchembl_value'] = 7.
    for frame in raw.values():
        frame['binary_label'] = subset_pair_labels(frame, np.arange(len(frame)))
    return raw, units, groups, summary


def token_dataset(raw):
    frame = raw.iloc[::-1].copy().reset_index(drop=True)
    frame['group_id'] = 'p__'+frame.assay_id
    frame['pchembl_value'] = frame.pchembl_value.astype(np.float32)
    columns = frame.drop(columns='assay_id').to_dict('list')
    columns.update(protein_input_ids=[[i+1, 0] for i in range(len(frame))],
        protein_attention_mask=[[1, 0]]*len(frame), molecule_input_ids=[[1, 2]]*len(frame),
        molecule_attention_mask=[[1, 1]]*len(frame), protein_length=[1]*len(frame),
        molecule_length=[2]*len(frame))
    return Dataset.from_dict(columns)


def bundle(directory, regime='B'):
    directory.mkdir(parents=True, exist_ok=True)
    raw, _, groups, summary = fixture(regime)
    group_path = directory/f'groups_{regime}.parquet'
    groups.to_parquet(group_path, index=False)
    primary = dict(bases={regime:{'data':{'tokenized_dataset_dir':str(directory/'tokens')}}},
                   source_data_sha256={regime:{}}, tokenized_data_sha256={regime:{}})
    metadata, sources = {}, {}
    for role in ('val', 'test'):
        path = directory/f'{role}.parquet'
        raw[role].to_parquet(path, index=False)
        cache = directory/'tokens'/f'{role}_examples'
        token_dataset(raw[role]).save_to_disk(str(cache), num_shards=2)
        ds = load_from_disk(str(cache))
        metadata[role] = views.align_metadata(raw[role], ds, groups[groups.source_partition == role])
        metadata_path = directory/f'{role}_metadata.parquet'
        metadata[role].to_parquet(metadata_path, index=False)
        sources[role] = dict(parquet=dict(path=str(path), sha256=sha256(path)),
            cache_dir=str(cache), cache_files_sha256=token_cache_hashes(cache),
            fingerprint=ds._fingerprint, rows=len(ds),
            metadata=dict(path=str(metadata_path), sha256=sha256(metadata_path)))
        primary['bases'][regime]['data'][role+'_parquet_path'] = str(path)
        primary['source_data_sha256'][regime][role] = sha256(path)
        primary['tokenized_data_sha256'][regime][role] = token_cache_hashes(cache)
    primary_path = directory/'primary.json'
    write_json(primary_path, primary)
    design_path = directory/'design.json'
    write_json(design_path, dict(primary_plan_sha256=sha256(primary_path), spec_sha256='f'*64,
        artifacts_sha256={f'groups_{regime}.parquet':sha256(group_path)}, regimes={regime:summary}))
    requests = views.view_requests(groups, k_values=(1, 2, 4, 64))
    indices, labels, records = views.pack_views(metadata, requests['views'], groups)
    np.save(directory/'indices.npy', indices)
    np.save(directory/'labels.npy', labels)
    record = dict(format_version=1, regime=regime, label_policy=views.LABEL_POLICY,
                  activity_threshold=6., spec_sha256='f'*64,
                  k_values=[1, 2, 4, 64], sources=sources, views=records,
                  **{k:v for k,v in requests.items() if k != 'views'})
    for name, path in [('primary_plan',primary_path), ('design',design_path), ('groups',group_path),
                       ('indices',directory/'indices.npy'), ('labels',directory/'labels.npy')]:
        record[name] = dict(path=str(path), sha256=sha256(path))
    manifest = directory/'views.json'
    write_json(manifest, record)
    return manifest, record


def test_membership_nesting_separate_unit_tasks_fixed_queries_and_zero_shot_baselines():
    _, _, groups, _ = fixture()
    plan = views.view_requests(groups, k_values=(1, 2, 4, 64))
    assert not plan['across_available']['64']
    assert 'across_k64' not in plan['tasks']
    assert plan['tasks']['across_k0']['support_view'] is None
    assert set(plan['views']['across_test']['group_ids']) == set(groups[groups.probe_role == 'test'].group_id)
    previous = set()
    for k in (1, 2, 4):
        ids = set(plan['views'][f'across_support_k{k}']['group_ids'])
        assert previous <= ids and len(ids) == k
        previous = ids
    for task in plan['tasks'].values():
        if task['kind'] != 'within' or task['k'] == 0:
            continue
        baseline = plan['tasks'][task['baseline_task']]
        assert baseline['support_view'] is None and baseline['query_view'] == task['query_view']
        support = set(plan['views'][task['support_view']]['group_ids'])
        query = set(plan['views'][task['query_view']]['group_ids'])
        assert not support & query and len(support) == task['k']
        assert set(groups[groups.group_id.isin(support|query)].unit) == {task['unit']}
    common = views.common_unit_cohort(plan['tasks'], 'test', [0, 1, 2, 4])
    assert common == plan['within_cohorts']['test']['4']
    assert len(common) < len(plan['within_cohorts']['test']['1'])


def test_impossible_budgets_and_rank_ineligible_queries_are_not_reassigned():
    _, _, groups, _ = fixture('A')
    plan = views.view_requests(groups, k_values=(1, 2, 4, 64))
    assert not any(t['kind'] == 'within' for t in plan['tasks'].values())
    assert plan['within_unavailable']
    assert all(not cohort for by_k in plan['within_cohorts'].values() for cohort in by_k.values())
    _, _, groups, _ = fixture()
    before = views.view_requests(groups, k_values=(1, 2))
    groups['ranking_eligible'] = False
    after = views.view_requests(groups.iloc[::-1], k_values=(1, 2))
    assert before['views'] == after['views']
    assert before['within_cohorts'] == after['within_cohorts']
    assert all(t['query_ranking_eligible_groups'] == 0 for t in after['tasks'].values() if t['kind'] == 'within')
    assert not any(after['within_validation_selection_possible'].values())


def test_disk_views_keep_tokens_raw_labels_and_native_consumers(tmp_path, monkeypatch):
    manifest, record = bundle(tmp_path)
    reader = views.AdaptationViews(manifest, sha256(manifest))
    def no_flatten(*args, **kwargs):
        raise AssertionError('Token payloads must not be flattened/copied')
    monkeypatch.setattr(Dataset, 'flatten_indices', no_flatten)
    changed = 0
    for name, item in record['views'].items():
        view = reader.load(name)
        source = reader.datasets[item['source_partition']]
        raw = reader.metadata[item['source_partition']]
        expected = raw[raw.group_id.isin(item['group_ids'])]
        assert view['compound_id'] == expected.compound_id.tolist()
        np.testing.assert_array_equal(view['pchembl_value'], expected.pchembl_value.astype(np.float32))
        # Independent expected pair medians on this exact role and budget.
        medians = expected.groupby(['target_chembl_id','compound_id']).pchembl_value.median()
        labels = [int(medians.loc[(r.target_chembl_id,r.compound_id)] >= 6.) for r in expected.itertuples()]
        assert view['binary_label'] == labels
        changed += np.count_nonzero(np.asarray(labels) != expected.binary_label.to_numpy())
        for column in ('protein_input_ids', 'molecule_input_ids', 'protein_attention_mask', 'molecule_attention_mask'):
            buffers = lambda ds: [c.values.buffers()[1].address for c in ds.data.column(column).chunks]
            # A contiguous selection can drop unused shards or slice a chunk;
            # every remaining token buffer must still belong to the source.
            assert buffers(view) and set(buffers(view)) <= set(buffers(source))
        assert {c['filename'] for c in view.cache_files} <= {c['filename'] for c in source.cache_files}
        assert pickle.loads(pickle.dumps(view))[:] == view[:]
        sampler = RewardAssayListDataset(view, include_all_assays_for_contrastive=True)
        assert sorted(sampler.epoch_example_indices()) == list(range(len(view)))
        evaluator = RewardEvaluationDataset(view, protein_shuffle_sensitivity=False)
        assert [evaluator[i]['rows'][0]['binary_label'] for i in range(len(evaluator))] == labels
        assert source['binary_label'] == raw.binary_label.tolist()
    assert changed > 0


@pytest.mark.parametrize('corruption', ['labels', 'offset', 'task', 'cache', 'source_precision', 'budget'])
def test_rehashed_artifacts_cannot_change_subset_semantics(tmp_path, corruption):
    manifest, record = bundle(tmp_path)
    if corruption == 'labels':
        path = Path(record['labels']['path'])
        labels = np.load(path)
        labels[0] = 1-labels[0]
        np.save(path, labels)
        record['labels']['sha256'] = sha256(path)
    elif corruption == 'offset':
        record['views']['across_validation']['offset'] += 1
    elif corruption == 'task':
        record['tasks']['across_k1']['query_view'] = 'across_validation'
    elif corruption == 'cache':
        source = record['sources']['val']
        shard = next(n for n in source['cache_files_sha256'] if n.endswith('.arrow'))
        path = Path(source['cache_dir'])/shard
        with path.open('ab') as stream:
            stream.write(b'tampered')
    elif corruption == 'source_precision':
        # A raw-label change that disappears in float32 still changes the
        # pinned raw source, and must never alter pair medians silently.
        source = record['sources']['val']
        path = Path(source['parquet']['path'])
        raw = pd.read_parquet(path)
        raw.loc[0, 'pchembl_value'] += 1e-10
        raw.to_parquet(path, index=False)
        source['parquet']['sha256'] = sha256(path)
    else:
        record['k_values'] = [1, 2, 4]
    write_json(manifest, record)
    with pytest.raises(ValueError):
        views.AdaptationViews(manifest, sha256(manifest))


def test_cache_alignment_rejects_missing_rows_changed_values_and_fragmented_frozen_assays():
    raw, _, groups, _ = fixture()
    ds = token_dataset(raw['val'])
    frozen = groups[groups.source_partition == 'val']
    with pytest.raises(ValueError, match='permutation'):
        views.align_metadata(raw['val'], ds.select(range(1,len(ds))), frozen)
    changed = raw['val'].copy()
    changed.loc[0, 'pchembl_value'] += 1.
    with pytest.raises(ValueError, match='labels disagree'):
        views.align_metadata(changed, ds, frozen)
    changed_groups = frozen.copy()
    changed_groups.loc[changed_groups.index[0], 'rows'] += 1
    with pytest.raises(ValueError, match='frozen assay'):
        views.align_metadata(raw['val'], ds, changed_groups)


def test_support_label_does_not_depend_on_query_values():
    raw, _, groups, _ = fixture()
    plan = views.view_requests(groups, k_values=(1, 2))
    before, after = {}, {}
    for role in ('val', 'test'):
        ds = token_dataset(raw[role])
        before[role] = views.align_metadata(raw[role], ds, groups[groups.source_partition == role])
        after[role] = before[role].copy()
        query_ids = set(groups[groups.within_role == 'query'].group_id)
        after[role].loc[after[role].group_id.isin(query_ids), 'pchembl_value'] = 1000.
    _, first, records = views.pack_views(before, plan['views'], groups)
    _, second, _ = views.pack_views(after, plan['views'], groups)
    for task in plan['tasks'].values():
        if task['kind'] == 'within' and task['k'] > 0:
            item = records[task['support_view']]
            sl = slice(item['offset'], item['offset']+item['length'])
            np.testing.assert_array_equal(first[sl], second[sl])


def test_pair_medians_keep_raw_precision_at_the_activity_threshold():
    raw, _, groups, _ = fixture()
    plan = views.view_requests(groups, k_values=(1, 2))
    task = next(t for t in plan['tasks'].values() if t['kind'] == 'within' and t['k'] == 1)
    request = plan['views'][task['support_view']]
    role = request['source_partition']
    chosen = raw[role].assay_id.map(lambda a:'p__'+a).isin(request['group_ids']) & raw[role].compound_id.str.endswith('shared')
    raw[role].loc[chosen, 'pchembl_value'] = 5.99999999
    raw[role]['binary_label'] = subset_pair_labels(raw[role], np.arange(len(raw[role])))
    ds = token_dataset(raw[role])
    metadata = views.align_metadata(raw[role], ds, groups[groups.source_partition == role])
    indices, labels, _ = views.pack_views({role:metadata}, {'support':request}, groups)
    shared = metadata.iloc[indices].compound_id.str.endswith('shared').to_numpy()
    assert np.asarray(ds['pchembl_value'])[indices][shared].tolist() == [6.]
    assert labels[shared].tolist() == [0]


def test_prepare_both_regimes_with_frozen_source_hashes_and_no_model(tmp_path, monkeypatch):
    work = tmp_path/'work'
    design_dir = work/'tier4/adaptation_design'
    design_dir.mkdir(parents=True)
    (work/'controlled').mkdir()
    (work/'tier1/data').mkdir(parents=True)
    _, units, _, _ = fixture()
    unit_path = work/'tier1/data/assay_units.parquet'
    units.rename_axis('assay_id').rename('component').reset_index().to_parquet(unit_path, index=False)
    primary = dict(bases={}, source_data_sha256={}, tokenized_data_sha256={})
    frozen = dict(spec_sha256='f'*64, source_sha256={str(unit_path):sha256(unit_path)},
        artifacts_sha256={}, regimes={}, fitting_gate='Tier 3 plus separate fit/runtime qualification',
        exposure_limit='Conditional on original validation model selection')
    for regime in ('A', 'B'):
        _, record = bundle(tmp_path/regime, regime)
        parent = json.loads(Path(record['primary_plan']['path']).read_text())
        for key in primary:
            primary[key].update(parent[key])
        original_design = json.loads(Path(record['design']['path']).read_text())
        frozen['regimes'].update(original_design['regimes'])
        group_path = design_dir/f'groups_{regime}.parquet'
        group_path.write_bytes(Path(record['groups']['path']).read_bytes())
        frozen['artifacts_sha256'][group_path.name] = sha256(group_path)
        for source in record['sources'].values():
            frozen['source_sha256'][source['parquet']['path']] = source['parquet']['sha256']
    write_json(work/'controlled/plan.json', primary)
    frozen['primary_plan_sha256'] = sha256(work/'controlled/plan.json')
    write_json(design_dir/'complete.json', frozen)
    # All fixture I/O remains under tmp_path; only the production guard and
    # governing-MD hash are substituted. Real cache bytes and raw files are read.
    monkeypatch.setattr(views, 'ROOT', Path('/home/atabey/Prot2Mol'))
    monkeypatch.setattr(views, 'WORK', work)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(views, 'archive_directory', lambda path:path.mkdir())
    monkeypatch.setattr(views, 'sha256', lambda path:'f'*64 if Path(path) == views.ROOT/'Paired input ranking generalization.md' else sha256(path))
    output = views.prepare()
    complete = json.loads((output/'complete.json').read_text())
    assert set(complete['manifests']) == {'A', 'B'}
    assert not complete['model_fitting'] and not complete['model_inference']
    assert not complete['production_runtime_qualified'] and not complete['token_payloads_copied']
    for reference in complete['manifests'].values():
        reader = views.AdaptationViews(reference['path'], reference['sha256'])
        assert reader.load('across_test').num_rows > 0
    assert not list(output.rglob('*.arrow'))
    with pytest.raises(RuntimeError, match='Existing adaptation views'):
        views.prepare()
