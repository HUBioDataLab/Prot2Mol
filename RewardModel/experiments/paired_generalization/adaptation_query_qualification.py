"""Synthetic CPU runtime qualification for the E49-E52 query worker.

No scientific checkpoint, source dataset, training queue or GPU is accessed.
The small two-epoch fits check implementation, not adaptation performance.
"""
from __future__ import annotations

import argparse
import copy
from importlib.metadata import version
import json
import math
import os
from pathlib import Path

from datasets import Dataset, load_from_disk
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from . import adaptation_design as design, adaptation_views as views, adaptation_query as query
from .adaptation_fit import SCOPES, fit_support, restore_adapter
from .checkpoint_evaluation import parameter_digest, read_native_predictions
from .control_plan import _write_immutable, token_cache_hashes
from .qualify_controls import make_model
from .run import ROOT, sha256, write_json
from reward_model.training.config import RewardTrainerConfig, RewardTrainingConfigBundle, RewardTrainingDataConfig
from reward_model.training.train_views import subset_pair_labels


def software():
    modules = ['adaptation_query_qualification.py', 'adaptation_query.py', 'adaptation_fit.py',
        'adaptation_design.py', 'adaptation_views.py', 'adaptation_protocol.py',
        'checkpoint_evaluation.py', 'qualify_controls.py', 'realization_protocol.py',
        'objective_probes.py', 'control_plan.py', 'run.py']
    paths = [Path(__file__).with_name(n) for n in modules]
    paths += sorted((ROOT/'RewardModel/reward_model/model').glob('*.py'))
    paths += [ROOT/'RewardModel/reward_model/training'/n for n in
              ('config.py', 'data.py', 'evaluation.py', 'trainer.py', 'train_views.py')]
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths}


def synthetic_views(directory, regime):
    """Build a real verified bundle from small synthetic whole-unit partitions."""
    directory = Path(directory)
    directory.mkdir(parents=True)
    raw, unit_map = {}, {}
    for role, unit_count, offset in (('train', 1, 0), ('val', 4, 100), ('test', 2, 200)):
        rows = []
        for u in range(unit_count):
            for a in range(4):
                assay = f'{role}_{u}_{a}'
                unit_map[assay] = offset+u
                # One small assay per unit exercises explicit ineligibility.
                for m, value in enumerate([5., 7., 9.][:2 if a==3 else 3]):
                    rows.append(dict(assay_id=assay, target_chembl_id=f'p{u%2}',
                        compound_id=f'{role}_{u}_{a}_{m}', pchembl_value=value+.1*(a%2)))
        raw[role] = pd.DataFrame(rows)
    units = pd.Series(unit_map)
    inventories = {role:design.group_inventory(frame, regime, role, units) for role,frame in raw.items()}
    groups, _, summary = design.build_design(inventories, regime, k_values=(1, 2))
    group_path = directory/'groups.parquet'
    groups.to_parquet(group_path, index=False)
    primary = dict(bases={regime:{'data':{'tokenized_dataset_dir':str(directory/'tokens')}}},
                   source_data_sha256={regime:{}}, tokenized_data_sha256={regime:{}})
    sources, metadata = {}, {}
    for role in ('val', 'test'):
        frame = raw[role]
        frame['binary_label'] = subset_pair_labels(frame, np.arange(len(frame)))
        raw_path = directory/f'{role}.parquet'
        frame.to_parquet(raw_path, index=False)
        token_rows = []
        for i, row in enumerate(frame.iloc[::-1].to_dict('records')):
            token_rows.append(dict(**row, group_id=row['target_chembl_id']+'__'+row['assay_id'],
                activity_type='IC50', protein_input_ids=[0, 5+int(row['target_chembl_id'][1:]), 2],
                protein_attention_mask=[1]*3, protein_length=3,
                molecule_input_ids=[0, 3+i%20, 4+(i*3)%20, 2],
                molecule_attention_mask=[1]*4, molecule_length=4))
        dataset = Dataset.from_list(token_rows)
        cache = directory/'tokens'/f'{role}_examples'
        dataset.save_to_disk(str(cache))
        dataset = load_from_disk(str(cache))
        metadata[role] = views.align_metadata(frame, dataset, groups[groups.source_partition==role])
        metadata_path = directory/f'{role}_metadata.parquet'
        metadata[role].to_parquet(metadata_path, index=False)
        sources[role] = dict(parquet=dict(path=str(raw_path), sha256=sha256(raw_path)),
            cache_dir=str(cache), cache_files_sha256=token_cache_hashes(cache),
            fingerprint=dataset._fingerprint, rows=len(dataset),
            metadata=dict(path=str(metadata_path), sha256=sha256(metadata_path)))
        primary['bases'][regime]['data'][role+'_parquet_path'] = str(raw_path)
        primary['source_data_sha256'][regime][role] = sha256(raw_path)
        primary['tokenized_data_sha256'][regime][role] = token_cache_hashes(cache)
    primary_path, design_path = directory/'primary.json', directory/'design.json'
    write_json(primary_path, primary)
    write_json(design_path, dict(primary_plan_sha256=sha256(primary_path), spec_sha256='0'*64,
        artifacts_sha256={f'groups_{regime}.parquet':sha256(group_path)}, regimes={regime:summary}))
    requests = views.view_requests(groups, k_values=(1, 2))
    indices, labels, records = views.pack_views(metadata, requests['views'], groups)
    np.save(directory/'indices.npy', indices)
    np.save(directory/'labels.npy', labels)
    record = dict(format_version=1, regime=regime, label_policy=views.LABEL_POLICY,
        activity_threshold=6., spec_sha256='0'*64, k_values=[1, 2], sources=sources, views=records,
        **{k:v for k,v in requests.items() if k!='views'})
    for name, path in [('primary_plan', primary_path), ('design', design_path), ('groups', group_path),
                       ('indices', directory/'indices.npy'), ('labels', directory/'labels.npy')]:
        record[name] = dict(path=str(path), sha256=sha256(path))
    manifest = directory/'views.json'
    write_json(manifest, record)
    return views.AdaptationViews(manifest, sha256(manifest))


def check_query(directory, bundle):
    """Independently recompute coverage and SciPy ranks from native payloads."""
    result = json.loads((directory/'complete.json').read_text())
    native_path = directory/'native/complete.json'
    if sha256(native_path) != result['native_complete_sha256']:
        raise ValueError('Native query completion changed')
    native = json.loads(native_path.read_text())
    if (not result['cpu_qualification'] or not result['base_restored']
            or not result['modes_and_rng_restored'] or result['optimizer_created']
            or result['selection_performed'] or not native['cpu_qualification']
            or native['device']!='cpu' or native['optimizer_created']):
        raise ValueError('Query did not remain an isolated CPU qualification')
    manifest = Path(result['prediction_manifest'])
    if (sha256(manifest) != result['prediction_manifest_sha256']
            or native['prediction_manifest'] != str(manifest)):
        raise ValueError('Query prediction manifest changed')
    frame, _ = read_native_predictions(manifest)
    source = result['source']
    _, examples, groups = query.query_view(bundle, source['task_id'], source['query_role'])
    expected = pd.DataFrame(examples.select_columns(query.IDENTITY+query.LABELS).to_dict())
    for name in query.IDENTITY:
        if frame[name].astype(str).tolist() != expected[name].astype(str).tolist():
            raise ValueError('Query observation identity changed')
    for name in query.LABELS:
        if not np.array_equal(frame[name].to_numpy(dtype=np.float32), expected[name].to_numpy(dtype=np.float32)):
            raise ValueError('Query labels changed')
    checked = 0
    for scorer in query.SCORERS:
        reported = {r['group_id']:r for r in result['assay_records'] if r['scorer']==scorer}
        if set(reported) != set(groups.group_id):
            raise ValueError('Query omitted an assay')
        correlations, eligible_count = [], 0
        for group in groups.itertuples():
            rows = frame[frame.group_id==group.group_id]
            labels = rows.pchembl_value.to_numpy(dtype=np.float32).astype(np.float64)
            eligible = len(rows)>=3 and np.ptp(labels)>=.5
            values = rows[scorer].to_numpy()
            rho = float(spearmanr(values, labels).statistic) if eligible and np.ptp(values)>0 else None
            defined = rho is not None and math.isfinite(rho)
            expected_row = dict(scorer=scorer, group_id=group.group_id, unit=group.unit,
                rows=len(rows), eligible=eligible, defined=defined, rho=rho if defined else None)
            if reported[group.group_id] != expected_row:
                raise ValueError('Query assay metrics failed independent reconstruction')
            eligible_count += eligible
            if defined:correlations.append(rho)
            checked += 1
        summary = result['pathways'][scorer]
        macro = float(np.mean(correlations)) if correlations else None
        if (summary['macro_spearman'] != macro or summary['spearman_num_groups'] != eligible_count
                or summary['spearman_num_defined_groups'] != len(correlations)):
            raise ValueError('Query macro or coverage failed independent reconstruction')
    return checked


def verify_cpu(output):
    output = Path(output)
    record = json.loads((output/'qualification.json').read_text())
    if record['source_sha256'] != software() or not record['synthetic_only']:
        raise ValueError('Qualification source or scope changed')
    for relative, digest in record['artifacts_sha256'].items():
        path = output/relative
        if not path.resolve().is_relative_to(output.resolve()) or sha256(path)!=digest:
            raise ValueError('Qualification artifact changed')
    bundles = {regime:views.AdaptationViews(output/regime/'views/views.json',
        sha256(output/regime/'views/views.json')) for regime in ('A','B')}
    checks = sum(check_query(output/item['directory'], bundles[item['regime']]) for item in record['queries'])
    return dict(query_exports_verified=len(record['queries']), assay_pathway_checks=checks,
        artifacts_verified=len(record['artifacts_sha256']), sources_verified=len(record['source_sha256']),
        qualification_sha256=sha256(output/'qualification.json'), scientific_results=0,
        production_gpu_qualified=False)


def qualify_cpu(output):
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('ACCELERATE_USE_CPU')!='true'
            or os.environ.get('WANDB_MODE')!='disabled'):
        raise RuntimeError('Explicit CPU-only, W&B-disabled qualification required')
    output = Path(output).resolve()
    if output.exists():raise RuntimeError('Existing qualification requires inspection; never retry implicitly')
    torch.set_num_threads(2)
    source_hashes = software()
    queries, fits = [], []
    for regime in ('A', 'B'):
        bundle = synthetic_views(output/regime/'views', regime)
        model = make_model(None, {})
        config = RewardTrainingConfigBundle(model=model.config,
            data=RewardTrainingDataConfig(train_parquet_path='synthetic_support', val_parquet_path='synthetic_query',
                test_parquet_path='synthetic_test_query', tokenized_dataset_dir='synthetic_cache',
                enforce_binary_label_threshold=False, evaluation_ranking_partitions=3),
            training=RewardTrainerConfig(output_dir='unused_output', training_mode='single_gpu',
                seed=42, num_train_epochs=2, per_device_train_batch_size=2, per_device_eval_batch_size=2,
                gradient_accumulation_steps=2, learning_rate=.001, logging_steps=1, report_to=[],
                evaluate_test=False, protein_shuffle_sensitivity=False))
        base, state = parameter_digest(model), copy.deepcopy(model.state_dict())
        tasks = [('across_k1', ['validation','test'])]
        if regime=='B':
            tasks += [(next(k for k,t in bundle.record['tasks'].items()
                      if t['kind']=='within' and t['k']==1 and t['probe_role']==role), [role])
                      for role in ('validation','test')]
        for task_id, roles in tasks:
            task = bundle.record['tasks'][task_id]
            baseline = task['baseline_task']
            for role in roles:
                directory = output/regime/baseline/role
                query.evaluate_query(model, bundle, config, directory, task_id=baseline,
                    role=role, epoch=0, base_parameter_sha256=base, cpu=True)
                queries.append(dict(regime=regime, directory=str(directory.relative_to(output))))
            for scope in SCOPES:
                directory = output/regime/task_id/scope
                source = query.task_source(bundle, task_id, 42)
                fit = fit_support(model, bundle.load(task['support_view']), config, directory/'fit',
                    scope=scope, base_parameter_sha256=base, source=source, cpu=True)
                fits.append(dict(regime=regime, task_id=task_id, scope=scope, steps=fit['global_step']))
                model.load_state_dict(state)
                for role in roles:
                    for epoch in ([1,2] if role=='validation' else [2]):
                        target = directory/f'{role}_epoch{epoch}'
                        result = query.evaluate_query(model, bundle, config, target, task_id=task_id,
                            role=role, epoch=epoch, scope=scope, base_parameter_sha256=base,
                            fit_directory=directory/'fit', fit_complete_sha256=sha256(directory/'fit/complete.json'), cpu=True)
                        snapshot = fit['snapshots'][epoch-1]
                        restore_adapter(model, directory/'fit'/snapshot['file'], snapshot['sha256'])
                        native = json.loads((target/'native/complete.json').read_text())
                        if parameter_digest(model)!=native['parameter_sha256']:
                            raise ValueError('Query did not use the specified adapter weights')
                        model.load_state_dict(state)
                        queries.append(dict(regime=regime, directory=str(target.relative_to(output))))
                print(json.dumps(dict(regime=regime, task=task_id, scope=scope, cpu_qualified=True)), flush=True)
    if software()!=source_hashes:raise RuntimeError('Qualification sources changed during execution')
    artifacts = {str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
    record = dict(synthetic_only=True, cpu_qualification=True, source_sha256=source_hashes,
        packages={n:version(n) for n in ('torch','transformers','accelerate','datasets','numpy','scipy','pandas')},
        synthetic_support_fits=fits, queries=queries, artifacts_sha256=artifacts,
        scientific_fits=0, scientific_query_results=0, production_gpu_qualified=False,
        limitation='Tiny synthetic two-epoch support fits and native CPU queries only. '
        'Does not qualify scientific data, full checkpoints, 15-epoch selection, GPU runtime or Tier-3 completion.')
    _write_immutable(output/'qualification.json', json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n')
    verified = verify_cpu(output)
    _write_immutable(output/'verification.json', json.dumps(verified,indent=2,sort_keys=True)+'\n')
    return verified


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['qualify-cpu','verify-cpu'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps((qualify_cpu if args.command=='qualify-cpu' else verify_cpu)(args.output),indent=2),flush=True)
