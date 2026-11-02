"""Freeze E56/E57 diagnostic contexts without fitting or model inference.

Training contexts follow the reference epoch-zero local 12-item sampler.
Cold validation contexts use the same native assay-list builder on the already
frozen own-regime cold cohort, with single-device 12-item batching. These allow
all three losses in a differentiable forward; they are not the ordinary native
32-row validation metric pass. No result chooses or enriches a batch.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .control_plan import digest_json
    from .prepare_random import archive_directory
    from .realization_protocol import IDENTITY, LABELS, RAW_COLUMNS, align_rows, example_frame
    from .objective_probe_design import unit_ids
else:
    from run import ROOT, WORK, sha256, write_json
    from control_plan import digest_json
    from prepare_random import archive_directory
    from realization_protocol import IDENTITY, LABELS, RAW_COLUMNS, align_rows, example_frame
    from objective_probe_design import unit_ids

BATCH_PAIRS = 32
CONTEXT_SEED = 42
LOCAL_ITEMS = 12
LENGTHS = ['protein_length', 'molecule_length']


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, digest):
    if sha256(Path(path)) != digest:
        raise ValueError('Frozen input changed: '+str(path))


def software():
    from importlib.metadata import version
    sources = {str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__))}
    for relative in ('RewardModel/reward_model/training/data.py', 'RewardModel/reward_model/training/trainer.py',
                     *['RewardModel/experiments/paired_generalization/'+name for name in
                       ('run.py', 'control_plan.py', 'prepare_random.py', 'objective_probe_design.py', 'realization_protocol.py')]):
        sources[relative] = sha256(ROOT/relative)
    return dict(source_sha256=sources, packages={name: version(name) for name in
        ('numpy', 'pandas', 'datasets', 'torch', 'transformers', 'PyYAML')})


def batch_numbers(total, count):
    if not isinstance(total, int) or not isinstance(count, int) or count < 1 or total < count:
        raise ValueError('Requested diagnostic batches must be distinct and available')
    return np.linspace(0, total-1, count, dtype=int).tolist()


def sampler_layout(examples, config, role):
    from reward_model.training.data import RewardAssayListDataset
    from reward_model.training.trainer import AssayListEpochSampler
    if role not in ('train', 'cold_validation'):
        raise ValueError('Only own-regime training and cold validation contexts are allowed')
    if (config.training.seed != CONTEXT_SEED or config.training.per_device_train_batch_size != LOCAL_ITEMS or
        config.model.ranking_balance_assays or config.model.contrastive_loss_weight <= 0 or
        config.data.sample_all_assays is False):
        raise ValueError('Context construction requires the frozen full-RBC seed42 reference')
    if not set(LENGTHS) <= set(examples.column_names):
        raise ValueError('Cached sequence lengths are required; do not scan token payloads')
    multiple = LOCAL_ITEMS * (4 if role == 'train' else 1)
    dataset = RewardAssayListDataset(examples, seed=CONTEXT_SEED,
        ranking_max_ligands=config.data.ranking_max_ligands,
        ranking_opportunity_divisor=config.data.ranking_opportunity_divisor,
        ranking_min_pchembl_span=config.data.ranking_min_pchembl_span,
        max_classification_only_per_item=config.data.max_classification_only_per_item,
        include_all_assays_for_contrastive=True, item_count_multiple=multiple,
        balance_ranking_assays=False)
    sampler = AssayListEpochSampler(dataset, seed=CONTEXT_SEED,
        length_bucketing=config.training.length_bucketing, batch_size=LOCAL_ITEMS,
        bucket_size_multiplier=config.training.length_bucket_size_multiplier)
    order = [index for epoch, index in sampler if epoch == 0]
    if len(order) != len(dataset) or len(order) % LOCAL_ITEMS or len(set(order)) != len(order):
        raise ValueError('Native context sampler did not produce distinct complete local batches')
    return dataset, [order[i:i+LOCAL_ITEMS] for i in range(0, len(order), LOCAL_ITEMS)]


def freeze_batches(dataset, batches, metadata, *, count):
    """Save list boundaries and original cache indices without reading tokens."""
    columns = [*IDENTITY, *LABELS, 'cache_row_index', 'source_row_index', 'unit']
    if (len(metadata) != len(dataset.example_dataset) or metadata[columns].isna().any().any() or
        metadata.duplicated(IDENTITY).any() or metadata.cache_row_index.duplicated().any()):
        raise ValueError('Context metadata is incomplete or ambiguous')
    for name in ('cache_row_index', 'source_row_index'):
        if not pd.api.types.is_integer_dtype(metadata[name]) or (metadata[name] < 0).any():
            raise ValueError('Context mappings must use nonnegative integer indices')
    records, used = [], set()
    for ordinal, number in enumerate(batch_numbers(len(batches), count)):
        items, row_order, contrastive_order = [], [], []
        for item_index in batches[number]:
            lists, classification = dataset._bundles[item_index]
            entries = []
            for group, indices, eligible in lists:
                if not metadata.iloc[list(indices)].group_id.eq(group).all():
                    raise ValueError('A native list crosses assay identity')
                entries.append(dict(group_id=group, ranking_eligible=bool(eligible),
                    cache_row_indices=metadata.iloc[list(indices)].cache_row_index.astype(int).tolist()))
                row_order.extend(indices)
                contrastive_order.extend(indices)
            row_order.extend(classification)
            items.append(dict(item_index=int(item_index), assay_lists=entries,
                classification_cache_row_indices=metadata.iloc[list(classification)].cache_row_index.astype(int).tolist()))
        flat = metadata.iloc[row_order]
        if len(flat) == 0:
            raise ValueError('Empty diagnostic batch')
        used.update(row_order)
        continuous = flat.pchembl_value.to_numpy(dtype=np.float32)
        records.append(dict(pair_index=ordinal, source_batch_index=number, items=items,
            cache_row_indices=flat.cache_row_index.astype(int).tolist(),
            rows=len(flat), unique_rows=int(flat.cache_row_index.nunique()), groups=int(flat.group_id.nunique()),
            units=int(flat.unit.nunique()), flagged_ranking_lists=sum(x['ranking_eligible'] for i in items for x in i['assay_lists']),
            contrastive_lists=sum(len(i['assay_lists']) for i in items),
            strict_active_rows=int((continuous > np.float32(6.)).sum()),
            strict_active_contrastive_rows=int((metadata.iloc[contrastive_order].pchembl_value.to_numpy(dtype=np.float32) > np.float32(6.)).sum()),
            continuous_label_span=float(continuous.max())-float(continuous.min())))
    rows = metadata.iloc[sorted(used)][columns].sort_values('cache_row_index').reset_index(drop=True)
    return dict(population_batches=len(batches), selected_batches=len(records),
        selection='Evenly spaced indices over the complete native epoch-zero batch order; no score or gradient selection.',
        native_sampling=asdict(dataset.stats), batches=records), rows


def materialize_batch(record, examples):
    """Reconstruct exact native collator inputs later; no resampling or model."""
    features = []
    for item in record['items']:
        lists = item['assay_lists']
        indices = [x for entry in lists for x in entry['cache_row_indices']]
        indices += item['classification_cache_row_indices']
        columns = examples[indices]
        rows = [{name: values[i] for name, values in columns.items()} for i in range(len(indices))]
        for entry in lists:
            actual = examples.select(entry['cache_row_indices'])['group_id']
            if any(str(g) != entry['group_id'] for g in actual):
                raise ValueError('Frozen batch points to a different cached assay')
        features.append(dict(rows=rows, example_indices=indices,
            contrastive_group_sizes=[len(x['cache_row_indices']) for x in lists],
            contrastive_assay_ids=[x['group_id'] for x in lists],
            ranking_group_flags=[x['ranking_eligible'] for x in lists],
            ranking_group_sizes=[len(x['cache_row_indices']) for x in lists if x['ranking_eligible']],
            ranking_assay_ids=[x['group_id'] for x in lists if x['ranking_eligible']]))
    if [i for feature in features for i in feature['example_indices']] != record['cache_row_indices']:
        raise ValueError('Frozen context flattening order changed')
    return features


def aligned_metadata(source, examples, regime, components):
    source = example_frame(source)
    metadata = examples.select_columns(IDENTITY+LABELS+LENGTHS).to_pandas()
    locations = align_rows(source, metadata)
    metadata = example_frame(metadata)
    metadata['cache_row_index'] = np.arange(len(metadata))
    metadata['source_row_index'] = locations
    source_units = unit_ids(source, regime, components)
    metadata['unit'] = source_units.to_numpy()[locations]
    if any(not pd.api.types.is_integer_dtype(metadata[name]) or (metadata[name] <= 0).any() for name in LENGTHS):
        raise ValueError('Positive integer cached sequence lengths are required')
    return metadata


def cold_view(examples, metadata, frozen_rows):
    required = ['source_row_index', 'unit', *IDENTITY]
    rows = frozen_rows[required]
    if rows.duplicated(IDENTITY).any() or rows.source_row_index.duplicated().any():
        raise ValueError('Cold membership is ambiguous')
    selected_units = set(rows.unit)
    selected = metadata[metadata.unit.isin(selected_units)].copy()
    expected = rows.sort_values('source_row_index').reset_index(drop=True)
    actual = selected[required].sort_values('source_row_index').reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError('Cold view must exactly preserve all rows of the frozen units')
    selected = selected.sort_values('cache_row_index').reset_index(drop=True)
    return examples.select(selected.cache_row_index.tolist()), selected


def prepare():
    from datasets import load_from_disk
    from reward_model.training.config import load_reward_training_config
    import torch
    if (ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or
        os.environ.get('ACCELERATE_USE_CPU', '').lower() != 'true'):
        raise RuntimeError('Use CPU-only batch preparation in authorized Atum Prot2Mol')
    torch.set_num_threads(2)
    code = software()
    output = WORK/'tier4/objective_probe_batches'
    if output.exists():
        raise RuntimeError('Existing batch preparation requires inspection')
    primary_path = WORK/'controlled/plan.json'; primary = read(primary_path)
    design = WORK/'tier4/objective_probe_design'; cold = read(design/'complete.json')
    verified = read(design/'verification.json')
    if cold['spec_sha256'] != primary['spec_sha256'] or cold['test_data_read'] or cold['model_inference']:
        raise ValueError('Cold membership design changed scope')
    if (verified['complete_sha256'] != sha256(design/'complete.json') or
        not verified['source_artifact_code_hashes_verified'] or verified['test_data_read'] or
        verified['model_fitting'] or verified['model_inference']):
        raise ValueError('Cold cohort independent verification is absent or inconsistent')
    files = {str(primary_path): sha256(primary_path), str(design/'complete.json'): sha256(design/'complete.json'),
             str(design/'verification.json'): sha256(design/'verification.json')}
    files.update(cold['source_sha256'])
    files.update({str(design/name): h for name, h in cold['artifacts_sha256'].items()})
    for name, digest in cold['code_sha256'].items():
        checked(Path(__file__).with_name(name), digest)
        files[str(Path(__file__).with_name(name))] = digest
    for path, digest in files.items(): checked(path, digest)
    checked(ROOT/'Paired input ranking generalization.md', primary['spec_sha256'])
    components = pd.read_parquet(WORK/'tier1/data/assay_units.parquet').set_index('assay_id').component
    if not components.index.is_unique: raise ValueError('Ambiguous indivisible unit mapping')
    output.parent.mkdir(parents=True, exist_ok=True); archive_directory(output)
    regimes, artifacts = {}, {}
    for regime in ('A', 'B'):
        config_path = WORK/'controlled/configs'/f'full__{regime}__seed42__lr1p0.yaml'
        job_path = WORK/'controlled/jobs'/f'full__{regime}__seed42__lr1p0.json'
        job = read(job_path)
        checked(config_path, job['config_sha256'])
        if (job['plan_sha256'] != sha256(primary_path) or job['id'] != f'full__{regime}__seed42__lr1p0' or
            Path(job['config_path']) != config_path): raise ValueError('Reference configuration belongs to another plan')
        files[str(config_path)] = sha256(config_path)
        files[str(job_path)] = sha256(job_path)
        config = load_reward_training_config(str(config_path))
        contexts = {}
        for role, split in (('train', 'train'), ('cold_validation', 'val')):
            source_path = Path(primary['bases'][regime]['data'][split+'_parquet_path'])
            checked(source_path, primary['source_data_sha256'][regime][split])
            files[str(source_path)] = primary['source_data_sha256'][regime][split]
            cache = Path(primary['bases'][regime]['data']['tokenized_dataset_dir'])/(split+'_examples')
            for name in ('state.json', 'dataset_info.json'):
                digest = primary['tokenized_data_sha256'][regime][split][name]
                checked(cache/name, digest); files[str(cache/name)] = digest
            examples = load_from_disk(str(cache))
            metadata = aligned_metadata(pd.read_parquet(source_path, columns=RAW_COLUMNS), examples, regime, components)
            if role == 'cold_validation':
                examples, metadata = cold_view(examples, metadata, pd.read_parquet(design/f'validation_rows_{regime}.parquet'))
            dataset, batches = sampler_layout(examples, config, role)
            contexts[role] = (dataset, batches, metadata)
        train_units, val_units = (set(contexts[role][2].unit) for role in ('train', 'cold_validation'))
        if train_units & val_units: raise ValueError('Training and validation contexts overlap in units')
        count = min(BATCH_PAIRS, *(len(value[1]) for value in contexts.values()))
        records = {}
        for role, (dataset, batches, metadata) in contexts.items():
            record, rows = freeze_batches(dataset, batches, metadata, count=count)
            record.update(role=role, regime=regime, context_seed=CONTEXT_SEED, epoch=0,
                local_items=LOCAL_ITEMS, item_count_multiple=dataset.item_count_multiple,
                source_rows=len(metadata), source_units=int(metadata.unit.nunique()))
            prefix = regime+'_'+role
            write_json(output/(prefix+'.json'), record); rows.to_parquet(output/(prefix+'_rows.parquet'), index=False)
            for name in (prefix+'.json', prefix+'_rows.parquet'): artifacts[name] = sha256(output/name)
            records[role] = {k:v for k,v in record.items() if k != 'batches'}
            records[role]['selected_unique_rows'] = len(rows)
            records[role]['selected_unique_units'] = int(rows.unit.nunique())
        regimes[regime] = dict(batch_pairs=count, contexts=records)
        del contexts, dataset, examples, metadata
    for path, digest in files.items(): checked(path, digest)
    if software() != code: raise ValueError('Batch preparation source or packages changed during execution')
    result = dict(format_version=1, experiments=['E56', 'E57'], spec_sha256=primary['spec_sha256'],
        input_sha256=files, **code, artifacts_sha256=artifacts, regimes=regimes,
        context_policy='One seed42 epoch-zero context set shared by later checkpoint/seed comparisons. Native local 12-item lists. Validation lists are built only inside frozen own-regime cold units, not ordinary 32-row evaluation batches.',
        interpretation='Fixed diagnostic contexts, not full-dataset metrics, historical device execution, independent units or a seed-variance estimate. Eligibility and inactive losses remain visible; no resampling to obtain nonzero gradients.',
        remaining='Tier3 results; selected checkpoint and complete token-payload verification; freeze step sizes; permitted-GPU numerical qualification; actual per-block gradients and cold-validation transfer.',
        token_payloads_read=False, token_payloads_rehashed=False, test_data_read=False, model_inference=False,
        new_fits_created=0, scientific_probe_results=0, study_complete=False)
    result['payload_sha256'] = digest_json(result)
    write_json(output/'complete.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args()
    print(json.dumps(prepare()), flush=True)
