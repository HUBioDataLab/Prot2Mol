"""Audit actual epoch list construction, batch composition and known false negatives.

No encoder/model fitting. Counts are conditional on epoch zero, seed 42 and
the reference 12-item local microbatch. Known positives use only that split's
training measurements, never validation/test labels. They are a lower bound
on unmeasured positives, not a complete biological truth table.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "RewardModel"))
from reward_model.training.config import load_reward_training_config
from reward_model.training.data import RewardAssayListDataset, get_tokenized_split_dataset_paths, load_tokenized_example_dataset
from reward_model.training.trainer import AssayListEpochSampler

if __package__:
    from .run import WORK, sha256, write_json
else:
    from run import WORK, sha256, write_json


def false_negative_counts(group_targets, molecules, owners, active, known_positives):
    group_targets, molecules = np.asarray(group_targets), np.asarray(molecules)
    owners, active = np.asarray(owners), np.asarray(active, bool)
    membership = np.arange(len(group_targets))[:, None] == owners[None, :]
    same_target = group_targets[:, None] == group_targets[owners][None, :]
    duplicate = np.zeros_like(membership)
    for group in range(len(group_targets)):
        duplicate[group] = np.isin(molecules, molecules[owners == group])
    # Own-list co-candidates are also hidden in protein->molecule retrieval;
    # own-list cells are targets rather than negatives in molecule->protein.
    negatives = ~membership & ~(same_target | duplicate)
    known = np.asarray([[m in known_positives.get(t, set()) for m in molecules] for t in group_targets])
    false_negatives = negatives & known
    positives_per_group = np.bincount(owners[active], minlength=len(group_targets))
    result = {
        "active_anchors": int(active.sum()),
        "protein_to_molecule_negative_comparisons": int((negatives.sum(1)*positives_per_group).sum()),
        "protein_to_molecule_known_false_negatives": int((false_negatives.sum(1)*positives_per_group).sum()),
        "molecule_to_protein_negative_comparisons": int(negatives[:, active].sum()),
        "molecule_to_protein_known_false_negatives": int(false_negatives[:, active].sum()),
        "unmasked_same_target_negatives": int((negatives & same_target).sum()),
        "unmasked_duplicate_molecule_negatives": int((negatives & duplicate).sum()),
        "same_target_cross_list_cells_masked": int((~membership & same_target).sum()),
        "duplicate_molecule_cross_list_cells_masked": int((~membership & duplicate).sum()),
    }
    return result


def distribution(values):
    values = np.asarray(values, float)
    return dict(count=len(values), mean=float(values.mean()), std=float(values.std()),
                quantiles=dict(zip(["min", "p05", "p25", "median", "p75", "p95", "max"],
                                   np.quantile(values, [0, .05, .25, .5, .75, .95, 1]).tolist())))


def main(regime, false_negative_batches=512):
    if ROOT != Path('/home/atabey/Prot2Mol'):
        raise RuntimeError('Dataset audit restricted to the authorized Atum project')
    if false_negative_batches <= 0:
        raise ValueError('false_negative_batches must be positive')
    torch.set_num_threads(4)
    config_path = ROOT / ('RewardModel/configs/reward_train_paired_generalization_random.yaml' if regime == 'A'
                          else 'RewardModel/configs/reward_train_paired_generalization_baseline.yaml')
    config = load_reward_training_config(str(config_path))
    output = WORK / 'tier1/sampling' / regime
    output.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(Path(config.data.train_parquet_path))
    examples = load_tokenized_example_dataset(get_tokenized_split_dataset_paths(config.data.tokenized_dataset_dir)['train'])
    source = pd.read_parquet(config.data.train_parquet_path, columns=['target_chembl_id', 'assay_id', 'compound_id', 'pchembl_value'])
    metadata = pd.DataFrame({key: examples[key] for key in ['group_id', 'target_chembl_id', 'compound_id', 'pchembl_value']})
    if len(source) != len(metadata):
        raise RuntimeError('Token cache/source row count mismatch')
    for key in ['target_chembl_id', 'compound_id']:
        if not np.array_equal(source[key].astype(str), metadata[key].astype(str)):
            raise RuntimeError(f'Token cache/source identity mismatch: {key}')
    if not np.array_equal(source.pchembl_value.to_numpy('float32'), metadata.pchembl_value.to_numpy('float32')):
        raise RuntimeError('Token cache/source continuous-label mismatch')
    known = defaultdict(set)
    for target, molecule in source.loc[source.pchembl_value > 6., ['target_chembl_id', 'compound_id']].itertuples(index=False, name=None):
        known[str(target)].add(str(molecule))
    dataset = RewardAssayListDataset(examples, seed=config.training.seed,
        ranking_max_ligands=config.data.ranking_max_ligands,
        ranking_opportunity_divisor=config.data.ranking_opportunity_divisor,
        ranking_min_pchembl_span=config.data.ranking_min_pchembl_span,
        max_classification_only_per_item=config.data.max_classification_only_per_item,
        include_all_assays_for_contrastive=True, item_count_multiple=4*config.training.per_device_train_batch_size)
    sampler = AssayListEpochSampler(dataset, seed=config.training.seed, length_bucketing=config.training.length_bucketing,
        batch_size=config.training.per_device_train_batch_size, bucket_size_multiplier=config.training.length_bucket_size_multiplier)
    order = [index for _, index in sampler]
    local_size = config.training.per_device_train_batch_size
    batches = [order[start:start+local_size] for start in range(0, len(order), local_size)]
    selected = set(np.linspace(0, len(batches)-1, min(false_negative_batches, len(batches)), dtype=int).tolist())
    values = metadata.pchembl_value.to_numpy('float32')
    targets, molecules = metadata.target_chembl_id.astype(str).to_numpy(), metadata.compound_id.astype(str).to_numpy()
    records, false_records, target_counts = [], [], Counter()
    for number, items in enumerate(batches):
        lists = [entry for item in items for entry in dataset._bundles[item][0]]
        classification_rows = sum(len(dataset._bundles[item][1]) for item in items)
        ranked = sum(bool(flag) for _, _, flag in lists)
        records.append(dict(batch=number, items=len(items), lists=len(lists), ranked_lists=ranked,
                            ranking_fraction=ranked/len(lists) if lists else 0., classification_only_rows=classification_rows))
        for _, rows, _ in lists:
            target_counts[targets[rows[0]]] += 1
        if number in selected:
            indices = np.asarray([i for _, rows, _ in lists for i in rows], dtype=int)
            owners = np.repeat(np.arange(len(lists)), [len(rows) for _, rows, _ in lists])
            group_targets = [targets[rows[0]] for _, rows, _ in lists]
            counts = false_negative_counts(group_targets, molecules[indices], owners, values[indices] > 6., known)
            false_records.append(dict(batch=number, **counts))
    batch_table, false_table = pd.DataFrame(records), pd.DataFrame(false_records)
    batch_table.to_parquet(output/'epoch0_batches.parquet', index=False)
    false_table.to_parquet(output/'known_false_negative_batches.parquet', index=False)
    pd.DataFrame(sorted(target_counts.items()), columns=['target_chembl_id', 'assay_list_opportunities']).to_parquet(output/'target_opportunities.parquet', index=False)
    totals = {key: int(false_table[key].sum()) for key in false_table if key != 'batch'}
    for direction in ['protein_to_molecule', 'molecule_to_protein']:
        count, denominator = totals[direction+'_known_false_negatives'], totals[direction+'_negative_comparisons']
        totals[direction+'_known_false_negative_fraction'] = count/denominator if denominator else None
        totals[direction+'_known_false_negatives_per_active_anchor'] = count/totals['active_anchors'] if totals['active_anchors'] else None
    frequency = np.asarray(sorted(target_counts.values(), reverse=True), float)
    result = dict(regime=regime, train_source_sha256=source_hash, config_sha256=sha256(config_path),
        source_rows_verified_against_token_cache=True, epoch=0, seed=config.training.seed, world_size=4,
        local_microbatch_items=local_size, sampling=asdict(dataset.stats),
        batch_ranking_fraction=distribution(batch_table.ranking_fraction),
        zero_ranking_batches=int((batch_table.ranked_lists==0).sum()),
        batches=len(batch_table), false_negative_batches=len(false_table),
        known_positive_policy='Any curated training assay observation with pchembl > 6; no validation/test labels',
        known_training_positive_pairs=sum(map(len, known.values())), false_negative_counts=totals,
        target_opportunity_distribution=distribution(frequency),
        target_top_one_percent_opportunity_share=float(frequency[:max(1, int(np.ceil(len(frequency)*.01)))].sum()/frequency.sum()),
        sampling_probability_correction='none in the reference contrastive objective',
        scope='All global epoch-zero local batches; deterministic evenly spaced subset for candidate false-negative counts. This audits the configured sampler, not a recording of historical device execution. No prevalence estimate for unmeasured positives.',
        projection_path='Both contrastive projections use pooled encoder output before cross-attention; R and B use fused representations',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(output/'complete.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regime', choices=['A','B'], required=True)
    parser.add_argument('--false-negative-batches', type=int, default=512)
    arguments = parser.parse_args()
    main(arguments.regime, arguments.false_negative_batches)
