"""Qualify E40 A training views against existing caches; no training is launched."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .control_plan import _write_immutable
    from .leak_design import CONSTRUCTION_VERSION
else:
    from run import ROOT, WORK, sha256, write_json
    from control_plan import _write_immutable
    from leak_design import CONSTRUCTION_VERSION

from reward_model.training.train_views import check_whole_units, load_training_view, subset_pair_labels


def align_subset(metadata, selected):
    keys = ['target_chembl_id','assay_id','compound_id']
    reference = pd.MultiIndex.from_frame(metadata[keys])
    candidate = pd.MultiIndex.from_frame(selected[keys])
    if not reference.is_unique or not candidate.is_unique or not len(candidate):
        raise ValueError('View needs unique nonempty observation identities')
    positions = reference.get_indexer(candidate)
    if (positions<0).any():
        raise ValueError('Selected example does not belong to the source cache')
    order = np.argsort(positions)
    indices = positions[order]
    check_whole_units(metadata.unit.to_numpy(),indices)
    if not np.array_equal(metadata.pchembl_value.to_numpy()[indices],selected.pchembl_value.to_numpy()[order]):
        raise ValueError('Selected continuous labels changed')
    labels = subset_pair_labels(metadata,indices)
    if not np.array_equal(labels,selected.binary_label.to_numpy()[order]):
        raise ValueError('Selected labels are not subset-local pair medians')
    return indices,labels


def main():
    from datasets import load_from_disk
    from reward_model.training.config import RewardTrainingDataConfig
    from reward_model.training.data import RewardAssayListDataset
    if ROOT != Path('/home/atabey/Prot2Mol'):
        raise RuntimeError('Use the authorized Atum project')
    construction = WORK/f'leak_design_v{CONSTRUCTION_VERSION}'
    directory = construction/'views/A'
    if directory.exists():
        raise RuntimeError('Existing view construction requires inspection before retry')
    parent_path = construction/'complete.json'
    parent = json.loads(parent_path.read_text())
    primary_path = WORK/'controlled/plan.json'
    primary = json.loads(primary_path.read_text())
    if sha256(primary_path) != parent['primary_plan_sha256'] or not parent['results']['A']['usable_leak_contrast']:
        raise ValueError('A matching is not qualified against the primary plan')
    curves_path = WORK/'learning_curves/plan.json'
    curves = json.loads(curves_path.read_text())
    endpoint = curves['views']['A']['fraction100']
    if curves['primary_plan_sha256'] != sha256(primary_path) or sha256(Path(endpoint['path'])) != endpoint['sha256']:
        raise ValueError('Full-cache reference changed')
    reference = json.loads(Path(endpoint['path']).read_text())
    metadata_path = Path(reference['source_metadata']['path'])
    if sha256(metadata_path) != reference['source_metadata']['sha256']:
        raise ValueError('Full-cache metadata changed')
    metadata = pd.read_parquet(metadata_path)
    original = load_from_disk(reference['source']['cache_dir'])
    directory.mkdir(parents=True)
    records = []
    for arm in ['leak_out','leak_in']:
        selected_path = construction/f'A_{arm}.parquet'
        if sha256(selected_path) != parent['artifacts_sha256'][selected_path.name]:
            raise ValueError('Matched A data changed')
        selected = pd.read_parquet(selected_path)
        indices,labels = align_subset(metadata,selected)
        indices_path,labels_path = directory/f'{arm}_indices.npy',directory/f'{arm}_labels.npy'
        np.save(indices_path,indices,allow_pickle=False)
        np.save(labels_path,labels,allow_pickle=False)
        record = dict(format_version=1,regime='A',arm=arm,unit_definition='whole assay',
            source=reference['source'],source_metadata=reference['source_metadata'],
            label_policy='within_selected_training_pair_median',activity_threshold=6.,
            indices=dict(path=str(indices_path),sha256=sha256(indices_path)),
            labels=dict(path=str(labels_path),sha256=sha256(labels_path)),
            statistics=parent['results']['A']['training'][arm],
            matched_construction_sha256=sha256(parent_path),selected_rows_sha256=sha256(selected_path))
        path = directory/f'{arm}.json'
        _write_immutable(path,json.dumps(record,sort_keys=True,indent=2)+'\n')
        config = RewardTrainingDataConfig(**primary['bases']['A']['data'])
        config = replace(config,train_view_manifest=str(path),train_view_sha256=sha256(path))
        view,provenance = load_training_view(original,config)
        for column in ['protein_input_ids','protein_attention_mask','molecule_input_ids','molecule_attention_mask']:
            before = {chunk.values.buffers()[1].address for chunk in original.data.column(column).chunks}
            after = {chunk.values.buffers()[1].address for chunk in view.data.column(column).chunks}
            if not after <= before:
                raise ValueError('Matched view copied original token buffers')
        sampler = RewardAssayListDataset(view,seed=42,ranking_max_ligands=config.ranking_max_ligands,
            ranking_opportunity_divisor=config.ranking_opportunity_divisor,
            ranking_min_pchembl_span=config.ranking_min_pchembl_span,
            max_classification_only_per_item=config.max_classification_only_per_item,
            include_all_assays_for_contrastive=True,item_count_multiple=48)
        if sampler.stats.num_eligible_assays != record['statistics']['sampler_eligible_groups']:
            raise ValueError('Matched data and production sampler eligibility differ')
        qualified = dict(arm=arm,**provenance,shared_token_buffers_verified=True,
            sampler_items=len(sampler),sampler_eligible_groups=sampler.stats.num_eligible_assays,
            optimizer_steps_per_epoch=math.ceil(len(sampler)/192))
        records.append(qualified)
        print(json.dumps({key:qualified[key] for key in ['arm','sampler_items','sampler_eligible_groups','optimizer_steps_per_epoch']}),flush=True)
        del sampler,view,selected
    if records[0]['sampler_items'] != records[1]['sampler_items']:
        raise ValueError('Matched training arms have different optimizer budgets')
    write_json(directory/'qualified.json',dict(regime='A',views=records,
        matched_construction_sha256=sha256(parent_path),primary_plan_sha256=sha256(primary_path),
        code_sha256=sha256(Path(__file__)),
        scope='Production cache/label/sampler qualification only. Both scoring-model cells can share each data view. No training or test predictions. B design awaits the user choice.'))


if __name__ == '__main__':
    main()
