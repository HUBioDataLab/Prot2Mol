"""Fixed-cohort E54/E55 representation export and numerical analysis.

This is a worker library, not a scientific launcher. The caller must verify the
Tier3 gate, selected checkpoint and complete source/token/cohort hashes, and
hold the shared GPU lock. CPU calls are explicitly marked qualification only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .control_plan import GPU_UUIDS
    from .checkpoint_evaluation import parameter_digest
    from .representation_geometry import (IDENTITY, PAIR_BUDGET, SEED, alignment,
        capture_projection_stages, effective_rank, uniformity)
else:
    from run import ROOT, WORK, sha256, write_json
    from control_plan import GPU_UUIDS
    from checkpoint_evaluation import parameter_digest
    from representation_geometry import (IDENTITY, PAIR_BUDGET, SEED, alignment,
        capture_projection_stages, effective_rank, uniformity)

STAGES = ('encoder_pooled', 'contrastive_projected')
SIDES = ('protein', 'molecule')
ARRAYS = tuple(side+'_'+stage for side in SIDES for stage in STAGES)
ARTIFACTS = {name+'.npy' for name in ARRAYS} | {'rows.parquet', 'memberships.parquet'}


def align_cohorts(dataset, source_frame, memberships):
    """Map frozen raw-row memberships to cache identities; never assume order."""
    required = IDENTITY+['view', 'view_row_index', 'source_row_index']
    members = memberships[required].copy()
    if members.empty or members.isna().any().any():
        raise ValueError('Nonempty complete geometry memberships required')
    if not all(pd.api.types.is_integer_dtype(members[k]) for k in ('source_row_index', 'view_row_index')):
        raise ValueError('Geometry row indices must be integers')
    if set(members.view) != {'observations', 'unique_proteins', 'unique_molecules'}:
        raise ValueError('Expected all three frozen geometry views')
    for view, group in members.groupby('view'):
        if group.duplicated(IDENTITY).any() or not np.array_equal(np.sort(group.view_row_index), np.arange(len(group))):
            raise ValueError('Duplicated identity or noncontiguous view ordering')
        entity = {'unique_proteins':'target_chembl_id', 'unique_molecules':'compound_id'}.get(view)
        if entity and group[entity].duplicated().any():
            raise ValueError('Unique-entity view repeats an entity')
    indices = members.source_row_index.to_numpy()
    if np.any(indices < 0) or np.any(indices >= len(source_frame)):
        raise ValueError('Membership row lies outside its source partition')
    source_ids = source_frame.iloc[indices][IDENTITY].astype(str).reset_index(drop=True)
    if not source_ids.equals(members[IDENTITY].astype(str).reset_index(drop=True)):
        raise ValueError('Frozen membership identity differs from raw source row')
    unique = members[IDENTITY+['source_row_index']].drop_duplicates()
    if unique.duplicated(IDENTITY).any() or unique.source_row_index.duplicated().any():
        raise ValueError('Ambiguous raw source identity')
    unique = unique.sort_values(IDENTITY).reset_index(drop=True)
    cache = dataset.select_columns(IDENTITY).to_pandas()
    if cache.isna().any().any() or cache.duplicated(IDENTITY).any():
        raise ValueError('Cache observation identities must be unique')
    cache = cache.astype(str)
    cache['cache_row_index'] = np.arange(len(cache))
    rows = unique.merge(cache, on=IDENTITY, how='left', validate='one_to_one', sort=False)
    if rows.cache_row_index.isna().any():
        raise ValueError('Frozen observation is missing from the token cache')
    rows['cache_row_index'] = rows.cache_row_index.astype(np.int64)
    examples = dataset.select(rows.cache_row_index.tolist())
    labels = np.asarray(examples['pchembl_value'], dtype=np.float32)
    raw_labels = source_frame.iloc[rows.source_row_index].pchembl_value.to_numpy(dtype=np.float32)
    if not np.isfinite(labels).all() or not np.array_equal(labels, raw_labels):
        raise ValueError('Cached continuous labels differ from the frozen source')
    rows['pchembl_value'] = labels
    rows['export_row_index'] = np.arange(len(rows))
    members = members.merge(rows[IDENTITY+['export_row_index']], on=IDENTITY,
                            how='left', validate='many_to_one', sort=False)
    members = members.sort_values(['view', 'view_row_index']).reset_index(drop=True)
    return examples, rows, members


def verify_export_files(output):
    output = Path(output)
    manifest = json.loads((output/'export.json').read_text())
    if set(manifest['artifacts_sha256']) != ARTIFACTS:
        raise ValueError('Geometry export artifact inventory changed')
    for name, digest in manifest['artifacts_sha256'].items():
        if sha256(output/name) != digest:
            raise ValueError('Geometry export artifact changed')
    rows = pd.read_parquet(output/'rows.parquet')
    if len(rows) != manifest['rows'] or not np.array_equal(rows.export_row_index, np.arange(len(rows))):
        raise ValueError('Geometry export row ordering changed')
    for name in ARRAYS:
        array = np.load(output/(name+'.npy'), mmap_mode='r', allow_pickle=False)
        if array.shape != tuple(manifest['shapes'][name]) or array.shape[0] != len(rows) or array.dtype != np.float32:
            raise ValueError('Geometry representation shape or precision changed')
    return manifest


def analyze_export(output, *, pair_budget=PAIR_BUDGET):
    output = Path(output)
    manifest = verify_export_files(output)
    rows = pd.read_parquet(output/'rows.parquet')
    members = pd.read_parquet(output/'memberships.parquet')
    arrays = {name:np.load(output/(name+'.npy'), mmap_mode='r', allow_pickle=False) for name in ARRAYS}
    views = {view:g.sort_values('view_row_index').export_row_index.to_numpy()
             for view,g in members.groupby('view')}
    statistics = {}
    for view, indices in views.items():
        sides = SIDES if view == 'observations' else (('protein',) if view == 'unique_proteins' else ('molecule',))
        for side in sides:
            for stage in STAGES:
                values = arrays[side+'_'+stage][indices].astype(np.float64)
                norms = np.linalg.norm(values, axis=1)
                statistics[view+'/'+side+'/'+stage] = dict(
                    rank=effective_rank(values), uniformity=uniformity(values, budget=pair_budget, seed=SEED),
                    norms=dict(minimum=float(norms.min()), median=float(np.median(norms)), maximum=float(norms.max()),
                               below_native_normalization_epsilon=int((norms < 1e-6).sum())))
    selected = views['observations']
    protein, molecule = (arrays[side+'_contrastive_projected'][selected] for side in SIDES)
    active = rows.iloc[selected].pchembl_value.to_numpy(dtype=np.float32) > np.float32(6.)
    return dict(statistics=statistics,
        alignment=dict(all_observed_pairs=alignment(protein, molecule),
                       strict_active_pairs=alignment(protein, molecule, selected=active)),
        active_pair_definition='Float32 continuous pChEMBL>6, not pair-median BCE labels; active alignment is label-conditioned.',
        preprojection_alignment=None,
        preprojection_alignment_limit='Independent encoder coordinates do not share a cross-modal distance space.',
        pair_budget=pair_budget, sample_seed=SEED,
        export_sha256=sha256(output/'export.json'), cpu_qualification=manifest['cpu_qualification'],
        interpretation='Fixed observation/entity cohorts; no checkpoint selection, confidence interval or ranking-performance conclusion. Training geometry is descriptive.')


def verify_complete(output, provenance, *, cpu=False):
    output = Path(output)
    manifest = verify_export_files(output)
    result = json.loads((output/'complete.json').read_text())
    if (manifest['source'] != provenance or result['source'] != provenance or
        manifest['cpu_qualification'] != cpu or result['cpu_qualification'] != cpu or
        not manifest['parameters_unchanged'] or manifest['optimizer_created']):
        raise ValueError('Geometry provenance or execution mode changed')
    if result['analysis']['export_sha256'] != sha256(output/'export.json'):
        raise ValueError('Geometry analysis refers to a changed export')
    if result['analysis']['cpu_qualification'] != cpu:
        raise ValueError('Geometry analysis execution mode changed')
    return result


def export_geometry(model, dataset, source_frame, memberships, output, provenance, *, cpu=False, batch_size=32):
    """Export the four raw representation stages, then compute fixed metrics.

    This function never constructs a Trainer/optimizer or chooses a checkpoint.
    Its caller owns the full scientific provenance/gate and shared GPU lock.
    """
    from reward_model.training.data import RewardPairCollator
    output = Path(output)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('Positive integer inference batch size required')
    if not cpu and (ROOT != Path('/home/atabey/Prot2Mol') or
                    os.environ.get('CUDA_VISIBLE_DEVICES') not in GPU_UUIDS or
                    not output.absolute().is_relative_to(WORK)):
        raise RuntimeError('Scientific geometry requires the authorized Atum project and one assigned GPU')
    if cpu and any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('CPU qualification cannot consume accelerator parameters')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Geometry export uses one independent worker per GPU')
    if output.exists():
        raise RuntimeError('Existing geometry output requires verification or inspection')
    examples, rows, members = align_cohorts(dataset, source_frame, memberships)
    before = parameter_digest(model)
    model.eval()
    device = 'cpu' if cpu else 'cuda'
    model.to(device)
    collator = RewardPairCollator(dynamic_padding=True,
        protein_pad_token_id=model.protein_tokenizer.pad_token_id,
        molecule_pad_token_id=model.molecule_tokenizer.pad_token_id)
    output.mkdir(parents=True)
    matrices, shapes = {}, {}
    max_native_difference = 0.
    for start in range(0, len(examples), batch_size):
        stop = min(start+batch_size, len(examples))
        batch_rows = [examples[i] for i in range(start, stop)]
        inputs = {k:v.to(device) for k,v in collator.collate_example_tokens(batch_rows).items()}
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=not cpu):
            with capture_projection_stages(model) as captured:
                result = model(**inputs, return_dict=True)
        if set(captured) != set(ARRAYS):
            raise ValueError('Expected all four pooled representation stages')
        native_norms = {}
        for name, values in captured.items():
            if not np.isfinite(values).all() or values.shape[0] != stop-start:
                raise ValueError('Incomplete or nonfinite representation batch')
            if name not in matrices:
                shapes[name] = (len(examples), values.shape[1])
                matrices[name] = np.lib.format.open_memmap(output/(name+'.npy'), mode='w+', dtype='float32', shape=shapes[name])
            matrices[name][start:stop] = values
            if name.endswith('_contrastive_projected'):
                # Match the model's epsilon for parity; sphere metrics later
                # normalize nonzero vectors without an epsilon floor.
                native_norms[name.split('_')[0]] = values.astype(np.float64)/np.maximum(np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True), 1e-6)
        cosine = np.sum(native_norms['protein']*native_norms['molecule'], axis=1)
        native = result.cosine_similarity.detach().float().cpu().numpy()
        if not np.isfinite(native).all():
            raise ValueError('Nonfinite native cosine')
        max_native_difference = max(max_native_difference, float(np.max(np.abs(cosine-native))))
    tolerance = 1e-5 if cpu else .01
    if max_native_difference > tolerance:
        raise ValueError('Captured projections do not reproduce the native cosine pathway')
    for array in matrices.values():
        array.flush()
    if parameter_digest(model) != before:
        raise RuntimeError('Geometry inference changed model weights')
    rows.to_parquet(output/'rows.parquet', index=False)
    members.to_parquet(output/'memberships.parquet', index=False)
    write_json(output/'export.json', dict(source=provenance, rows=len(rows), shapes=shapes,
        cache_fingerprint=dataset._fingerprint, selected_cache_fingerprint=examples._fingerprint,
        batch_size=batch_size, cpu_qualification=cpu, parameters_unchanged=True, optimizer_created=False,
        parameter_sha256=before, native_cosine_max_abs_difference=max_native_difference,
        native_cosine_tolerance=tolerance,
        dtype='float32 CPU qualification' if cpu else 'bf16 autocast; raw stages exported float32',
        artifacts_sha256={name:sha256(output/name) for name in sorted(ARTIFACTS)}))
    analysis = analyze_export(output)
    write_json(output/'complete.json', dict(source=provenance, cpu_qualification=cpu, analysis=analysis))
    return verify_complete(output, provenance, cpu=cpu)
