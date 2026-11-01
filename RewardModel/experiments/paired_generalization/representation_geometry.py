"""E54/E55 geometry definitions and fixed identity-only evaluation cohorts.

The CLI prepares memberships only. Scientific inference remains behind Tier 3.
The metric kernels and projection hooks are also usable in CPU qualification.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .prepare_random import archive_directory
else:
    from run import ROOT, WORK, sha256, write_json
    from prepare_random import archive_directory

IDENTITY = ['group_id', 'target_chembl_id', 'compound_id']
SAMPLE_CAP = 25600
PAIR_BUDGET = 262144
SEED = 42


def priority(namespace, identity, seed=SEED):
    return hashlib.sha256(json.dumps([seed, namespace, identity], separators=(',', ':')).encode()).hexdigest()


def sampling_design(frame, regime, partition, *, cap=SAMPLE_CAP, seed=SEED):
    """Return row indices in three fixed distributions, never using labels."""
    if regime not in ('A', 'B') or partition not in ('train', 'val', 'test'):
        raise ValueError('Unknown regime or partition')
    if not isinstance(cap, int) or cap < 1:
        raise ValueError('Positive integer sample cap required')
    if frame.empty or frame[IDENTITY].isna().any().any() or frame.duplicated(IDENTITY).any():
        raise ValueError('Expected nonempty unique observation identities')
    rows = frame[IDENTITY].astype(str).copy()
    rows['source_row_index'] = np.arange(len(rows))
    # A representative occurrence is chosen by identity, never input order.
    rows = rows.sort_values(IDENTITY)
    namespace = regime+'/'+partition
    outputs, summary = [], {}
    for view, identity_column in [('observations', None), ('unique_proteins', 'target_chembl_id'),
                                  ('unique_molecules', 'compound_id')]:
        candidates = rows if identity_column is None else rows.drop_duplicates(identity_column)
        identities = (candidates[IDENTITY].values.tolist() if identity_column is None
                      else candidates[identity_column].tolist())
        keys = [priority(namespace+'/'+view, value, seed) for value in identities]
        chosen = candidates.assign(priority=keys).sort_values(['priority', *IDENTITY]).head(cap).copy()
        chosen['view'] = view
        chosen['view_row_index'] = np.arange(len(chosen))
        outputs.append(chosen.drop(columns='priority'))
        summary[view] = dict(population=len(candidates), sampled=len(chosen), cap=cap)
    return pd.concat(outputs, ignore_index=True), summary


def matrix(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not all(result.shape) or not np.isfinite(result).all():
        raise ValueError('Expected a finite nonempty representation matrix')
    return result


def normalized(values):
    values = matrix(values)
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError('Representation norm overflow')
    valid = norms > 0
    result = np.zeros_like(values)
    result[valid] = values[valid]/norms[valid, None]
    return result, valid


def effective_rank(values):
    """Uncentered singular-value entropy, not covariance-eigenvalue entropy.

    Implements the zero-epsilon limit of Garrido et al. Eq.1/2, treating
    0*log(0) as zero. An all-zero matrix has rank zero by explicit convention.
    """
    values = matrix(values)
    # Global scaling preserves the singular probability distribution.
    scale = np.abs(values).max()
    singular = (np.linalg.svd(values/scale, compute_uv=False) if scale else
                np.zeros(min(values.shape)))
    total = singular.sum()
    probabilities = singular[singular > 0]/total if total else np.array([])
    rank = float(np.exp(-np.sum(probabilities*np.log(probabilities)))) if total else 0.
    return dict(value=rank, rows=len(values), dimension=values.shape[1],
                maximum_rank=min(values.shape), centered=False,
                spectrum='singular values', zero_matrix_convention=0.)


def pair_indices(rows, *, budget=PAIR_BUDGET, seed=SEED):
    if not isinstance(rows, int) or rows < 1 or not isinstance(budget, int) or budget < 1:
        raise ValueError('Positive integer row count and pair budget required')
    count = rows*(rows-1)//2
    if count <= budget:
        left, right = np.triu_indices(rows, 1)
        method = 'all unordered distinct-row pairs'
    else:
        rng = np.random.default_rng(seed)
        left = rng.integers(rows, size=budget)
        right = rng.integers(rows-1, size=budget)
        right += right >= left
        left, right = np.minimum(left, right), np.maximum(left, right)
        method = 'uniform unordered distinct-row pairs sampled with replacement'
    pairs = np.column_stack([left, right]).astype('<i8')
    return pairs, dict(population_pairs=count, requested_pairs=len(pairs), method=method,
                       seed=seed, pairs_sha256=hashlib.sha256(pairs.tobytes()).hexdigest())


def uniformity(values, *, budget=PAIR_BUDGET, seed=SEED, block_size=2048):
    """Wang-Isola log mean exp(-2*distance^2), on normalized rows.

    Pairs are fixed before checking zero norms. Zero-vector exclusions and
    resulting coverage remain visible; no jitter fabricates sphere directions.
    """
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError('Positive integer block size required')
    values, valid = normalized(values)
    pairs, metadata = pair_indices(len(values), budget=budget, seed=seed)
    total, count = 0., 0
    for start in range(0, len(pairs), block_size):
        selected = pairs[start:start+block_size]
        selected = selected[valid[selected[:, 0]] & valid[selected[:, 1]]]
        distances = np.sum((values[selected[:, 0]]-values[selected[:, 1]])**2, axis=1)
        total += float(np.exp(-2.*distances).sum())
        count += len(selected)
    return dict(**metadata, value=float(np.log(total/count)) if count else None,
        valid_pairs=count, valid_rows=int(valid.sum()), rows=len(values), temperature=2.,
        pair_coverage=count/len(pairs) if len(pairs) else None,
        interpretation='Descriptive fixed-cohort statistic. Sampled log-mean is a Monte Carlo estimate; no seed, unit or pair-sampling confidence interval is implied.')


def alignment(protein, molecule, *, stage='contrastive_projected', selected=None):
    """Squared distance of normalized paired views in their learned shared space."""
    if stage != 'contrastive_projected':
        raise ValueError('Preprojection encoder coordinates do not share an alignment space')
    protein, pv = normalized(protein)
    molecule, mv = normalized(molecule)
    if protein.shape != molecule.shape:
        raise ValueError('Paired projected representations must have identical shapes')
    if selected is None:
        selected = np.ones(len(protein), dtype=bool)
    selected = np.asarray(selected)
    if selected.dtype != bool or selected.shape != (len(protein),):
        raise ValueError('Alignment selection must be an aligned boolean mask')
    valid = pv & mv & selected
    distances = np.sum((protein[valid]-molecule[valid])**2, axis=1)
    return dict(value=float(distances.mean()) if len(distances) else None,
                requested_pairs=int(selected.sum()), valid_pairs=int(valid.sum()),
                alpha=2., stage=stage)


@contextmanager
def capture_projection_stages(model):
    """Capture the *contrastive* pooled branch, not token/fusion projections."""
    if model.training:
        raise ValueError('Geometry capture requires model.eval()')
    captured, handles = {}, []
    try:
        for side in ('protein', 'molecule'):
            projection = getattr(model, 'contrastive_'+side+'_projection', None)
            if projection is None:
                raise ValueError('Expected the reference contrastive projection pathway')

            def hook(_module, inputs, output, *, side=side):
                if inputs[0].ndim != 2 or output.ndim != 2:
                    raise ValueError('Expected pooled vectors at both projection boundaries')
                for stage, value in [('encoder_pooled', inputs[0]), ('contrastive_projected', output)]:
                    key = side+'_'+stage
                    if key in captured:
                        raise ValueError('Capture context must contain exactly one model forward')
                    captured[key] = value.detach().float().cpu().numpy().copy()

            handles.append(projection.register_forward_hook(hook))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def main():
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use CPU-only preparation in authorized Atum Prot2Mol')
    directory = WORK/'tier4/geometry_design'
    if directory.exists():
        raise RuntimeError('Existing geometry design requires verification before reuse')
    primary_path = WORK/'controlled/plan.json'
    primary = json.loads(primary_path.read_text())
    if sha256(ROOT/'Paired input ranking generalization.md') != primary['spec_sha256']:
        raise ValueError('The governing MD changed')
    sources, outputs, summaries = {str(primary_path):sha256(primary_path)}, {}, {}
    for regime in ('A', 'B'):
        for partition in ('train', 'val', 'test'):
            path = Path(primary['bases'][regime]['data'][partition+'_parquet_path'])
            sources[str(path)] = sha256(path)
            if sources[str(path)] != primary['source_data_sha256'][regime][partition]:
                raise ValueError('Frozen source partition changed')
            frame = pd.read_parquet(path, columns=['assay_id', 'target_chembl_id', 'compound_id'])
            frame['group_id'] = frame.target_chembl_id.astype(str)+'__'+frame.assay_id.astype(str)
            key = regime+'_'+partition
            outputs[key], summaries[key] = sampling_design(frame, regime, partition)
            print(json.dumps(dict(partition=key, cohorts=summaries[key])), flush=True)
    directory.parent.mkdir(parents=True, exist_ok=True)
    archive_directory(directory)
    for key, table in outputs.items():
        table.to_parquet(directory/(key+'.parquet'), index=False)
    write_json(directory/'complete.json', dict(format_version=1, experiment_ids=['E54','E55'],
        spec_sha256=primary['spec_sha256'], primary_plan_sha256=sha256(primary_path),
        source_sha256=sources, code_sha256=sha256(Path(__file__)),
        artifacts_sha256={key+'.parquet':sha256(directory/(key+'.parquet')) for key in outputs},
        cohorts=summaries, sample_cap=SAMPLE_CAP, pair_budget=PAIR_BUDGET, seed=SEED,
        stages=['encoder_pooled', 'contrastive_projected'],
        rank_definition='Uncentered raw-vector singular-value entropy; no covariance squaring. Zero-epsilon limit with 0 log 0 = 0; all-zero matrix rank0 by convention.',
        uniformity_definition='Row-L2-normalized log mean exp(-2 squared_distance), fixed distinct-row pairs; zero-vector pair exclusions explicit.',
        alignment_definition='Postprojection normalized paired squared distance, all sampled observations and separately native float32 pChEMBL>6 pairs. Active-pair alignment uses labels and is not label-free. No artificial cross-modal comparison of unrelated encoder coordinate systems.',
        sampling_policy='Identity-hash samples up to25600 per partition/view. Observation-weighted and unique-protein/molecule views are distinct distributions; representative occurrences chosen by canonical observation identity. Same rows and pairs across checkpoints; no score-dependent sampling.',
        interpretation='Representation geometry alone does not establish ranking transfer, gradient conflict or model superiority. Counts/dimensions bound rank and differ across cohorts; compare within matched cohorts/stages.',
        fitting_gate='Membership and numerical preparation only. Complete Tier3 comparisons and qualify frozen checkpoint/row/token provenance and permitted-GPU inference before scientific geometry execution.',
        test_predictions_read=False, labels_read=False, model_fitting=False, model_inference=False))
    return directory


if __name__ == '__main__':
    print(main())
