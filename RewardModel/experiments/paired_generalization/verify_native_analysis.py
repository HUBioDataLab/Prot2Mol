"""Independently check and freeze one native-analysis trajectory prefix on CPU.

Correlations use average ranks and centered dot products, rather than the
analysis implementation's Spearman call. Noise counts enumerate real pairs in
one bounded-size assay per available size bin. These checks do not independently
reconstruct bootstrap intervals, temperature fitting or every NDCG value.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

if __package__:
    from .run import ROOT, WORK, sha256
    from .control_plan import _write_immutable
else:
    from run import ROOT, WORK, sha256
    from control_plan import _write_immutable

IDENTITY = ['group_id', 'target_chembl_id', 'compound_id']
LABELS = ['pchembl_value', 'binary_label']
SCORERS = ('ranking_score', 'activity_logits', 'cosine_similarity')
PREFIXES = ('eval', 'train_eval')


def read(path):
    return json.loads(Path(path).read_text())


def require_hash(path, expected):
    if sha256(Path(path)) != expected:
        raise ValueError(f'Artifact hash changed: {path}')


def same_number(actual, expected, label):
    a = np.nan if actual is None else float(actual)
    b = np.nan if expected is None else float(expected)
    if not np.isclose(a, b, atol=1e-14, rtol=0, equal_nan=True):
        raise ValueError(f'Independent numerical check failed: {label}: {a} != {b}')


def correlation(labels, scores):
    y, s = np.asarray(labels, float), np.asarray(scores, float)
    if len(y) < 2 or np.ptp(y) == 0 or np.ptp(s) == 0:
        return np.nan
    y, s = rankdata(y, method='average'), rankdata(s, method='average')
    y, s = y-y.mean(), s-s.mean()
    return float(np.dot(y, s)/np.sqrt(np.dot(y, y)*np.dot(s, s)))


def enumerated_pairs(labels, scores, epsilon):
    counts = dict(all_pairs=0, comparable_pairs=0, concordant=0, discordant=0,
                  prediction_ties=0, label_band_pairs=0)
    for i, j in itertools.combinations(range(len(labels)), 2):
        counts['all_pairs'] += 1
        dy, ds = float(labels[i])-float(labels[j]), float(scores[i])-float(scores[j])
        if abs(dy) <= epsilon:
            counts['label_band_pairs'] += 1
            continue
        counts['comparable_pairs'] += 1
        key = 'prediction_ties' if ds == 0 else ('concordant' if dy*ds > 0 else 'discordant')
        counts[key] += 1
    return counts


def snapshot_frame(snapshot, reference, reference_info):
    prefix = snapshot['prefix']
    if prefix not in PREFIXES:
        raise ValueError('Only validation and fixed training-probe predictions may be read')
    source = snapshot['source']
    path = Path(source['manifest'])
    require_hash(path, source['manifest_sha256'])
    manifest = read(path)
    if (manifest['metric_prefix'] != prefix or manifest['phase'] != 'trajectory' or
        manifest['epoch'] != snapshot['epoch'] or manifest['step'] != snapshot['step'] or
        manifest['dataset_fingerprint'] != reference_info['fingerprint'] or
        manifest['metrics'] != snapshot['native_metrics'] or
        manifest['ranking_readout'] != snapshot['ranking_readout']):
        raise ValueError('Native snapshot metadata differs from the analyzed snapshot')
    name = manifest['predictions_file']
    if Path(name).name != name:
        raise ValueError('Prediction payload must belong to its manifest directory')
    payload = path.parent/name
    require_hash(payload, manifest['predictions_sha256'])
    if manifest['predictions_sha256'] != source['payload_sha256']:
        raise ValueError('Analyzed payload digest differs from the native manifest')
    with np.load(payload, allow_pickle=False) as arrays:
        if not np.array_equal(arrays['row_index'], np.arange(len(reference))):
            raise ValueError('Native rows are not an ordered complete reference')
        frame = pd.DataFrame({k: arrays[k].copy() for k in arrays.files})
    if len(frame) != manifest['rows'] or len(frame) != snapshot['rows']:
        raise ValueError('Native snapshot row count differs')
    if not np.isfinite(frame[list(SCORERS)+LABELS].to_numpy()).all():
        raise ValueError('Native snapshot contains nonfinite scores or labels')
    for name in IDENTITY:
        if not np.array_equal(frame[name].astype(str), reference[name].astype(str)):
            raise ValueError(f'Native row identity/order differs: {name}')
    for name in LABELS:
        expected = reference[name].to_numpy(dtype=np.float32).astype(np.float64)
        if not np.array_equal(frame[name].to_numpy(dtype=np.float64), expected):
            raise ValueError(f'Native float32 labels differ: {name}')
    return frame


def verify_snapshot(snapshot, directory, reference, reference_info, noise_rms,
                    composition_groups=None, composition_cells=None):
    frame = snapshot_frame(snapshot, reference, reference_info)
    directory = Path(directory)
    for name, digest in snapshot['artifacts_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Analysis artifact must belong to its snapshot directory')
        require_hash(directory/name, digest)
    indices = frame.groupby('group_id', sort=True).indices
    group_ids = list(indices)
    sizes = np.array([len(indices[g]) for g in group_ids])
    labels = frame.pchembl_value.to_numpy(dtype=np.float64)
    eligible = np.array([len(ii) >= 3 and np.ptp(labels[ii]) >= .5 for ii in indices.values()])
    independent, macros, checks = {}, {}, 0
    for scorer in SCORERS:
        scores = frame[scorer].to_numpy(dtype=np.float64)
        rho = np.array([correlation(labels[ii], scores[ii]) for ii in indices.values()])
        table = pd.read_parquet(directory/f'groups_{scorer}.parquet')
        if (table.group_id.tolist() != group_ids or
            not np.array_equal(table['size'], sizes) or
            not np.array_equal(table.original_ranking_eligible, eligible)):
            raise ValueError('Scorer group identity, size or eligibility differs')
        if not np.allclose(rho, table.spearman, atol=1e-14, rtol=0, equal_nan=True):
            raise ValueError('Independent group correlations disagree')
        finite = eligible & np.isfinite(rho)
        macro = float(rho[finite].mean()) if finite.any() else None
        summary = snapshot['metrics'][scorer]
        same_number(macro, summary['original_eligible_macro_spearman'], scorer)
        if summary['groups'] != len(group_ids) or summary['eligible_groups'] != int(eligible.sum()):
            raise ValueError('Reported scorer cohort counts differ')
        defined = float(finite.sum()/eligible.sum()) if eligible.any() else None
        same_number(defined, summary['eligible_defined_fraction'], scorer+' defined fraction')
        macros[scorer], independent[scorer] = macro, rho
        if scorer == 'ranking_score':
            same_number(macro, snapshot['native_metrics'][snapshot['prefix']+'_macro_spearman'], 'native macro')
            native = snapshot['native_metrics']
            if int(native[snapshot['prefix']+'_spearman_num_groups']) != int(eligible.sum()):
                raise ValueError('Native eligible cohort count differs')
            same_number(defined, native[snapshot['prefix']+'_spearman_defined_fraction'], 'native defined fraction')
        # Include a singleton check when available, and one nontrivial bounded
        # assay from every other bin. Larger groups are covered by rank checks.
        bounds = [(1,1), (2,2), (3,3), (4,10), (11,25), (26,128)]
        selected = [np.flatnonzero((sizes >= lo) & (sizes <= hi))[0]
                    for lo, hi in bounds if ((sizes >= lo) & (sizes <= hi)).any()]
        for band, epsilon in [('training_replicate_rms', noise_rms), ('reference_margin', float(np.log10(3.)))]:
            pairs = pd.read_parquet(directory/f'noise_{band}_{scorer}.parquet')
            if pairs.group_id.tolist() != group_ids or not np.array_equal(pairs['size'], sizes):
                raise ValueError('Noise-band group identity or size differs')
            for position in selected:
                ii = indices[group_ids[position]]
                expected = enumerated_pairs(labels[ii], scores[ii], epsilon)
                if any(int(pairs.iloc[position][k]) != value for k, value in expected.items()):
                    raise ValueError('Independent enumerated noise-band pairs disagree')
                checks += 1
    weighted = None
    if snapshot['composition_standardization'] is not None:
        if composition_groups is None or composition_cells is None:
            raise ValueError('Composition standardization requires its frozen design')
        groups = composition_groups.set_index('group_id')
        if set(groups.index) != set(group_ids) or not groups.index.is_unique:
            raise ValueError('Composition group identities differ')
        groups = groups.loc[group_ids]
        if not np.array_equal(groups.eligible, eligible):
            raise ValueError('Composition eligibility differs')
        lookup = composition_cells.set_index('cell').A_group_weight
        weights = groups.cell.map(lookup).fillna(0).to_numpy()*eligible
        rho = independent['ranking_score']
        for name, w in [('raw', eligible.astype(float)), ('common', (weights > 0).astype(float)), ('standardized', weights)]:
            valid = np.isfinite(rho) & (w > 0)
            mean = float(np.average(rho[valid], weights=w[valid])) if valid.any() else None
            same_number(mean, snapshot['composition_standardization']['scores'][name]['score'], 'composition '+name)
            if name == 'standardized':
                weighted = mean
    return dict(epoch=snapshot['epoch'], prefix=snapshot['prefix'], rows=len(frame), groups=len(group_ids),
        eligible_groups=int(eligible.sum()), scorer_group_correlations_checked=len(group_ids)*len(SCORERS),
        brute_force_band_group_checks=checks, macros=macros, composition_standardized=weighted)


def inherit_previous(directory, data, previous_epoch):
    """Reuse numerical evidence only for an identical, fully checked prefix."""
    if previous_epoch is None:
        return [], None, []
    if not 1 <= previous_epoch < len(data['trajectory']):
        raise ValueError('Previous epoch must be an earlier verified prefix')
    old_path = directory/f'trajectory_through_epoch{previous_epoch:02d}.json'
    verification_path = directory/f'verification_through_epoch{previous_epoch:02d}.json'
    old, verified = read(old_path), read(verification_path)
    require_hash(old_path, verified['trajectory_sha256'])
    if (old['analysis_plan_sha256'] != data['analysis_plan_sha256'] or
        verified['analysis_plan_sha256'] != data['analysis_plan_sha256'] or
        old['trajectory'] != data['trajectory'][:previous_epoch] or
        not verified['source_and_artifact_hashes_verified'] or verified['test_predictions_read']):
        raise ValueError('Previous verified prefix differs')
    previous, checks = old['trajectory'], list(verified['checks'])
    expected = {(r['epoch'], p) for r in previous for p in PREFIXES}
    if len(checks) != len(expected) or {(r['epoch'], r['prefix']) for r in checks} != expected:
        raise ValueError('Previous verification does not cover both snapshots of every epoch')
    return previous, sha256(verification_path), checks


def verify(directory, previous_epoch=None):
    import fcntl
    directory = Path(directory)
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Scientific verification is restricted to CPU-only Atum Prot2Mol')
    if directory.resolve().is_relative_to((WORK/'tier1/native_trajectories').resolve()) is False:
        raise ValueError('Expected a native trajectory analysis directory')
    with (directory/'analysis.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        trajectory_path = directory/'trajectory.json'
        data = read(trajectory_path)
        plan_path = directory/'analysis_plan.json'
        require_hash(plan_path, data['analysis_plan_sha256'])
        plan = read(plan_path)
        job = data['job']
        if Path(job['id']).name != job['id'] or job['regime'] not in ('A', 'B'):
            raise ValueError('Expected a registered primary-grid trajectory')
        job_path = WORK/'controlled/jobs'/f"{job['id']}.json"
        require_hash(job_path, plan['job_metadata_sha256'])
        if read(job_path) != job:
            raise ValueError('Trajectory job differs from its registration')
        require_hash(job['config_path'], job['config_sha256'])
        require_hash(WORK/'controlled/plan.json', job['plan_sha256'])
        require_hash(Path(job['output_dir'])/'launch_provenance.json', plan['launch_sha256'])
        for name, digest in plan['software']['source_sha256'].items():
            require_hash(ROOT/name, digest)
        require_hash(directory/'references/complete.json', plan['reference_sha256'])
        refs = read(directory/'references/complete.json')
        references = {}
        for prefix in PREFIXES:
            path = directory/'references'/f'{prefix}.parquet'
            require_hash(path, refs['references'][prefix]['sha256'])
            references[prefix] = pd.read_parquet(path)
        for source in refs['sources'].values():
            require_hash(source['path'], source['sha256'])
            for name, digest in source['cache_metadata_sha256'].items():
                require_hash(Path(source['cache'])/name, digest)
        require_hash(WORK/'tier1/data/assay_units.parquet', plan['units_sha256'])
        require_hash(WORK/'tier1/replicates/noise_summary.json', plan['noise_summary_sha256'])
        design = WORK/'tier1/similarity_composition/design'
        require_hash(design/'complete.json', plan['similarity_design_sha256'])
        design_record = read(design/'complete.json')
        for path, digest in design_record['source_sha256'].items():
            require_hash(path, digest)
        for name, digest in design_record['artifacts_sha256'].items():
            require_hash(design/name, digest)
        rows = data['trajectory']
        epochs = [r['epoch'] for r in rows]
        if (not rows or epochs != list(range(1, len(rows)+1)) or epochs != data['epochs_analyzed'] or
            len(rows) > data['horizon'] or data['test_predictions_read'] or
            data['complete_horizon_observed'] != (len(rows) == data['horizon'])):
            raise ValueError('Trajectory is not a consistent contiguous prefix')
        previous, predecessor, checks = inherit_previous(directory, data, previous_epoch)
        regime = data['job']['regime']
        groups = pd.read_parquet(design/f'groups_{regime}_val.parquet')
        cells = pd.read_parquet(design/'weights_val.parquet')
        for row in rows:
            for prefix in PREFIXES:
                snapshot = row[prefix]
                dest = directory/f"{prefix}_step{row['step']:08d}"
                if (snapshot['epoch'] != row['epoch'] or snapshot['step'] != row['step'] or
                    snapshot['prefix'] != prefix or snapshot['source']['provenance'] != plan or
                    read(dest/'complete.json') != snapshot):
                    raise ValueError('Trajectory record differs from its committed snapshot')
                if row['epoch'] <= len(previous):
                    # Re-read original arrays/identities and rehash every table;
                    # only independent numerical calculations are inherited.
                    snapshot_frame(snapshot, references[prefix], refs['references'][prefix])
                    for name, digest in snapshot['artifacts_sha256'].items():
                        require_hash(dest/name, digest)
                    continue
                result = verify_snapshot(snapshot, dest, references[prefix], refs['references'][prefix],
                    plan['noise_rms'], groups if prefix == 'eval' else None, cells if prefix == 'eval' else None)
                checks.append(result)
                print(json.dumps(result), flush=True)
        frozen = directory/f'trajectory_through_epoch{len(rows):02d}.json'
        _write_immutable(frozen, trajectory_path.read_text())
        output = directory/f'verification_through_epoch{len(rows):02d}.json'
        result = dict(analysis_plan_sha256=data['analysis_plan_sha256'], trajectory_sha256=sha256(frozen),
            verifier_sha256=sha256(Path(__file__)), predecessor_verification_sha256=predecessor, checks=checks,
            independent_rank_method='Average ranks and centered dot products from original native arrays, every group and all three scorers. Earlier calculations inherited only after byte-identical prefix and rehashed source/output checks.',
            noise_method='Enumerate all pairs in one real assay of each available size bin, capped at128 rows, both bands and all scorers; singleton zero-pair checks included when available.',
            scope='Independent correlation, sampled pair-count and composition-weighted point estimates. Bootstrap intervals, temperature fitting and NDCG are not independently reconstructed here. No token buffers, model weights or test predictions are read.',
            source_and_artifact_hashes_verified=True, test_predictions_read=False)
        if output.exists():
            cached = read(output)
            if {k:v for k,v in cached.items() if k != 'verified_unix'} != result:
                raise ValueError('Existing verification has different evidence or methods')
            return cached
        result['verified_unix'] = time.time()
        _write_immutable(output, json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--previous-epoch', type=int)
    args = parser.parse_args()
    result = verify(args.directory, args.previous_epoch)
    print(json.dumps(dict(trajectory_sha256=result['trajectory_sha256'], snapshots=len(result['checks']))))
