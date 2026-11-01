"""Independently check Q11 scalar/listwise tables from saved validation scores.

Runs on CPU from a separate checkout if needed. --root names the authoritative
Atum project; its analysis/model sources are read and never changed. All scalar
groups and list descriptors are checked; production list loss is sampled by
partition/assay-size bin. Bootstrap intervals are outside this verifier's scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import log_expit
from scipy.stats import rankdata


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def checked(path, digest):
    path = Path(path)
    if sha256(path) != digest:
        raise ValueError(f'Artifact changed: {path}')
    return path


def read(path):
    return json.loads(Path(path).read_text())


def close(actual, expected, atol=2e-12):
    if expected is None:
        if actual is not None:
            raise AssertionError('Expected an undefined statistic')
    else:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=atol, equal_nan=True)


def stratum_members(groups):
    """Derive fixed memberships directly, without either audit's mask helper."""
    return {'all': set(groups.group_id),
        **{'size/'+key:set(groups.loc[groups.size_bin == key, 'group_id'])
           for key in ('1', '2', '3', '4-10', '11-25', '>25')},
        **{'scaffold/'+key:set(groups.loc[groups.similarity_class == key, 'group_id'])
           for key in ('all_shared', 'mixed', 'all_unseen')}}


def rank_auc(labels, logits):
    labels = np.asarray(labels)
    positives = int(labels.sum())
    negatives = len(labels)-positives
    if not positives or not negatives:
        return None
    ranks = rankdata(np.asarray(logits, dtype=np.float64))
    return float((ranks[labels == 1].sum()-positives*(positives+1)/2)/(positives*negatives))


def verify_scalar(frame, groups, table, summaries):
    frame = frame.copy()
    frame['pchembl_value'] = frame.pchembl_value.to_numpy(np.float32).astype(np.float64)
    if not frame.binary_label.isin([0, 1]).all() or not np.isfinite(
            frame[['activity_logits', 'ranking_score', 'pchembl_value']]).all().all():
        raise ValueError('Nonfinite scores or invalid binary labels')
    frame['bce'] = np.where(frame.binary_label.to_numpy() == 1,
                           -log_expit(frame.activity_logits.to_numpy(float)),
                           -log_expit(-frame.activity_logits.to_numpy(float)))
    # Exported logits may be float32; accumulate their exact values in float64
    # just as for the independently reconstructed Bernoulli loss above.
    frame['abs_logit'] = np.abs(frame.activity_logits.to_numpy(np.float64))
    groups = groups.sort_values('group_id').reset_index(drop=True)
    table = table.sort_values('group_id').reset_index(drop=True)
    pd.testing.assert_frame_equal(table[groups.columns], groups, check_exact=True)
    observed = []
    for name, row in frame.groupby('group_id', sort=True):
        labels = row.pchembl_value.to_numpy()
        eligible = len(row) >= 3 and float(labels.max()-labels.min()) >= .5
        rho = np.nan
        if eligible and np.ptp(row.ranking_score.to_numpy()) > 0:
            rho = float(np.corrcoef(rankdata(labels), rankdata(row.ranking_score))[0, 1])
        observed.append(dict(group_id=name, size=len(row), eligible=eligible, spearman=rho,
            bce_sum=float(row.bce.sum()), abs_logit_sum=float(row.abs_logit.sum())))
    expected = pd.DataFrame(observed)
    pd.testing.assert_frame_equal(table[['group_id', 'size', 'eligible']],
                                  expected[['group_id', 'size', 'eligible']], check_exact=True)
    close(table.spearman, expected.spearman, atol=1e-14)
    close(table.bce_sum, expected.bce_sum, atol=1e-9)
    close(table.abs_logit_sum, expected.abs_logit_sum, atol=1e-9)
    expected['unit_id'] = groups.unit_id.to_numpy()
    strata = stratum_members(groups)
    if set(summaries) != set(strata):
        raise ValueError('Scalar strata differ from the frozen design')
    for name, members in strata.items():
        rows = frame[frame.group_id.isin(members)]
        selected = expected[expected.group_id.isin(members)]
        stats = summaries[name]
        counts = dict(rows=len(rows), groups=len(selected), units=int(selected.unit_id.nunique()),
            eligible_groups=int(selected.eligible.sum()), defined_groups=int(selected.spearman.notna().sum()))
        if any(stats[key] != value for key, value in counts.items()):
            raise ValueError('Scalar stratum coverage differs')
        close(stats['row_bce'], float(rows.bce.mean()) if len(rows) else None)
        close(stats['assay_macro_bce'], float((selected.bce_sum/selected['size']).mean()) if len(selected) else None)
        close(stats['mean_absolute_logit'], float(rows.abs_logit.mean()) if len(rows) else None)
        close(stats['macro_spearman'], float(selected.spearman.mean()) if selected.spearman.notna().any() else None)
        close(stats['roc_auc'], rank_auc(rows.binary_label, rows.activity_logits))
    return dict(groups_checked=len(table), eligible_groups=int(expected.eligible.sum()),
                strata_checked=len(strata), row_bce=summaries['all']['row_bce'])


def native_lists(frame):
    import torch
    from reward_model.training.evaluation import build_complete_coverage_ranking_partitions
    groups, names = pd.factorize(frame.group_id, sort=True)
    labels = torch.tensor(frame.pchembl_value.to_numpy(copy=True), dtype=torch.float32)
    ids = torch.tensor(groups, dtype=torch.long)
    partitions = build_complete_coverage_ranking_partitions(pchembl_values=labels,
        ranking_group_ids=ids, max_list_size=16, num_partitions=3, seed=42, min_pchembl_span=.5)
    descriptors, members = [], []
    for partition, values in enumerate(partitions):
        values = values.numpy()
        valid = np.flatnonzero(values >= 0)
        order = valid[np.argsort(values[valid], kind='stable')]
        sizes = np.bincount(values[valid])
        if len(sizes) and (sizes.min() < 3 or sizes.max() > 16):
            raise ValueError('Invalid native ranking list size')
        offsets = np.r_[0, sizes.cumsum()]
        for identity, size in enumerate(sizes):
            index = order[offsets[identity]:offsets[identity+1]]
            if len(np.unique(groups[index])) != 1:
                raise ValueError('Native list crosses complete assays')
            descriptors.append(dict(partition=partition, list_id=identity,
                group_id=names[groups[index[0]]], list_size=int(size)))
            members.append(index)
    return pd.DataFrame(descriptors), members


def verify_ranking(frame, groups, table, summaries, pool=None):
    import torch
    from reward_model.model.losses import ligunity_listwise_loss
    descriptors, members = native_lists(frame) if pool is None else pool
    expected = descriptors.merge(groups[['group_id', 'size', 'size_bin', 'similarity_class', 'unit_id']],
                                  on='group_id', validate='many_to_one')
    pd.testing.assert_frame_equal(table[expected.columns].reset_index(drop=True), expected, check_exact=True)
    if not np.isfinite(table.loss).all() or (table.loss < 0).any():
        raise ValueError('Invalid listwise losses')
    strata = stratum_members(groups)
    if set(summaries) != set(strata):
        raise ValueError('Ranking strata differ from the frozen design')
    for name, identities in strata.items():
        rows = table[table.group_id.isin(identities)]
        stats = summaries[name]
        if len(rows) and rows.groupby('partition').size().tolist() != [len(rows)//3]*3:
            raise ValueError('Native partitions have different stratum sizes')
        counts = dict(list_instances=len(rows), lists_per_partition=len(rows)//3,
                      groups=int(rows.group_id.nunique()), units=int(rows.unit_id.nunique()))
        if any(stats[key] != value for key, value in counts.items()):
            raise ValueError('Ranking stratum coverage differs')
        close(stats['loss'], float(rows.loss.to_numpy().mean()) if len(rows) else None)
    sample = set(table.groupby(['partition', 'size_bin'], sort=True).head(1).index)
    zeros = table.index[table.loss == 0]
    if len(zeros):
        sample.add(int(zeros[0]))
    errors = []
    for index in sorted(sample):
        selected = frame.iloc[members[index]]
        actual = float(ligunity_listwise_loss(
            torch.tensor(selected.ranking_score.to_numpy(copy=True), dtype=torch.float32),
            torch.tensor(selected.pchembl_value.to_numpy(copy=True), dtype=torch.float32),
            torch.zeros(len(selected), dtype=torch.long), affinity_margin=.47712125472, min_pchembl_span=0.))
        error = abs(actual-float(table.loss.iloc[index])); errors.append(error)
        close(actual, float(table.loss.iloc[index]), atol=2e-6)
    return dict(list_descriptors_checked=len(table), native_list_checks=len(sample),
                maximum_sampled_list_difference=max(errors, default=0.), strata_checked=len(strata))


def verify(root, a_trajectory, scalar, ranking):
    root, a_trajectory, scalar, ranking = map(Path, (root, a_trajectory, scalar, ranking))
    if root != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use CPU-only verification against the authorized Atum project')
    sys.path.insert(0, str(root/'RewardModel'))
    import torch
    torch.set_num_threads(4)
    from experiments.paired_generalization.native_analysis import native_frame
    from experiments.paired_generalization.realization_protocol import align_rows
    from experiments.paired_generalization.similarity_composition import verify_design
    work = root/'RewardModel/outputs/paired_generalization'
    for path in (a_trajectory, scalar, ranking):
        if not path.resolve().is_relative_to((work/'tier1').resolve()):
            raise ValueError('Verification inputs must belong to Tier 1')
    audits = {}
    for kind, directory in [('scalar', scalar), ('ranking', ranking)]:
        if (directory/'verification.json').exists():
            raise RuntimeError('Existing verification requires inspection before reuse')
        record = read(directory/'complete.json')
        plan = read(checked(directory/'plan.json', record['plan_sha256']))
        if plan['sources_sha256'].get(str(a_trajectory)) != sha256(a_trajectory):
            raise ValueError('Localization did not use this exact A trajectory')
        for path, digest in plan['sources_sha256'].items():
            checked(path, digest)
        for path, digest in plan['code_sha256'].items():
            checked(root/path, digest)
        for name, digest in record['artifacts_sha256'].items():
            if Path(name).name != name:
                raise ValueError('Audit table escapes its directory')
            checked(directory/name, digest)
        audits[kind] = record
    trajectory = read(a_trajectory)
    native_verification = read(a_trajectory.with_name(a_trajectory.name.replace('trajectory_', 'verification_')))
    if (trajectory['epochs_analyzed'] != list(range(1,16)) or not trajectory['complete_horizon_observed']
            or native_verification['trajectory_sha256'] != sha256(a_trajectory)
            or not native_verification['source_and_artifact_hashes_verified']):
        raise ValueError('Expected a fully verified 15-epoch A trajectory')
    plan = read(checked(a_trajectory.parent/'analysis_plan.json', trajectory['analysis_plan_sha256']))
    for path, digest in plan['software']['source_sha256'].items():
        checked(root/path, digest)
    reference = read(checked(a_trajectory.parent/'references/complete.json', plan['reference_sha256']))['references']['eval']
    expected = pd.read_parquet(checked(a_trajectory.parent/'references/eval.parquet', reference['sha256']))
    design = work/'tier1/similarity_composition/design'
    verify_design(design)
    results = {'scalar':[], 'ranking':[]}
    for regime in ('A', 'B'):
        groups = pd.read_parquet(design/f'groups_{regime}_val.parquet')
        rows = pd.read_parquet(design/f'rows_{regime}_val.parquet')
        for record in audits.values():
            if record['results'][regime]['epochs'] != list(range(1,16)):
                raise ValueError('Localization omits a reference epoch')
        pool, first_identity = None, None
        for epoch in range(1,16):
            if regime == 'A':
                source = trajectory['trajectory'][epoch-1]['eval']['source']
                manifest = checked(source['manifest'], source['manifest_sha256'])
                frame, info = native_frame(manifest, expected, reference['fingerprint'], 'eval')
                checked(manifest.parent/info['predictions_file'], source['payload_sha256'])
            else:
                dest = work/f'tier1/predictions/epoch{epoch:02d}_val'
                metrics = read(work/f'tier1/metrics_float32/epoch{epoch:02d}_val.json')
                frame = pd.read_parquet(checked(dest/'predictions.parquet', metrics['predictions_sha256']))
            frame['pchembl_value'] = frame.pchembl_value.to_numpy(np.float32).astype(np.float64)
            align_rows(rows, frame)
            identity = frame[['group_id', 'compound_id', 'pchembl_value']]
            if pool is None:
                pool, first_identity = native_lists(frame), identity.copy()
            else:
                pd.testing.assert_frame_equal(identity, first_identity, check_exact=True)
            scalar_table = pd.read_parquet(scalar/f'{regime}_epoch{epoch:02d}_groups.parquet')
            ranking_table = pd.read_parquet(ranking/f'{regime}_epoch{epoch:02d}_lists.parquet')
            for kind, result in [('scalar', verify_scalar(frame, groups, scalar_table,
                    audits['scalar']['results'][regime]['trajectory'][epoch-1]['strata'])),
                ('ranking', verify_ranking(frame, groups, ranking_table,
                    audits['ranking']['results'][regime]['trajectory'][epoch-1]['strata'], pool))]:
                results[kind].append(dict(regime=regime, epoch=epoch, **result))
            print(json.dumps(dict(regime=regime, epoch=epoch, scalar=results['scalar'][-1],
                                  ranking=results['ranking'][-1])), flush=True)
    for kind, directory in [('scalar', scalar), ('ranking', ranking)]:
        from importlib.metadata import version
        result = dict(complete_sha256=sha256(directory/'complete.json'), verifier_sha256=sha256(__file__),
            packages={name:version(name) for name in ('numpy', 'pandas', 'scipy', 'torch')},
            native_A_verification_sha256=sha256(a_trajectory.with_name(a_trajectory.name.replace('trajectory_', 'verification_'))),
            checks=results[kind], source_and_output_hashes_verified=True,
            scope='All scalar group correlations, BCE sums, fixed stratum counts/means and rank-sum AUC; all native list descriptors and stratum reductions, plus production loss on sampled lists. Bootstrap intervals and endpoint-selection uncertainty are not independently reconstructed.',
            model_fitting=False, model_inference=False, test_predictions_read=False, verified_unix=time.time())
        target = directory/'verification.json'
        with target.open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False);stream.write('\n')
    return {kind:len(checks) for kind,checks in results.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--a-trajectory', type=Path, required=True)
    parser.add_argument('--scalar', type=Path, required=True)
    parser.add_argument('--ranking', type=Path, required=True)
    args = parser.parse_args()
    print(verify(args.root, args.a_trajectory, args.scalar, args.ranking))
