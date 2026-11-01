"""Q11: localize the reference's actual deterministic listwise validation loss.

Uses saved ranking scores, preserves native row order and complete-coverage
partitions, and checks every epoch against the native loss. No model fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__:
    from .run import ROOT, WORK, sha256
    from .control_plan import digest_json
    from .label_precision import trainer_label_frame
    from .native_analysis import native_frame
    from .realization_protocol import align_rows
    from .similarity_composition import verify_design, immutable
    from .stratified_trajectory import masks
else:
    from run import ROOT, WORK, sha256
    from control_plan import digest_json
    from label_precision import trainer_label_frame
    from native_analysis import native_frame
    from realization_protocol import align_rows
    from similarity_composition import verify_design, immutable
    from stratified_trajectory import masks

from reward_model.training.evaluation import build_complete_coverage_ranking_partitions

POLICY = dict(ranking_temperature=1., ranking_affinity_margin=0.47712125472,
    ranking_min_pchembl_span=.5, ranking_objective='ligunity_listwise',
    ranking_normalize_per_group=False, ranking_balance_assays=False,
    ranking_position_weighting='top', ranking_max_ligands=16,
    evaluation_ranking_partitions=3, seed=42)


def read(path):
    return json.loads(Path(path).read_text())


def check_model_policy(config):
    defaults = dict(ranking_objective='ligunity_listwise', ranking_normalize_per_group=False,
        ranking_balance_assays=False, ranking_position_weighting='top')
    for name in POLICY.keys()-{'ranking_max_ligands', 'evaluation_ranking_partitions', 'seed'}:
        if config.get(name, defaults.get(name)) != POLICY[name]:
            raise ValueError(f'Unsupported reference ranking policy: {name}')


def make_pool(frame):
    """No row sorting before native partitioning: row order affects the shuffle."""
    ids, names = pd.factorize(frame.group_id, sort=True)
    if 'evaluation_group_index' in frame and not np.array_equal(ids, frame.evaluation_group_index):
        raise ValueError('Native assay index differs from sorted local group identities')
    labels = torch.tensor(frame.pchembl_value.to_numpy(copy=True), dtype=torch.float32)
    groups = torch.tensor(ids, dtype=torch.long)
    partitions = build_complete_coverage_ranking_partitions(pchembl_values=labels,
        ranking_group_ids=groups, max_list_size=16, num_partitions=3, seed=42, min_pchembl_span=.5)
    layouts, records = [], []
    digest = hashlib.sha256()
    for array in (groups, labels):
        digest.update(array.numpy().tobytes())
    for part, values in enumerate(partitions):
        valid = np.flatnonzero(values.numpy() >= 0)
        order = valid[np.argsort(values.numpy()[valid], kind='stable')]
        counts = np.bincount(values.numpy()[valid]) if len(valid) else np.zeros(0, int)
        if len(counts) and (counts.min() < 3 or counts.max() > 16):
            raise ValueError('Native list size violates complete-coverage policy')
        offsets = np.r_[0, counts.cumsum()]
        padded = np.zeros((len(counts), 16), dtype=np.int64)
        for index, count in enumerate(counts):
            members = order[offsets[index]:offsets[index+1]]
            padded[index, :count] = members
            if len(np.unique(ids[members])) != 1:
                raise ValueError('Native ranking list crosses assays')
            records.append(dict(partition=part, list_id=index,
                group_id=names[ids[members[0]]], list_size=int(count)))
        layouts.append((padded, counts))
        digest.update(values.numpy().tobytes())
    return layouts, pd.DataFrame(records, columns=['partition', 'list_id', 'group_id', 'list_size']), digest.hexdigest()


def list_losses(scores, labels, indices, counts):
    """Vectorized reference formula, checked against the production loss in tests."""
    result = []
    scores = torch.tensor(np.array(scores, copy=True), dtype=torch.float32)
    labels = torch.tensor(np.array(labels, copy=True), dtype=torch.float32)
    if not torch.isfinite(scores).all() or not torch.isfinite(labels).all():
        raise ValueError('Nonfinite saved scores or labels')
    for start in range(0, len(counts), 2048):
        count = torch.tensor(counts[start:start+2048])
        members = torch.tensor(indices[start:start+2048])
        valid = torch.arange(16)[None, :] < count[:, None]
        target = labels[members].masked_fill(~valid, -torch.inf)
        order = torch.argsort(target, dim=1, descending=True, stable=True)
        target = torch.gather(target, 1, order)
        score = torch.gather(scores[members], 1, order)
        same = (target[:, :, None] == target[:, None, :]) & valid[:, :, None] & valid[:, None, :]
        base = 1./(count.float().sqrt()[:, None]*torch.log(torch.arange(2., 18.))[None, :])
        weight = (same*base[:, None, :]).sum(dim=2)/same.sum(dim=2).clamp_min(1)
        weaker = (target[:, None, :] < target[:, :, None]-POLICY['ranking_affinity_margin']) & valid[:, None, :]
        competitors = score[:, None, :].expand(-1, 16, -1).masked_fill(~weaker, -torch.inf)
        normalizer = torch.logsumexp(competitors, dim=2)
        losses = ((torch.logaddexp(score, normalizer)-score)*weight).masked_fill(~valid, 0.).sum(dim=1)
        result.extend(losses.tolist())
    return np.asarray(result, float)


def evaluate(frame, pool, design_groups):
    layouts, records, _ = pool
    losses = [list_losses(frame.ranking_score, frame.pchembl_value, indices, counts) for indices, counts in layouts]
    table = records.assign(loss=np.concatenate(losses)).merge(
        design_groups[['group_id', 'size', 'size_bin', 'similarity_class', 'unit_id']], on='group_id', validate='many_to_one')
    if len(table) != len(records):
        raise ValueError('Ranking lists lack fixed assay descriptors')
    summary = {}
    for key, selected in masks(table).items():
        subset = table[selected]
        per_partition = subset.groupby('partition').loss.agg(['mean', 'size'])
        if len(subset) and (len(per_partition) != 3 or per_partition['size'].nunique() != 1):
            raise ValueError('Stratum list counts differ across native partitions')
        summary[key] = dict(loss=float(per_partition['mean'].mean()) if len(subset) else None,
            list_instances=len(subset), lists_per_partition=len(subset)//3,
            groups=int(subset.group_id.nunique()), units=int(subset.unit_id.nunique()))
    if summary['all']['loss'] is None:
        raise ValueError('Reference validation has no ranking lists')
    return summary, table


def changes(first, last, all_units, repetitions=2000):
    keys = ['partition', 'list_id', 'group_id', 'size', 'size_bin', 'similarity_class', 'unit_id']
    pd.testing.assert_frame_equal(first[keys], last[keys], check_exact=True)
    units = pd.Index(np.unique(all_units))
    if not first.unit_id.isin(units).all():
        raise ValueError('Bootstrap units omit ranking lists')
    output = {}
    for key, mask in masks(first).items():
        chosen = first[mask].assign(delta=last.loss[mask].to_numpy()-first.loss[mask].to_numpy())
        by_unit = chosen.groupby('unit_id').delta.agg(['sum', 'size']).reindex(units, fill_value=0)
        values, counts = by_unit['sum'].to_numpy(), by_unit['size'].to_numpy()
        samples = []; rng = np.random.default_rng(42)
        for start in range(0, repetitions, 100):
            index = rng.integers(len(units), size=(min(100, repetitions-start), len(units)))
            numerator, denominator = values[index].sum(axis=1), counts[index].sum(axis=1)
            samples.extend((numerator[denominator>0]/denominator[denominator>0]).tolist())
        output[key] = dict(change=float(values.sum()/counts.sum()) if counts.sum() else None,
            interval=np.quantile(samples, [.025, .975]).tolist() if samples and len(units)>1 else None,
            all_partition_units=len(units), stratum_units=int(chosen.unit_id.nunique()),
            list_instances=len(chosen), defined_repetitions=len(samples), repetitions=repetitions, seed=42)
    return output


def main(a_path):
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use CPU-only analysis in the authorized Atum project')
    torch.set_num_threads(4)
    base = WORK/'tier1'; design = base/'similarity_composition/design'
    verify_design(design)
    sources = {}

    def checked(path, expected=None):
        path = Path(path); digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Artifact changed: {path}')
        sources[str(path)] = digest
        return path

    checked(design/'complete.json')
    a_path = Path(a_path)
    if a_path.parent.parent.name != 'full__A__seed42__lr1p0' or not a_path.name.startswith('trajectory_through_epoch'):
        raise ValueError('Use an immutable verified A reference prefix')
    a = read(checked(a_path)); verify = read(checked(a_path.with_name(a_path.name.replace('trajectory_', 'verification_'))))
    if verify['trajectory_sha256'] != sha256(a_path) or not verify['source_and_artifact_hashes_verified']:
        raise ValueError('A prefix verification differs')
    analysis_plan = read(checked(a_path.parent/'analysis_plan.json', a['analysis_plan_sha256']))
    refs = read(checked(a_path.parent/'references/complete.json', analysis_plan['reference_sha256']))
    expected = pd.read_parquet(checked(a_path.parent/'references/eval.parquet', refs['references']['eval']['sha256']))
    b = read(checked(base/'baseline_trajectory_summary_v2.json'))
    for name, digest in b['sources_sha256'].items():
        checked(WORK/name, digest)
    from reward_model.training.config import load_reward_training_config
    configs = [Path(a['job']['config_path']), ROOT/'RewardModel/configs/reward_train_paired_generalization_baseline.yaml']
    for path in configs:
        config = load_reward_training_config(checked(path)); check_model_policy(vars(config.model))
        if (config.data.ranking_max_ligands != 16 or config.data.evaluation_ranking_partitions != 3 or
                config.data.ranking_min_pchembl_span != .5 or config.training.seed != 42):
            raise ValueError('Reference list partition policy differs')
    candidates = {'A': [], 'B': []}
    for row in a['trajectory']:
        snapshot = row['eval']; manifest = checked(snapshot['source']['manifest'], snapshot['source']['manifest_sha256'])
        frame, native = native_frame(manifest, expected, refs['references']['eval']['fingerprint'], 'eval')
        checked(manifest.parent/native['predictions_file'], snapshot['source']['payload_sha256'])
        candidates['A'].append((int(row['epoch']), frame, native['metrics']['eval_ranking_loss']))
    for row in b['rows']:
        epoch = row['epoch']; name = f'epoch{epoch:02d}_val'
        metrics = read(checked(base/'metrics_float32'/f'{name}.json'))
        export = read(checked(base/'predictions'/name/'complete.json'))
        if export['split'] != 'val' or export['checkpoint_sha256'] != metrics['checkpoint_sha256']:
            raise ValueError('B validation export provenance differs')
        check_model_policy(read(checked(Path(export['checkpoint'])/'config.json')))
        frame = trainer_label_frame(pd.read_parquet(checked(base/'predictions'/name/'predictions.parquet', metrics['predictions_sha256'])))
        if len(frame) != export['rows'] or not np.array_equal(frame.row_index, np.arange(len(frame))):
            raise ValueError('B prediction coverage differs')
        candidates['B'].append((epoch, frame, row['native_validation']['eval_ranking_loss']))
    from importlib.metadata import version
    code_paths = [Path(__file__), Path(__file__).with_name('stratified_trajectory.py'),
        ROOT/'RewardModel/reward_model/training/evaluation.py', ROOT/'RewardModel/reward_model/model/losses.py']
    plan = dict(policy=POLICY, sources_sha256=sources,
        code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in code_paths},
        packages={name: version(name) for name in ['torch', 'numpy', 'pandas']})
    output = base/'ranking_localization'/digest_json(plan)[:16]
    if output.exists():
        raise RuntimeError('Existing audit requires inspection; never overwrite evidence')
    output.mkdir(parents=True); immutable(output/'plan.json', plan)
    results = {}
    for regime, snapshots in candidates.items():
        rows = pd.read_parquet(design/f'rows_{regime}_val.parquet')
        groups = pd.read_parquet(design/f'groups_{regime}_val.parquet')
        reference = snapshots[0][1]; pool = make_pool(reference)
        records, tables = [], {}
        for epoch, frame, native in snapshots:
            # Validate against the frozen partition, but do not reorder native rows.
            align_rows(rows, frame)
            pd.testing.assert_frame_equal(reference[['group_id', 'compound_id', 'pchembl_value']],
                frame[['group_id', 'compound_id', 'pchembl_value']], check_exact=True)
            stats, table = evaluate(frame, pool, groups)
            np.testing.assert_allclose(stats['all']['loss'], native, atol=1e-5, rtol=0)
            table.to_parquet(output/f'{regime}_epoch{epoch:02d}_lists.parquet', index=False)
            tables[epoch] = table; records.append(dict(epoch=epoch, native_ranking_loss=native,
                reconstruction_difference=stats['all']['loss']-native, strata=stats))
            print(json.dumps(dict(regime=regime, epoch=epoch, ranking_loss=stats['all']['loss'], native=native)), flush=True)
        epochs = [r['epoch'] for r in records]
        if epochs != list(range(1, len(records)+1)) or len(records)>15:
            raise ValueError('Expected a contiguous observed reference prefix')
        minimum = min(records, key=lambda r: r['native_ranking_loss'])['epoch']
        results[regime] = dict(epochs=epochs, complete_horizon=len(records)==15,
            native_loss_min_epoch=minimum, pool_sha256=pool[2], trajectory=records,
            endpoint_changes={f'{start}_to_{epochs[-1]}': changes(tables[start], tables[epochs[-1]], groups.unit_id)
                for start in sorted({1, minimum})})
    immutable(output/'complete.json', dict(plan_sha256=sha256(output/'plan.json'), results=results,
        artifacts_sha256={p.name: sha256(p) for p in output.glob('*.parquet')}, test_predictions_read=False,
        interpretation='Reference native list-weighted objective, not assay-macro ranking. Three fixed partitions; zero-loss lists retain denominator mass. Whole-unit paired intervals condition on one trajectory, fixed allocation and observed endpoint choices; no endpoint-selection/seed uncertainty or multiple-stratum adjustment. A prefix cannot establish full-training behavior. Contrastive localization remains separate.'))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a-trajectory', type=Path, required=True)
    print(main(parser.parse_args().a_trajectory), flush=True)
