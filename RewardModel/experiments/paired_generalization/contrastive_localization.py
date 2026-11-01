"""Q06/Q10/Q11: attribute fixed-pool contrastive loss to original assay strata.

The negative candidates stay intact. This is attribution of the original
objective by query origin, not evaluation against a stratum-only candidate set.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

if __package__:
    from .run import ROOT, WORK, sha256
    from .control_plan import digest_json
    from .contrastive_audit import candidate_pool
    from .label_precision import trainer_label_frame
    from .realization_protocol import align_rows
    from .similarity_composition import verify_design, immutable
    from .stratified_trajectory import masks
else:
    from run import ROOT, WORK, sha256
    from control_plan import digest_json
    from contrastive_audit import candidate_pool
    from label_precision import trainer_label_frame
    from realization_protocol import align_rows
    from similarity_composition import verify_design, immutable
    from stratified_trajectory import masks

from reward_model.model.losses import ligunity_bidirectional_contrastive_loss, CONTRASTIVE_MASK_VALUE

DIRECTIONS = ('protein_to_molecule', 'molecule_to_protein', 'total')


def read(path):
    return json.loads(Path(path).read_text())


def check_policy(config):
    if (config.get('pair_scoring_mode') != 'mlp_contrastive' or
            config.get('contrastive_active_threshold') != 6. or
            config.get('contrastive_strict_active_only') is not True or
            config.get('contrastive_mask_policy', 'reference') != 'reference'):
        raise ValueError('Only the fixed reference contrastive policy is supported')


def batch_contributions(scores, batch):
    """Weighted positive-query sums for each list, before division by list count."""
    scores = scores.float()
    owners, target_ids, molecule_ids, labels = batch.owners, batch.target_ids, batch.molecule_ids, batch.labels
    count, ligands = scores.shape
    if (len(target_ids) != count or len(owners) != ligands or len(labels) != ligands or
            len(molecule_ids) != ligands or not torch.isfinite(scores).all() or
            not torch.isfinite(labels).all() or
            not torch.equal(torch.unique(owners, sorted=True), torch.arange(count))):
        raise ValueError('Invalid reference candidate batch')
    own = F.one_hot(owners, num_classes=count).T.bool()
    same_target = target_ids[:, None] == target_ids[owners][None, :]
    same_molecule = molecule_ids[:, None] == molecule_ids[None, :]
    duplicate = (own.float() @ same_molecule.float()) > 0
    masked = scores.masked_fill((~own) & (same_target | duplicate), CONTRASTIVE_MASK_VALUE)
    same_owner = owners[:, None] == owners[None, :]
    same_owner.fill_diagonal_(False)
    pm_logits = masked[owners].masked_fill(same_owner, CONTRASTIVE_MASK_VALUE)
    pm = F.cross_entropy(pm_logits, torch.arange(ligands), reduction='none')
    mp = F.cross_entropy(masked.T, owners, reduction='none')
    sizes = torch.bincount(owners, minlength=count).float()
    weight = sizes.rsqrt()[owners] * (labels > 6.).float()
    result = {}
    for name, values in [('protein_to_molecule', pm), ('molecule_to_protein', mp)]:
        result[name] = torch.zeros(count).scatter_add_(0, owners, values*weight)
    result['total'] = result['protein_to_molecule']+result['molecule_to_protein']
    return result


def localize(frame, protein, molecule, pools, scales, descriptors, *, check_every=100):
    if (protein.ndim != 2 or protein.shape != molecule.shape or len(protein) != len(frame) or
            not torch.isfinite(protein).all() or not torch.isfinite(molecule).all()):
        raise ValueError('Saved embeddings must be aligned and finite')
    if any(not np.isfinite(scale) or scale <= 0 for scale in scales.values()):
        raise ValueError('Expected positive finite saved scales')
    records = {name: [] for name in scales}; checked = 0; max_drift = 0.
    for part, batches in enumerate(pools):
        index = 0
        for batch_index, batch in enumerate(batches):
            cosines = protein[batch.representatives] @ molecule[batch.members].T
            sizes = torch.bincount(batch.owners, minlength=len(batch.representatives))
            active_counts = torch.zeros(len(sizes), dtype=torch.long).scatter_add_(0, batch.owners, (batch.labels > 6.).long())
            identities = frame.iloc[batch.representatives.numpy()].group_id.tolist()
            for scale_name, scale in scales.items():
                contributions = batch_contributions(cosines*scale, batch)
                if batch_index % check_every == 0 or batch_index == len(batches)-1:
                    native = ligunity_bidirectional_contrastive_loss(cosines*scale, batch.labels,
                        batch.owners, batch.target_ids, batch.molecule_ids, active_threshold=6., strict_active_only=True)
                    for name, expected in zip(('total', 'protein_to_molecule', 'molecule_to_protein'), native):
                        difference = abs(float(contributions[name].mean())-float(expected))
                        if difference > 1e-5:
                            raise ValueError('List attribution differs from native contrastive loss')
                        max_drift = max(max_drift, difference)
                    checked += 1
                for offset, identity in enumerate(identities):
                    records[scale_name].append(dict(partition=part, list_id=index+offset,
                        batch_index=batch_index, group_id=identity, list_size=int(sizes[offset]),
                        active_queries=int(active_counts[offset]),
                        **{name: float(values[offset]) for name, values in contributions.items()}))
            index += len(identities)
    summaries, tables = {}, {}
    for scale_name, records_for_scale in records.items():
        table = pd.DataFrame(records_for_scale).merge(
            descriptors[['group_id', 'size', 'size_bin', 'similarity_class', 'unit_id']],
            on='group_id', validate='many_to_one')
        if len(table) != len(records_for_scale):
            raise ValueError('Candidate lists lack fixed assay descriptors')
        tables[scale_name] = table; summaries[scale_name] = {}
        for key, mask in masks(table).items():
            subset = table[mask]
            counts = subset.groupby('partition').size()
            if len(subset) and (len(counts) != 3 or counts.nunique() != 1):
                raise ValueError('Stratum list counts differ across partitions')
            summaries[scale_name][key] = dict(groups=int(subset.group_id.nunique()),
                lists_per_partition=len(subset)//3, list_instances=len(subset),
                active_query_instances=int(subset.active_queries.sum()),
                **{name: float(subset.groupby('partition')[name].mean().mean()) if len(subset) else None for name in DIRECTIONS})
    return summaries, tables, dict(native_batches_checked=checked, max_absolute_native_batch_difference=max_drift)


def trajectory_summary(records):
    epochs = [r['epoch'] for r in records]
    if epochs != list(range(1, 16)):
        raise ValueError('Reference attribution needs all fifteen saved epochs')
    if not all(np.isfinite(r['native_macro_spearman']) for r in records):
        raise ValueError('Original native validation macro scores are required')
    macro_peak = max(records, key=lambda r: r['native_macro_spearman'])['epoch']
    extrema, changes = {}, {}
    for scale in records[0]['strata']:
        extrema[scale], changes[scale] = {}, {}
        for direction in DIRECTIONS:
            minimum = min(records, key=lambda r: r['strata'][scale]['all'][direction])['epoch']
            extrema[scale][direction] = minimum
            starts = sorted({1, macro_peak, minimum})
            changes[scale][direction] = {f'{start}_to_15': {
                key: records[-1]['strata'][scale][key][direction]-records[start-1]['strata'][scale][key][direction]
                if records[-1]['strata'][scale][key][direction] is not None else None
                for key in records[-1]['strata'][scale]} for start in starts}
    return dict(epochs=epochs, complete_horizon=True, native_macro_peak_epoch=macro_peak, directional_minimum_epochs=extrema,
        endpoint_point_changes=changes, trajectory=records,
        interpretation='Loss attribution by original query assay, holding every negative candidate fixed. Native 1/sqrt(list size) query weights and denominator mass from zero-active lists are preserved. No stratum-only candidate evaluation. Endpoint changes are descriptive; no seed uncertainty or confidence intervals are claimed. Candidate contexts couple units, so a simple query-unit bootstrap would not measure redrawn-pool uncertainty. Fixed-scale rescoring does not establish a benefit of fixed-temperature training.')


def main(regime, a_reference=None):
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use CPU-only analysis in the authorized Atum project')
    if (regime == 'A') != (a_reference is not None) or regime not in ('A', 'B'):
        raise ValueError('A requires its completed reference embedding audit; B uses the original baseline')
    torch.set_num_threads(4)
    base = WORK/'tier1'; design = base/'similarity_composition/design'; verify_design(design)
    sources = {}

    def checked(path, expected=None):
        path = Path(path); digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Changed source: {path}')
        sources[str(path)] = digest
        return path

    checked(design/'complete.json')
    candidates = []
    if regime == 'B':
        reference = read(checked(base/'baseline_trajectory_summary_v2.json'))
        for name, digest in reference['sources_sha256'].items():
            checked(WORK/name, digest)
        for row in reference['rows']:
            epoch = row['epoch']; name = f'epoch{epoch:02d}_val'; folder = base/'predictions'/name
            marker = read(checked(folder/'complete.json'))
            metric = read(checked(base/'metrics_float32'/f'{name}.json'))
            direction = read(checked(base/'contrastive'/f'{name}.json'))
            if (marker['split'] != 'val' or marker['checkpoint_sha256'] != metric['checkpoint_sha256'] or
                    marker['checkpoint_sha256'] != direction['checkpoint_sha256'] or marker['score_scale'] != direction['score_scale']):
                raise ValueError('B reference export provenance differs')
            check_policy(read(checked(Path(marker['checkpoint'])/'config.json')))
            checked(folder/'predictions.parquet', metric['predictions_sha256'])
            candidates.append(dict(epoch=epoch, folder=str(folder), rows=marker['rows'], scale=marker['score_scale'],
                pool_sha256=direction['candidate_pool_sha256'], expected_losses=direction['losses'],
                native_total=row['native_validation']['eval_contrastive_loss'],
                native_macro_spearman=row['native_validation']['eval_macro_spearman']))
    else:
        if __package__:
            from .reference_contrastive import verify_export
        else:
            from reference_contrastive import verify_export
        reference = read(checked(a_reference)); parent = Path(a_reference).parent
        plan = read(checked(parent/'plan.json', reference['plan_sha256']))
        if (reference['exports'] != 30 or reference['test_data_used'] or
                plan['job']['id'] != 'full__A__seed42__lr1p0'):
            raise ValueError('Expected the completed protected A reference embedding audit')
        for item in reference['trajectory']:
            if item['analysis']['prefix'] != 'eval':
                continue
            folder = Path(item['path']); checked(folder/'complete.json', item['sha256'])
            marker = read(folder/'complete.json'); verify_export(folder, marker['source'])
            if marker['source']['prefix'] != 'eval' or marker['source']['plan_sha256'] != reference['plan_sha256']:
                raise ValueError('A export does not belong to this validation audit')
            info = marker['analysis']; check_policy(read(checked(Path(marker['source']['checkpoint']['path'])/'config.json')))
            candidates.append(dict(epoch=info['epoch'], folder=str(folder), rows=info['rows'], scale=info['score_scale'],
                pool_sha256=info['candidate_pool_sha256'], expected_losses=info['losses'], native_total=info['native_contrastive_loss'],
                native_macro_spearman=info['native_macro_spearman']))
            for name, digest in marker['artifacts_sha256'].items():
                checked(folder/name, digest)
    candidates.sort(key=lambda r: r['epoch'])
    if [r['epoch'] for r in candidates] != list(range(1, 16)):
        raise ValueError('Missing or duplicate reference epochs')
    for item in candidates:
        # Native Trainer manifests serialize integral epochs as floats. The
        # exact 1..15 check above rejects fractional or missing epochs first.
        item['epoch'] = int(item['epoch'])
        for name in ['predictions.parquet', 'protein.npy', 'molecule.npy']:
            checked(Path(item['folder'])/name)
    from importlib.metadata import version
    paths = [Path(__file__), *(Path(__file__).with_name(name) for name in
        ['contrastive_audit.py', 'stratified_trajectory.py', 'label_precision.py', 'realization_protocol.py',
         'similarity_composition.py', 'reference_contrastive.py', 'run.py', 'control_plan.py']),
        ROOT/'RewardModel/reward_model/model/losses.py', ROOT/'RewardModel/reward_model/training/evaluation.py']
    plan = dict(regime=regime, candidates=candidates, sources_sha256=sources,
        code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in paths},
        packages={name: version(name) for name in ['torch', 'numpy', 'pandas']})
    output = base/'contrastive_localization'/regime/digest_json(plan)[:16]
    if output.exists():
        raise RuntimeError('Existing audit requires inspection; never overwrite evidence')
    output.mkdir(parents=True); immutable(output/'plan.json', plan)
    rows = pd.read_parquet(design/f'rows_{regime}_val.parquet')
    groups = pd.read_parquet(design/f'groups_{regime}_val.parquet')
    pools, identity, records = None, None, []
    for item in candidates:
        folder = Path(item['folder']); frame = trainer_label_frame(pd.read_parquet(folder/'predictions.parquet'))
        if len(frame) != item['rows'] or not np.array_equal(frame.row_index, np.arange(len(frame))):
            raise ValueError('Reference prediction coverage differs')
        align_rows(rows, frame)  # Keep native row order for candidate construction.
        columns = ['group_id', 'target_chembl_id', 'compound_id', 'pchembl_value', 'binary_label']
        if pools is None:
            pools, pool_hash = candidate_pool(frame)
            identity = frame[columns].copy()
        pd.testing.assert_frame_equal(frame[columns], identity, check_exact=True)
        if item['pool_sha256'] != pool_hash:
            raise ValueError('Fixed candidate pool differs from the original direction audit')
        protein = torch.from_numpy(np.load(folder/'protein.npy', allow_pickle=False))
        molecule = torch.from_numpy(np.load(folder/'molecule.npy', allow_pickle=False))
        stats, tables, checks = localize(frame, protein, molecule, pools,
            dict(learned_scale=item['scale'], fixed_initial_scale13=13.), groups)
        for scale, table in tables.items():
            for direction in DIRECTIONS:
                np.testing.assert_allclose(stats[scale]['all'][direction], item['expected_losses'][scale][direction], atol=1e-5, rtol=0)
            table.to_parquet(output/f'epoch{item["epoch"]:02d}_{scale}_lists.parquet', index=False)
        if abs(stats['learned_scale']['all']['total']-item['native_total']) > .01:
            raise ValueError('Contrastive attribution differs materially from original native evaluation')
        records.append(dict(epoch=item['epoch'], native_total=item['native_total'], scale=item['scale'],
            native_macro_spearman=item['native_macro_spearman'],
            candidate_pool_sha256=pool_hash, strata=stats, direct_native_batch_checks=checks))
        print(json.dumps(dict(epoch=item['epoch'], learned=stats['learned_scale']['all'], checks=checks)), flush=True)
    immutable(output/'complete.json', dict(plan_sha256=sha256(output/'plan.json'), result=trajectory_summary(records),
        artifacts_sha256={p.name: sha256(p) for p in output.glob('*.parquet')}, test_predictions_read=False,
        model_inference=False, model_fitting=False))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regime', choices=['A', 'B'], required=True)
    parser.add_argument('--a-reference', type=Path)
    args = parser.parse_args()
    print(main(args.regime, args.a_reference), flush=True)
