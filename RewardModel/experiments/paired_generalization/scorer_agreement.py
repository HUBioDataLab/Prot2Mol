"""Q31: within-assay scorer agreement across the saved baseline trajectory."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

if __package__:
    from .analyze_predictions import unit_bootstrap
    from .run import WORK, ROOT, sha256, write_json
else:
    from analyze_predictions import unit_bootstrap
    from run import WORK, ROOT, sha256, write_json

SCORERS = ('ranking_score', 'activity_logits', 'cosine_similarity', 'pchembl_value')
PAIRS = {f'{left}__{right}': (i, j) for (i, left), (j, right) in itertools.combinations(enumerate(SCORERS), 2)}


def group_agreement(frame, component_map):
    if not np.isfinite(frame[list(SCORERS)].to_numpy()).all():
        raise ValueError('Scorer agreement requires finite aligned predictions')
    records = []
    for group_id, group in frame.groupby('group_id', sort=True):
        ranks = rankdata(group[list(SCORERS)].to_numpy(), axis=0, method='average')
        centered = ranks - ranks.mean(axis=0)
        norms = np.sqrt(np.square(centered).sum(axis=0))
        denominator = np.outer(norms, norms)
        correlations = np.divide(centered.T @ centered, denominator,
                                 out=np.full((4, 4), np.nan), where=denominator > 0)
        correlations = np.clip(correlations, -1., 1.)
        records.append(dict(group_id=str(group_id), size=len(group),
            component=component_map[str(group_id).split('__', 1)[1]],
            original_eligible=len(group) >= 3 and np.ptp(group.pchembl_value.to_numpy()) >= .5,
            **{name: correlations[i, j] for name, (i, j) in PAIRS.items()}))
    return pd.DataFrame(records)


def summarize(table):
    result = {}
    for cohort, selected in (('all_size_at_least_two', table[table['size'] >= 2]),
                             ('original_ranking_eligible', table[table.original_eligible])):
        result[cohort] = {}
        for name in PAIRS:
            values = selected[name].dropna()
            result[cohort][name] = dict(
                groups=len(selected), defined_groups=len(values),
                defined_fraction=len(values)/len(selected) if len(selected) else None,
                macro_spearman=float(values.mean()) if len(values) else None)
    return result


def matched_endpoint_changes(first, last):
    columns = ['group_id', 'component', 'size', 'original_eligible']
    if not first[columns].equals(last[columns]):
        raise ValueError('Trajectory groups, components or eligibility changed')
    changes = {}
    for name in PAIRS:
        mask = first.original_eligible & first[name].notna() & last[name].notna()
        delta = last[name] - first[name]
        bootstrap = first[['component']].copy()
        bootstrap['spearman'] = delta.where(mask)
        interval = unit_bootstrap(bootstrap)
        interval['interpretation'] = (
            'Percentile interval over observed scaffold units, conditional on this fitted trajectory; '
            'not training-seed or split-realization uncertainty. Training-unit resampling is '
            'descriptive and does not estimate held-out generalization.')
        changes[name] = dict(
            matched_groups=int(mask.sum()), eligible_groups=int(first.original_eligible.sum()),
            first_macro=float(first.loc[mask, name].mean()) if mask.any() else None,
            last_macro=float(last.loc[mask, name].mean()) if mask.any() else None,
            mean_change=float(delta[mask].mean()) if mask.any() else None,
            unit_bootstrap=interval)
    return changes


def main():
    base = WORK/'tier1'
    destination = base/'scorer_agreement'
    destination.mkdir(parents=True, exist_ok=True)
    units = pd.read_parquet(base/'data/assay_units.parquet')
    if units.assay_id.duplicated().any():
        raise ValueError('Assay unit mapping is not unique')
    component_map = units.set_index('assay_id').component.to_dict()
    trajectory, identities, endpoints = [], {}, {}
    exports = sorted((base/'predictions').glob('*/complete.json'))
    if len(exports) != 31:
        raise ValueError('Expected all fifteen validation/training exports and selected test')
    for marker in exports:
        info = json.loads(marker.read_text())
        name, split = info['name'], info['split']
        source = marker.parent/'predictions.parquet'
        frame = pd.read_parquet(source)
        if len(frame) != info['rows'] or not np.array_equal(frame.row_index, np.arange(len(frame))):
            raise ValueError('Incomplete or unordered prediction export')
        ids = frame[['group_id', 'target_chembl_id', 'compound_id', 'pchembl_value', 'binary_label']]
        if split in identities and not identities[split].equals(ids):
            raise ValueError('Source rows/labels changed between epochs')
        identities.setdefault(split, ids)
        table = group_agreement(frame, component_map)
        table.to_parquet(destination/f'{name}.parquet', index=False)
        result = dict(name=name, split=split, checkpoint_sha256=info['checkpoint_sha256'],
                      source_sha256=sha256(source), summary=summarize(table))
        write_json(destination/f'{name}.json', result)
        trajectory.append(result)
        if name.startswith(('epoch01_', 'epoch15_')):
            endpoints[name] = table
        print(f'Scorer agreement complete: {name}', flush=True)
    write_json(destination/'complete.json', dict(
        source='fresh B baseline; seed42; one scaffold realization',
        source_code_sha256=sha256(ROOT/'RewardModel/experiments/paired_generalization/scorer_agreement.py'),
        exports=len(exports), trajectory=trajectory,
        endpoint_changes={split: matched_endpoint_changes(endpoints[f'epoch01_{suffix}'], endpoints[f'epoch15_{suffix}'])
                          for split, suffix in [('val', 'val'), ('train', 'train_probe')]},
        interpretation='Within-assay average-rank correlation, not pooled correlation. Endpoint changes use the same mutually defined eligible assays. Unit intervals condition on this fitted trajectory, not seed or split uncertainty. Agreement changes do not establish gradient conflict.',
        remaining='Matched A trajectory and controlled model comparisons'))


if __name__ == '__main__':
    main()
