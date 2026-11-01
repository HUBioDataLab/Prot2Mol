"""Q16: B test target support and fixed-checkpoint ranking by training coverage."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .analyze_predictions import unit_bootstrap
else:
    from run import ROOT, WORK, sha256, write_json
    from analyze_predictions import unit_bootstrap

# Fixed support bins, before reading prediction outcomes. Zero is retained so
# absence of cold targets is explicit rather than silently excluding them.
BOUNDS = [('0',0,0), ('1',1,1), ('2-5',2,5), ('6-20',6,20), ('21-100',21,100), ('>100',101,None)]


def stratify(groups, training_counts):
    groups = groups.copy()
    groups['target_chembl_id'] = groups.group_id.str.split('__', n=1).str[0]
    groups['training_assays'] = groups.target_chembl_id.map(training_counts).fillna(0).astype(int)
    groups['support_bin'] = ''
    for name, low, high in BOUNDS:
        groups.loc[(groups.training_assays >= low) & ((groups.training_assays <= high) if high is not None else True), 'support_bin'] = name
    if (groups.support_bin == '').any():
        raise ValueError('Unclassified training-support count')
    summaries = []
    for name, _, _ in BOUNDS:
        subset = groups[groups.support_bin == name].copy()
        subset.loc[~subset.original_ranking_eligible,'spearman'] = np.nan
        eligible = subset[subset.original_ranking_eligible]
        defined = eligible.spearman.notna()
        summaries.append(dict(support_bin=name, targets=int(subset.target_chembl_id.nunique()),
            assays=len(subset), observations=int(subset['size'].sum()), eligible_assays=len(eligible),
            defined_assays=int(defined.sum()), defined_fraction=float(defined.mean()) if len(eligible) else None,
            macro_spearman=float(eligible.spearman.mean()) if defined.any() else None,
            unit_bootstrap=unit_bootstrap(subset)))
    return groups, summaries


def main():
    if ROOT != Path('/home/atabey/Prot2Mol'):
        raise RuntimeError('Use authorized Atum project')
    base = WORK/'tier1'
    parity_path = base/'label_precision_complete.json'
    parity = json.loads(parity_path.read_text())
    if len(parity['native_parity_checks']) != 93:
        raise ValueError('Corrected metric parity must be complete')
    plan_path = WORK/'controlled/plan.json'
    plan = json.loads(plan_path.read_text())
    paths = {role:Path(plan['bases']['B']['data'][f'{role}_parquet_path']) for role in ('train','test')}
    for role,path in paths.items():
        if sha256(path) != plan['source_data_sha256']['B'][role]:
            raise ValueError('Frozen B source changed')
    frames = {role:pd.read_parquet(path,columns=['target_chembl_id','assay_id']) for role,path in paths.items()}
    counts = frames['train'].drop_duplicates().groupby('target_chembl_id').size()
    metric_path = base/'metrics_float32/groups_best_test_ranking_score.parquet'
    groups = pd.read_parquet(metric_path)
    test_groups = frames['test'].target_chembl_id+'__'+frames['test'].assay_id
    if set(groups.group_id) != set(test_groups) or groups.group_id.duplicated().any():
        raise ValueError('Prediction groups do not cover frozen B test assays')
    groups, summaries = stratify(groups,counts)
    recomposed = sum(row['macro_spearman']*row['defined_assays'] for row in summaries if row['defined_assays']) / sum(row['defined_assays'] for row in summaries)
    if not np.isclose(recomposed,parity['selected_test_macro'],rtol=0,atol=1e-14):
        raise ValueError('Coverage strata do not reconstruct the selected headline metric')
    targets = groups[['target_chembl_id','training_assays','support_bin']].drop_duplicates().sort_values('target_chembl_id')
    destination = base/'target_coverage'
    destination.mkdir(parents=True,exist_ok=True)
    for name,table in [('test_groups',groups),('test_targets',targets)]:
        table.to_parquet(destination/f'{name}.parquet',index=False)
    result = dict(question='Q16',regime='B',test_targets=len(targets),
        unseen_training_targets=int((targets.training_assays==0).sum()),
        training_assays_per_test_target=targets.training_assays.describe(percentiles=[.25,.5,.75,.9,.99]).to_dict(),
        training_assays_per_test_assay=groups.training_assays.describe(percentiles=[.25,.5,.75,.9,.99]).to_dict(),
        strata=summaries, recomposed_macro_spearman=recomposed,
        source_data_sha256={role:sha256(path) for role,path in paths.items()},
        parity_sha256=sha256(parity_path),predicted_groups_sha256=sha256(metric_path),
        artifacts_sha256={name:sha256(destination/name) for name in ['test_groups.parquet','test_targets.parquet']},
        code_sha256=sha256(Path(__file__)),
        scope='Original B seed42 macro-selected checkpoint only; bins fixed from training-assay counts, not outcomes. Intervals resample whole scaffold-assay components conditional on this model/allocation. Strata also differ in assay composition; this is not a causal effect of protein coverage.')
    write_json(destination/'complete.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
