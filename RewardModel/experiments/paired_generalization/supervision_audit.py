"""Answer Q14/Q21/Q23 using current partition-local labels, without training."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .audit_data import group_descriptors
else:
    from run import ROOT, WORK, sha256, write_json
    from audit_data import group_descriptors


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(count=0,mean=None,quantiles=None)
    return dict(count=len(values),mean=float(values.mean()),
        quantiles=dict(zip(('min','p25','median','p75','p90','p99','max'),
                           np.quantile(values,[0,.25,.5,.75,.9,.99,1]).tolist())))


def composition(frame, noise_rms, unit_ids):
    if not np.isfinite(noise_rms) or noise_rms <= 0 or len(unit_ids) != len(frame) or pd.isna(unit_ids).any():
        raise ValueError('Finite training noise and complete row-level units are required')
    if not np.isfinite(frame.pchembl_value).all() or not frame.binary_label.isin([0,1]).all():
        raise ValueError('Supervision labels must be finite and binary')
    groups = group_descriptors(frame)
    eligible = groups.ranking_eligible
    profiles = groups.groupby('target_chembl_id').agg(groups=('group_id','size'),eligible=('ranking_eligible','sum'))
    profiles['eligible_fraction'] = profiles.eligible/profiles.groups
    profiles['ineligible'] = profiles.groups-profiles.eligible
    p = profiles.eligible/profiles.eligible.sum() if profiles.eligible.sum() else None
    q = profiles.ineligible/profiles.ineligible.sum() if profiles.ineligible.sum() else None
    def subset_summary(selected):
        same_side = (selected.positive_rate == 0) | (selected.positive_rate == 1)
        return dict(groups=len(selected),rows=int(selected['size'].sum()),
            size=distribution(selected['size']),variance=distribution(selected.variance),
            span=distribution(selected.span),targets=int(selected.target_chembl_id.nunique()),
            all_positive_groups=int((selected.positive_rate == 1).sum()),
            all_negative_groups=int((selected.positive_rate == 0).sum()),
            same_side_fraction=float(same_side.mean()) if len(selected) else None)
    pairs = frame.groupby(['target_chembl_id','compound_id'],sort=False).pchembl_value
    medians = pairs.transform('median')
    if not np.array_equal((medians >= 6.).to_numpy(),frame.binary_label.to_numpy(dtype=bool)):
        raise ValueError('Audit must use labels harmonized only within the current partition')
    unique_medians = pairs.median()
    units = pd.DataFrame(dict(unit=np.asarray(unit_ids),assay_id=frame.assay_id.to_numpy(),
                              target=frame.target_chembl_id.to_numpy())).groupby('unit',sort=True).agg(
        rows=('assay_id','size'),assays=('assay_id','nunique'),targets=('target','nunique')).reset_index()
    result = dict(rows=len(frame), groups=len(groups),
        eligibility=dict(eligible_groups=int(eligible.sum()),eligible_fraction=float(eligible.mean()),
            too_small_groups=int((groups['size'] < 3).sum()),
            enough_items_but_insufficient_span_groups=int(((groups['size'] >= 3) & ~eligible).sum()),
            criterion='n >= 3 and float32 pChEMBL range >= 0.5; same criterion in training and headline evaluation'),
        all_groups=subset_summary(groups),eligible=subset_summary(groups[eligible]),ineligible=subset_summary(groups[~eligible]),
        targets=dict(total=len(profiles),with_no_eligible_groups=int((profiles.eligible == 0).sum()),
            with_only_eligible_groups=int((profiles.ineligible == 0).sum()),
            with_both=int(((profiles.eligible > 0) & (profiles.ineligible > 0)).sum()),
            eligibility_fraction=distribution(profiles.eligible_fraction),
            eligible_vs_ineligible_group_target_total_variation=float(.5*np.abs(p-q).sum()) if p is not None and q is not None else None),
        threshold=dict(value=6.,training_replicate_rms=float(noise_rms),noise_fold_scale=float(10**noise_rms),
            observation_fraction_within_rms=float((np.abs(frame.pchembl_value.to_numpy(dtype=np.float32)-6.) <= noise_rms).mean()),
            row_weighted_pair_median_fraction_within_rms=float((np.abs(medians.to_numpy()-6.) <= noise_rms).mean()),
            unique_pair_median_fraction_within_rms=float((np.abs(unique_medians.to_numpy()-6.) <= noise_rms).mean()),
            interpretation='Descriptive threshold proximity using training raw-replicate RMS; not a posterior label uncertainty or confidence interval'),
        units=dict(row_count=distribution(units.rows),largest_rows=int(units.rows.max()),
            largest_fraction=float(units.rows.max()/len(frame)),assays_per_unit=distribution(units.assays)))
    return result,groups,profiles.reset_index(),units


def main():
    if ROOT != Path('/home/atabey/Prot2Mol'):
        raise RuntimeError('Audit is restricted to Atum Prot2Mol')
    primary_path = WORK/'controlled/plan.json'
    plan = json.loads(primary_path.read_text())
    noise_path = WORK/'tier1/replicates/noise_summary.json'
    noise = json.loads(noise_path.read_text())
    unit_path = WORK/'tier1/data/assay_units.parquet'
    split_audit = json.loads((WORK/'tier1/split_realizations/complete.json').read_text())
    if sha256(unit_path) != split_audit['unit_map_sha256']:
        raise ValueError('Scaffold unit map changed after assignment audit')
    unit_map = pd.read_parquet(unit_path).set_index('assay_id').component
    destination = WORK/'tier1/supervision_composition'
    destination.mkdir(parents=True,exist_ok=True)
    results,artifacts = {},{}
    for regime in ('A','B'):
        results[regime] = {}
        for role in ('train','val','test'):
            source = Path(plan['bases'][regime]['data'][f'{role}_parquet_path'])
            if sha256(source) != plan['source_data_sha256'][regime][role]:
                raise ValueError('Prepared partition changed')
            frame = pd.read_parquet(source,columns=['assay_id','target_chembl_id','compound_id',
                'pchembl_value','binary_label','measurement_count'])
            units = frame.assay_id if regime == 'A' else frame.assay_id.map(unit_map)
            summary,groups,targets,unit_table = composition(frame,noise[regime]['train']['noise_scale']['rms_replicate_difference'],units)
            results[regime][role] = summary
            for suffix,table in [('groups',groups),('targets',targets),('units',unit_table)]:
                path = destination/f'{regime}_{role}_{suffix}.parquet'
                table.to_parquet(path,index=False)
                artifacts[path.name] = sha256(path)
            print(json.dumps(dict(regime=regime,partition=role,eligibility=summary['eligibility'],
                all_positive=summary['all_groups']['all_positive_groups'],all_negative=summary['all_groups']['all_negative_groups'])),flush=True)
    record = dict(format_version=1,questions=['Q14','Q21','Q23'],results=results,
        primary_plan_sha256=sha256(primary_path),data_sha256=plan['source_data_sha256'],
        unit_map_sha256=sha256(unit_path),noise_summary_sha256=sha256(noise_path),artifacts_sha256=artifacts,
        code_sha256=sha256(Path(__file__)),
        interpretation='Descriptive complete-partition audit with current partition-local labels. No model outputs, causal attribution, selection test or training performed.')
    write_json(destination/'complete.json',record)


if __name__ == '__main__':
    main()
