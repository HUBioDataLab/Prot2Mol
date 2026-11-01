"""E40 fixed-test, exactly matched leak-in/out data construction, before model fitting.

A keeps its original test assays. B first reserves one label-blind donor assay
from every multi-assay test component, fixing the remaining test groups before
matching. Only matched donors enter leak-in; unused donors remain unused.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
else:
    from run import ROOT, WORK, sha256, write_json

from reward_model.training.train_views import subset_pair_labels

SEED = 42
COLS = ['assay_id','target_chembl_id','compound_id','pchembl_value','binary_label','measurement_count']


def priority(assay):
    return hashlib.sha256(f'{SEED}:{assay}'.encode()).hexdigest()


def relabel(frame):
    frame = frame.reset_index(drop=True).copy()
    frame['binary_label'] = subset_pair_labels(frame,np.arange(len(frame)))
    return frame


def reserve_b_donors(test):
    """Use identities/components only; never read labels to choose the fixed test set."""
    assays = test[['assay_id','component']].drop_duplicates()
    if assays.assay_id.duplicated().any():
        raise ValueError('An assay belongs to multiple components')
    donors = []
    for _,group in assays.groupby('component',sort=True):
        if len(group) >= 2:
            donors.append(min(group.assay_id,key=priority))
    donor_rows = test.assay_id.isin(donors)
    return test.loc[donor_rows].copy(),test.loc[~donor_rows].copy()


def pair_independent_assays(frame):
    """Label-blind greedy whole-assay set with no shared target/molecule pairs."""
    frame = frame.reset_index(drop=True)
    if frame.duplicated(['assay_id','target_chembl_id','compound_id']).any():
        raise ValueError('Expected one curated observation per assay/pair')
    codes = pd.factorize(pd.MultiIndex.from_frame(frame[['target_chembl_id','compound_id']]),sort=False)[0]
    indices = frame.groupby('assay_id',sort=False).indices
    seen,selected = set(),[]
    for assay in sorted(indices,key=priority):
        current = set(codes[indices[assay]].tolist())
        if seen.isdisjoint(current):
            selected.append(assay)
            seen.update(current)
    return frame[frame.assay_id.isin(selected)].copy()


def match_features(frame):
    records = []
    for assay,group in frame.groupby('assay_id',sort=True):
        if group.target_chembl_id.nunique() != 1:
            raise ValueError('An assay must have one protein')
        values = group.pchembl_value.to_numpy(dtype=np.float64)
        replicas = group.measurement_count.to_numpy(dtype=np.int64)
        if np.any(replicas < 1) or not np.array_equal(replicas,group.measurement_count):
            raise ValueError('Replicate counts must be positive integers')
        profile = sorted(Counter(replicas.tolist()).items())
        descriptor = [str(group.target_chembl_id.iloc[0]),len(group),int(group.binary_label.sum()),profile,
                      bool(len(group)>=3 and np.ptp(values)>=.5),
                      bool(len(group)>=3 and np.ptp(values.astype(np.float32).astype(np.float64))>=.5)]
        records.append(dict(assay_id=assay,key=json.dumps(descriptor,separators=(',',':')),
                            rows=len(group),target_chembl_id=descriptor[0]))
    return pd.DataFrame(records,columns=['assay_id','key','rows','target_chembl_id'])


def match_assays(replacements, donors):
    out,inside = match_features(replacements),match_features(donors)
    matches = []
    donor_bins = {key:sorted(group.assay_id,key=priority) for key,group in inside.groupby('key')}
    for key,group in out.groupby('key',sort=True):
        for removed,added in zip(sorted(group.assay_id,key=priority),donor_bins.get(key,[])):
            matches.append(dict(removed_assay=removed,donor_assay=added,matching_key=key))
    return pd.DataFrame(matches,columns=['removed_assay','donor_assay','matching_key'])


def composition(frame):
    sizes = frame.groupby('assay_id').size()
    groups = frame.groupby('assay_id').pchembl_value.agg(['size','min','max'])
    return dict(rows=len(frame),assays=len(sizes),targets=int(frame.target_chembl_id.nunique()),
        positive_rows=int(frame.binary_label.sum()),positive_rate=float(frame.binary_label.mean()),
        replicate_histogram={str(k):int(v) for k,v in frame.measurement_count.value_counts().sort_index().items()},
        assay_size_histogram={str(k):int(v) for k,v in sizes.value_counts().sort_index().items()},
        sampler_eligible_groups=int(((groups['size']>=3)&((groups['max']-groups['min'])>=.5)).sum()),
        pchembl_mean=float(frame.pchembl_value.mean()),pchembl_std=float(frame.pchembl_value.std()))


def construct(frames, regime):
    train,val,original_test = (frames[role].copy() for role in ('train','val','test'))
    for left,right in [('train','val'),('train','test'),('val','test')]:
        if set(frames[left].assay_id)&set(frames[right].assay_id):
            raise ValueError('Original assay partitions overlap')
    original_relatives = int(pd.concat([train,val]).scaffold.isin(original_test.scaffold.unique()).sum())
    if regime == 'A':
        related = train.scaffold.isin(original_test.scaffold.unique())
        any_related = train.assay_id[related].unique()
        out = relabel(train[~train.assay_id.isin(any_related)])
        # Fully related donors cannot share a molecule with the leak-out pool.
        fully_related = related.groupby(train.assay_id).all()
        donors = pair_independent_assays(train[train.assay_id.isin(fully_related[fully_related].index)])
        test = original_test.copy()
        # Removing these assays cannot change a common row's pair-median label.
        repeated = out.duplicated(['target_chembl_id','compound_id'],keep=False)
        unsafe = out.assay_id[repeated].unique()
        replacements = out[~out.assay_id.isin(unsafe)].copy()
        reservation = dict(policy='Original A test groups unchanged',reserved_donor_assays=int(donors.assay_id.nunique()))
    elif regime == 'B':
        donors,test = reserve_b_donors(original_test)
        out = relabel(train)
        unit_assays = train[['component','assay_id']].drop_duplicates().groupby('component').size()
        replacements = out[out.component.isin(unit_assays[unit_assays==1].index)].copy()
        reservation = dict(policy='One hash-selected donor per original multi-assay B test component; unused donors never returned to test',
            reserved_donor_assays=int(donors.assay_id.nunique()),reserved_donor_rows=len(donors),
            original_test_assays=int(original_test.assay_id.nunique()),fixed_test_assays=int(test.assay_id.nunique()))
    else:
        raise ValueError('Unknown split')
    donors = relabel(donors)
    test = relabel(test)
    matches = match_assays(replacements,donors)
    removed = out[out.assay_id.isin(matches.removed_assay)].copy()
    common = out[~out.assay_id.isin(matches.removed_assay)].copy()
    added = donors[donors.assay_id.isin(matches.donor_assay)].copy()
    inside = relabel(pd.concat([common,added],ignore_index=True))
    if not np.array_equal(inside.binary_label.iloc[:len(common)],common.binary_label):
        raise ValueError('Exchange changed common training pair-median labels')
    if sorted(match_features(removed).key) != sorted(match_features(inside.iloc[len(common):]).key):
        raise ValueError('Final subset-local labels or matching descriptors differ')
    summaries = {key:composition(frame) for key,frame in [('leak_out',out),('leak_in',inside)]}
    for key in ['rows','assays','targets','positive_rows','replicate_histogram','assay_size_histogram','sampler_eligible_groups']:
        if summaries['leak_out'][key] != summaries['leak_in'][key]:
            raise ValueError(f'Training composition changed: {key}')
    for label,frame in [('out',out),('in',inside)]:
        if set(frame.assay_id)&set(test.assay_id) or set(frame.assay_id)&set(val.assay_id):
            raise ValueError('A held-out assay entered training')
    if out.scaffold.isin(test.scaffold.unique()).any():
        raise ValueError('Leak-out retains a fixed-test scaffold relative')
    out_profile = out.groupby('target_chembl_id').agg(rows=('assay_id','size'),assays=('assay_id','nunique'),positives=('binary_label','sum'))
    in_profile = inside.groupby('target_chembl_id').agg(rows=('assay_id','size'),assays=('assay_id','nunique'),positives=('binary_label','sum'))
    pd.testing.assert_frame_equal(out_profile,in_profile)
    exposure = test[['assay_id','target_chembl_id','compound_id','scaffold','component']].copy()
    exposure['relative_in_leak_in'] = exposure.scaffold.isin(inside.scaffold.unique())
    exposure['molecule_in_leak_in'] = exposure.compound_id.isin(inside.compound_id.unique())
    exposure['training_assays_for_target'] = exposure.target_chembl_id.map(out_profile.assays).fillna(0).astype(int)
    result = dict(regime=regime,seed=SEED,original_full_test_relative_rows_outside_test=original_relatives,
        fixed_test_rows=len(test),fixed_test_assays=int(test.assay_id.nunique()),reservation=reservation,
        replacement_candidate_assays=int(replacements.assay_id.nunique()),donor_candidate_assays=int(donors.assay_id.nunique()),
        matched_assay_exchanges=len(matches),exchanged_rows=len(added),training=summaries,
        common_binary_labels_unchanged=True,per_protein_row_assay_positive_counts_identical=True,
        exposed_test_rows=int(exposure.relative_in_leak_in.sum()),
        exposed_test_assays=int(exposure.groupby('assay_id').relative_in_leak_in.any().sum()),
        exposed_test_molecule_rows=int(exposure.molecule_in_leak_in.sum()),
        cold_protein_test_rows=int((exposure.training_assays_for_target==0).sum()),
        fixed_test_binary_labels_changed=int((test.binary_label.to_numpy()!=original_test.set_index('source_row_index').loc[test.source_row_index,'binary_label'].to_numpy()).sum()),
        usable_leak_contrast=bool(len(matches) and exposure.relative_in_leak_in.any()),
        interpretation='Data construction only, no model outcomes. Identical test groups in all four cells; shared validation. Exactly matched protein/size/positive/replicate/eligibility descriptors. Whole A assays and whole original B training components retained/removed. B leak-in intentionally places a reserved whole donor assay from a test component in training; component isolation is the manipulated variable. Test exposure may be sparse and limits power. Unused donor assays are excluded in both arms.')
    return result,dict(leak_out=out,leak_in=inside,test=test,donors=donors,matches=matches,exposure=exposure)


def main():
    if __package__:
        from .prepare_random import archive_directory
    else:
        from prepare_random import archive_directory
    if ROOT != Path('/home/atabey/Prot2Mol'):
        raise RuntimeError('Use authorized Atum project')
    directory = WORK/'leak_design'
    if directory.exists():
        raise RuntimeError('Existing construction must be inspected before replacing it')
    archive_directory(directory)
    plan_path = WORK/'controlled/plan.json'
    plan = json.loads(plan_path.read_text())
    map_path = WORK/'tier1/data/molecule_scaffolds.parquet'
    unit_path = WORK/'tier1/data/assay_units.parquet'
    scaffolds = pd.read_parquet(map_path).set_index('compound_id').scaffold
    units = pd.read_parquet(unit_path).set_index('assay_id').component
    if scaffolds.index.has_duplicates or units.index.has_duplicates:
        raise ValueError('Ambiguous molecular or unit mapping')
    results,artifacts = {},{}
    for regime in ('A','B'):
        frames = {}
        for role in ('train','val','test'):
            path = Path(plan['bases'][regime]['data'][f'{role}_parquet_path'])
            if sha256(path) != plan['source_data_sha256'][regime][role]:
                raise ValueError('Frozen source data changed')
            frame = pd.read_parquet(path,columns=COLS).reset_index(drop=True)
            frame['source_partition'],frame['source_row_index'] = role,np.arange(len(frame))
            frame['scaffold'] = frame.compound_id.map(scaffolds)
            frame['component'] = frame.assay_id.map(units)
            if frame[['scaffold','component']].isna().any().any():
                raise ValueError('Source observation lacks its split relation/unit')
            frames[role] = frame
        print(f'Constructing matched E40 data: {regime}',flush=True)
        results[regime],tables = construct(frames,regime)
        for name,table in tables.items():
            path = directory/f'{regime}_{name}.parquet'
            table.to_parquet(path,index=False)
            artifacts[path.name] = sha256(path)
        print(json.dumps(results[regime]),flush=True)
    write_json(directory/'complete.json',dict(format_version=1,experiment='E40',results=results,
        primary_plan_sha256=sha256(plan_path),source_data_sha256=plan['source_data_sha256'],
        molecule_scaffolds_sha256=sha256(map_path),assay_units_sha256=sha256(unit_path),
        artifacts_sha256=artifacts,code_sha256=sha256(Path(__file__)),
        remaining='Review realized exposure and matching before freezing model configs; tokenize/view qualification, all four-cell multi-seed fitting and performance comparisons remain pending'))


if __name__ == '__main__':
    main()
