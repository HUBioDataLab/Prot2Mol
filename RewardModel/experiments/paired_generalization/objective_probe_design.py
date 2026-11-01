"""Freeze held-out cold validation units for E56/E57; no diagnostic inference.

The global B validation table is not automatically held out from A training.
Each model regime therefore uses its own validation units with no scaffold
relative in that regime's training partition. Units and assays remain intact.
"""
from __future__ import annotations

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

IDENTITY = ['assay_id','target_chembl_id','compound_id']


def unit_ids(frame, regime, components):
    if regime == 'A':
        return 'assay:'+frame.assay_id.astype(str)
    if regime != 'B':
        raise ValueError('Unknown probe regime')
    units = frame.assay_id.map(components)
    if units.isna().any():
        raise ValueError('Missing scaffold component')
    return 'component:'+units.astype(np.int64).astype(str)


def cold_units(training, validation, scaffolds, components, regime):
    for frame in (training, validation):
        if frame.empty or frame[IDENTITY].isna().any().any() or frame.duplicated(IDENTITY).any():
            raise ValueError('Expected unique complete source observations')
        if frame.groupby('assay_id').target_chembl_id.nunique().max() != 1:
            raise ValueError('A source assay mixes targets')
    train_units, val_units = unit_ids(training, regime, components), unit_ids(validation, regime, components)
    if set(train_units) & set(val_units) or set(training.assay_id) & set(validation.assay_id):
        raise ValueError('Training and validation units/assays overlap')
    train_scaffolds, val_scaffolds = training.compound_id.map(scaffolds), validation.compound_id.map(scaffolds)
    if train_scaffolds.isna().any() or val_scaffolds.isna().any():
        raise ValueError('Missing molecule scaffold relation')
    rows = validation[IDENTITY].copy().reset_index(drop=True)
    rows['source_row_index'] = np.arange(len(rows))
    rows['unit'] = val_units.to_numpy()
    rows['group_id'] = rows.target_chembl_id.astype(str)+'__'+rows.assay_id.astype(str)
    rows['scaffold_seen_in_training'] = val_scaffolds.isin(set(train_scaffolds)).to_numpy()
    units = rows.groupby('unit',sort=True).agg(rows=('assay_id','size'),assays=('assay_id','nunique'),
        any_training_scaffold=('scaffold_seen_in_training','max')).reset_index()
    selected = units.loc[~units.any_training_scaffold, 'unit']
    chosen = rows[rows.unit.isin(selected)].copy()
    if chosen.scaffold_seen_in_training.any():
        raise ValueError('Selected cold unit contains a training scaffold')
    values = validation.iloc[chosen.source_row_index].pchembl_value.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite selected continuous labels')
    # Labels describe feasibility only; they do not select or order units.
    groups = chosen.assign(value=values.astype(np.float64)).groupby('group_id').value.agg(['size','min','max'])
    eligible = (groups['size'] >= 3) & (groups['max']-groups['min'] >= .5)
    summary = dict(regime=regime, source_validation_rows=len(validation), source_validation_units=len(units),
        selected_rows=len(chosen), selected_assays=chosen.assay_id.nunique(), selected_units=len(selected),
        selected_molecules=chosen.compound_id.nunique(), ranking_eligible_assays=int(eligible.sum()),
        strict_active_rows=int((values > np.float32(6.)).sum()),
        selected_source_row_fraction=len(chosen)/len(validation),
        selection='Own held-out validation units whose every molecule lacks an exact scaffold match in own training; no label/prediction-dependent selection.')
    return chosen.reset_index(drop=True), units, summary


def main():
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use CPU-only preparation in authorized Atum Prot2Mol')
    output = WORK/'tier4/objective_probe_design'
    if output.exists():
        raise RuntimeError('Existing objective-probe design requires inspection')
    primary_path=WORK/'controlled/plan.json'; primary=json.loads(primary_path.read_text())
    relation_path=WORK/'tier1/similarity_composition/design/complete.json'
    relation=json.loads(relation_path.read_text())
    if sha256(ROOT/'Paired input ranking generalization.md') != primary['spec_sha256']:
        raise ValueError('Governing MD changed')
    if relation['source_sha256'][str(primary_path)] != sha256(primary_path):
        raise ValueError('Similarity relation belongs to a different split plan')
    sources={str(primary_path):sha256(primary_path),str(relation_path):sha256(relation_path)}
    frames={}
    for regime in ('A','B'):
        for split in ('train','val'):
            path=Path(primary['bases'][regime]['data'][split+'_parquet_path'])
            sources[str(path)]=sha256(path)
            if sources[str(path)] != primary['source_data_sha256'][regime][split]:
                raise ValueError('Frozen partition changed')
            frames[(regime,split)]=pd.read_parquet(path,columns=IDENTITY+['pchembl_value'])
    scaffold_path=WORK/'tier1/data/molecule_scaffolds.parquet';unit_path=WORK/'tier1/data/assay_units.parquet'
    for path in (scaffold_path,unit_path):
        sources[str(path)]=sha256(path)
        if sources[str(path)] != relation['source_sha256'][str(path)]:
            raise ValueError('Frozen similarity/unit relation changed')
    scaffolds=pd.read_parquet(scaffold_path).set_index('compound_id').scaffold
    components=pd.read_parquet(unit_path).set_index('assay_id').component
    if not scaffolds.index.is_unique or not components.index.is_unique:
        raise ValueError('Ambiguous relation identity')
    results={regime:cold_units(frames[(regime,'train')],frames[(regime,'val')],scaffolds,components,regime)
             for regime in ('A','B')}
    cross=frames[('B','val')][IDENTITY].merge(frames[('A','train')][IDENTITY],on=IDENTITY,how='inner',validate='one_to_one')
    output.parent.mkdir(parents=True,exist_ok=True);archive_directory(output)
    artifacts={}
    for regime,(rows,units,_) in results.items():
        for name,table in [(f'validation_rows_{regime}.parquet',rows),(f'validation_units_{regime}.parquet',units)]:
            table.to_parquet(output/name,index=False);artifacts[name]=sha256(output/name)
    write_json(output/'complete.json',dict(format_version=1,experiment_ids=['E56','E57'],
        spec_sha256=primary['spec_sha256'],source_sha256=sources,artifacts_sha256=artifacts,
        code_sha256={name:sha256(Path(__file__).with_name(name)) for name in ('objective_probe_design.py','objective_probes.py')},
        regimes={regime:value[2] for regime,value in results.items()},
        cross_split_warning=dict(B_validation_observations_in_A_training=len(cross),B_validation_rows=len(frames[('B','val')]),
            interpretation='B validation is not assumed held out from A training. Own-regime cold validation preserves the intended held-out meaning; A cold units are a selected subset, not a full-A metric.'),
        remaining='Freeze training/validation batch identities, candidate pools and step sizes; qualify source/token/checkpoint provenance and permitted-GPU numerical resolution after Tier3. This does not measure any gradient cosine or lookahead transfer.',
        labels_used_only_for_feasibility=True,test_data_read=False,model_inference=False,model_fitting=False))
    return output


if __name__ == '__main__':
    print(main())
