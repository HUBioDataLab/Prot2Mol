"""Q30/E38: seven loss systems and auxiliary-loss contrasts with R present."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT,WORK,sha256
    from . import controlled_summary,control_plan,similarity_composition
else:
    from run import ROOT,WORK,sha256
    import controlled_summary,control_plan,similarity_composition

read=controlled_summary.read
immutable=controlled_summary.immutable
finite=controlled_summary.selected_analysis.finite_json
SUBSETS={s:('full' if s=='RBC' else 'loss_'+s) for s in ('R','B','C','RB','RC','BC','RBC')}
READOUTS={s:('ranking_head' if 'R' in s else 'activity_logit' if 'B' in s else 'cosine') for s in SUBSETS}
ANCHORED=('R','RB','RC','RBC')
CONTRASTS={
    'B_given_R':dict(RB=1,R=-1),
    'B_given_RC':dict(RBC=1,RC=-1),
    'C_given_R':dict(RC=1,R=-1),
    'C_given_RB':dict(RBC=1,RB=-1),
    'B_by_C_given_R':dict(RBC=1,RB=-1,RC=-1,R=1),
}


def software():
    result=controlled_summary.software()
    result['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha256(Path(__file__))
    return result


def validate_grid(primary):
    declared=[a['name'] for a in primary['arms'] if a['family']=='loss_subsets']
    if (len(declared)!=6 or set(declared)!=set(SUBSETS.values())-{'full'} or
        sum(a['name']=='full' for a in primary['arms'])!=1 or
        primary['shared_reference']['loss_RBC']!='full'):
        raise ValueError('Loss-subset registry changed')
    for subset,arm in SUBSETS.items():
        jobs=[j for j in primary['tuning_jobs'] if j['arm']==arm]
        expected={control_plan.job_id(arm,r,42,lr) for r in ('A','B') for lr in control_plan.LEARNING_RATE_MULTIPLIERS}
        if len(jobs)!=6 or {j['id'] for j in jobs}!=expected:raise ValueError('Incomplete loss-subset tuning grid')
        weights=tuple(w if s in subset else 0. for s,w in zip('RBC',(.25,.25,1.)))
        for job in jobs:
            if (job['seed']!=42 or job['phase']!='tuning' or
                job['id']!=control_plan.job_id(arm,job['regime'],job['seed'],job['learning_rate_multiplier'])):
                raise ValueError('Loss-subset job identity changed')
            model=job['config']['model']
            if (tuple(model[k] for k in ('ranking_loss_weight','classification_loss_weight','contrastive_loss_weight'))!=weights or
                model['ranking_readout']!=READOUTS[subset] or job['config']['training']['num_train_epochs']!=15):
                raise ValueError('Loss weights, readout or horizon differ from the declared grid')


def paired_effects(tables):
    """Five contrasts use one common eligible/defined cohort and paired unit draws."""
    if set(tables)!=set(ANCHORED):raise ValueError('Need exactly R, RB, RC and RBC')
    columns=['group_id','component','size','original_ranking_eligible']
    ordered={k:v.sort_values('group_id').reset_index(drop=True) for k,v in tables.items()}
    first=ordered['R']
    if first.empty or any(t.group_id.duplicated().any() or not t[columns].equals(first[columns]) for t in ordered.values()):
        raise ValueError('Loss comparison changed groups, units, size or eligibility')
    eligible=first.original_ranking_eligible.to_numpy(bool)
    scores={k:t.spearman.to_numpy(float) for k,t in ordered.items()}
    matched=eligible&np.logical_and.reduce([np.isfinite(scores[k]) for k in ANCHORED])
    pieces=[]
    for name,coefficients in CONTRASTS.items():
        delta=sum(weight*scores[subset] for subset,weight in coefficients.items())
        pieces.append(pd.DataFrame(dict(unit_id=first.component,spearman=np.where(matched,delta,np.nan),
            eligible=eligible,contrast=name)))
    stacked=pd.concat(pieces,ignore_index=True)
    weights={name:((stacked.contrast==name)&stacked.eligible).to_numpy(float) for name in CONTRASTS}
    boot=similarity_composition.bootstrap_scores(stacked,weights,first.component.unique())
    return finite(dict(groups=len(first),eligible_groups=int(eligible.sum()),mutually_defined_groups=int(matched.sum()),
        defined_groups={k:int((eligible&np.isfinite(v)).sum()) for k,v in scores.items()},
        common_cohort_means={k:float(v[matched].mean()) if matched.any() else None for k,v in scores.items()},
        original_cohort_means={k:float(v[eligible&np.isfinite(v)].mean()) if (eligible&np.isfinite(v)).any() else None for k,v in scores.items()},
        contrasts=boot,coefficients=CONTRASTS,
        interpretation='All contrasts use the same mutually defined eligible assays across R/RB/RC/RBC. Whole-unit draws include units with zero eligible scores and are paired across contrasts. Intervals condition on these fitted models and assignments. The B-by-C interaction conditions on R present and the ranking-head readout; it is not a three-factor interaction or a gradient diagnostic. Arm-specific B-selected LRs and checkpoint choices remain part of the compared training procedures.'))


def summarize_effects(effects):
    expected={(rule,split,regime,seed) for rule in controlled_summary.RULES for split in ('val','test') for regime in ('A','B') for seed in (42,43,44)}
    if set(effects)!=expected:raise ValueError('Need every rule, partition, regime and seed')
    result={}
    for rule in controlled_summary.RULES:
        result[rule]={}
        for split in ('val','test'):
            values={r:{str(s):{k:effects[rule,split,r,s]['contrasts'][k]['score'] for k in CONTRASTS}
                for s in (42,43,44)} for r in ('A','B')}
            result[rule][split]={r:controlled_summary.seed_spread(v) for r,v in values.items()}
            result[rule][split]['A_minus_B']=controlled_summary.seed_spread({str(s):controlled_summary.subtract(values['A'][str(s)],values['B'][str(s)]) for s in (42,43,44)})
    return result


def verify_summary(path):
    path=Path(path);result=read(path);arm=result['summary']['arm']
    if arm not in SUBSETS.values():raise ValueError('Unregistered loss arm')
    analyses=result['source']['analysis_sha256']
    for name,digest in analyses.items():
        if sha256(Path(name))!=digest:raise ValueError('Primary analysis changed')
    verified=controlled_summary.summarize(arm,sorted(analyses))
    if verified.resolve()!=path.resolve() or read(verified)!=result:
        raise ValueError('Loss comparison must use the verified canonical primary summary')
    records=[(Path(p),read(Path(p))) for p in analyses]
    return result,{record['job']['id']:(path,record) for path,record in records}


def analyze(paths):
    primary_path=WORK/'controlled/plan.json';primary=read(primary_path);validate_grid(primary)
    paths=[Path(p) for p in paths]
    if len(paths)!=7 or {read(p)['summary']['arm'] for p in paths}!=set(SUBSETS.values()):
        raise ValueError('Need exactly seven complete primary-arm summaries')
    summaries={};analyses={};qualification=None;protocols=set();runtime_evidence={}
    for path in paths:
        summary,records=verify_summary(path);arm=summary['summary']['arm']
        if summary['source']['primary_plan_sha256']!=sha256(primary_path) or summary['source']['software']!=controlled_summary.software():
            raise ValueError('Primary plan or reporting implementation differs across losses')
        subset=next(s for s,a in SUBSETS.items() if a==arm)
        if summary['summary']['ranking_readout']!=READOUTS[subset]:raise ValueError('Loss readout changed')
        summaries[subset]=summary;analyses[subset]=records
        for _,record in records.values():
            protocols.add(control_plan.digest_json({k:record['source'][k] for k in ('software','design_sha256','noise_summary_sha256')}))
            runtime_path=Path(record['job']['output_dir'])/'runtime_qualification.json';runtime=read(runtime_path)
            if runtime['world_size']!=4 or runtime['signature']!=control_plan.digest_json(dict(sources=runtime['sources'],plan_sha256=sha256(primary_path))):
                raise ValueError('Invalid training qualification')
            if qualification is None:
                qualification=runtime
                for name,digest in runtime['sources'].items():
                    if not (ROOT/name).resolve().is_relative_to(ROOT.resolve()) or sha256(ROOT/name)!=digest:
                        raise ValueError('Qualified training source changed')
            elif runtime!=qualification:raise ValueError('Training implementations differ across losses')
            runtime_evidence[str(runtime_path)]=sha256(runtime_path)
    if len(protocols)!=1:raise ValueError('Metric, composition or noise protocols differ across losses')
    source=dict(primary_plan_sha256=sha256(primary_path),summary_sha256={str(p):sha256(p) for p in paths},
        runtime_qualification_sha256=runtime_evidence,software=software())
    target=WORK/'loss_subset_analysis'/control_plan.digest_json(source)[:16]/'complete.json'
    if target.exists():
        cached=read(target);payload={k:v for k,v in cached.items() if k not in ('source','payload_sha256')}
        if cached['source']!=source or cached['payload_sha256']!=control_plan.digest_json(payload):raise ValueError('Completed loss report changed')
        for path,digest in payload['group_tables_sha256'].items():
            if sha256(Path(path))!=digest:raise ValueError('Compared group table changed')
        return target
    overview={s:dict(arm=SUBSETS[s],readout=READOUTS[s],learning_rate_multiplier=r['summary']['selected_learning_rate_multiplier'],
        results={rule:{split:r['summary']['selected'][rule][split]['ranking_score'] for split in ('val','test')} for rule in controlled_summary.RULES}) for s,r in summaries.items()}
    effects={};nested={};cache={};table_hashes={}
    for rule in controlled_summary.RULES:
        nested[rule]={}
        for split in ('val','test'):
            nested[rule][split]={}
            for regime in ('A','B'):
                nested[rule][split][regime]={}
                for seed in (42,43,44):
                    selected={};inputs={}
                    for subset in ANCHORED:
                        lr=overview[subset]['learning_rate_multiplier'];identity=control_plan.job_id(SUBSETS[subset],regime,seed,lr)
                        path,record=analyses[subset][identity];checkpoint=record['rules'][rule]['checkpoint']
                        table=path.parent/checkpoint/split/'groups_ranking_score.parquet'
                        selected[subset]=table;table_hashes[str(table)]=sha256(table)
                        inputs[subset]=dict(job_id=identity,checkpoint=checkpoint,learning_rate_multiplier=lr)
                    key=tuple(str(selected[s]) for s in ANCHORED)
                    if key not in cache:cache[key]=paired_effects({s:pd.read_parquet(p) for s,p in selected.items()})
                    value=dict(inputs=inputs,**cache[key]);effects[rule,split,regime,seed]=value
                    nested[rule][split][regime][str(seed)]=value
    payload=dict(scoring_systems=overview,conditional_auxiliary_effects=nested,seed_spread=summarize_effects(effects),
        group_tables_sha256=table_hashes,study_complete=False,new_fits_created=0,test_used_for_selection=False,
        accounting=dict(settings=7,tuning_fits_per_setting=6,confirmation_fits_per_setting=4,
            unique_tuning_fits=42,unique_confirmation_fits=28,unique_total_fits=70,
            shared_full_reference_fits=10,additional_loss_subset_fits=60),
        interpretation='Seven complete scoring systems use their declared readouts and equal per-setting LR budgets. Conditional auxiliary effects keep R and its ranking-head readout. The full three-factor interaction would require an empty-loss cell, which the MD does not request. Individual seed42 results have tuning exposure; seed43/44 confirmations and conditional unit intervals remain distinct. A-minus-B effects compare different held-out populations and have no cross-split confidence interval. No reported test result chooses a setting or budget; gradient conflict remains a separate question.')
    for path,digest in {str(primary_path):source['primary_plan_sha256'],**source['summary_sha256'],**runtime_evidence,**table_hashes}.items():
        if sha256(Path(path))!=digest:raise ValueError('Loss-comparison evidence changed during analysis')
    immutable(target,dict(source=source,payload_sha256=control_plan.digest_json(payload),**payload));return target


def preflight():
    primary=read(WORK/'controlled/plan.json');validate_grid(primary)
    return dict(arms=SUBSETS,readouts=READOUTS,summary_candidates={s:[str(p) for p in sorted((WORK/'controlled_summaries'/a).glob('*/complete.json'))] for s,a in SUBSETS.items()},
        new_fits_created=0,scientific_comparison_created=False,
        interpretation='Read-only inventory of the seven existing primary-arm reports. Paths alone are not verified scientific completion.')


if __name__=='__main__':
    if ROOT!=Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Use CPU-only Atum Prot2Mol')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('analyze','preflight'));parser.add_argument('--summaries',nargs='+',type=Path)
    args=parser.parse_args()
    print(json.dumps(preflight(),indent=2) if args.action=='preflight' else analyze(args.summaries))
