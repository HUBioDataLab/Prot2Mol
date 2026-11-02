"""E33: validation-selected decay and matched checkpoint-rule comparisons.

Use the three already registered settings. One B-validation-selected decay/LR
is shared across checkpoint rules; test scores never choose settings or rules.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path

import pandas as pd

if __package__:
    from .run import ROOT,WORK,sha256
    from . import controlled_summary,control_plan,postselection
else:
    from run import ROOT,WORK,sha256
    import controlled_summary,control_plan,postselection

read=controlled_summary.read
immutable=controlled_summary.immutable
ARMS={'full':.01,'weight_decay_0p001':.001,'weight_decay_0p1':.1}


def software():
    result=controlled_summary.software()
    for name in ('regularization_analysis.py','control_driver.py','control_plan.py'):
        path=Path(__file__).with_name(name)
        result['source_sha256'][str(path.relative_to(ROOT))]=sha256(path)
    return result


def family_jobs(primary):
    declared=[a for a in primary['arms'] if a['family']=='regularization']
    if (len(declared)!=2 or {a['name'] for a in declared}!=set(ARMS)-{'full'} or
        primary['shared_reference']['weight_decay_0p01']!='full' or primary['gpu_uuids']!=list(control_plan.GPU_UUIDS)):
        raise ValueError('Regularization family or devices changed')
    jobs=[j for j in primary['tuning_jobs'] if j['arm'] in ARMS]
    expected={control_plan.job_id(a,r,42,lr) for a in ARMS for r in ('A','B') for lr in control_plan.LEARNING_RATE_MULTIPLIERS}
    if len(jobs)!=18 or {j['id'] for j in jobs}!=expected:raise ValueError('Need exactly eighteen registered tuning fits')
    reference={(j['regime'],j['learning_rate_multiplier']):j for j in jobs if j['arm']=='full'}
    for job in jobs:
        config=job['config'];training=config['training']
        if (job['seed']!=42 or job['phase']!='tuning' or
            job['id']!=control_plan.job_id(job['arm'],job['regime'],42,job['learning_rate_multiplier']) or
            (training['num_train_epochs'],training['per_device_train_batch_size'],training['gradient_accumulation_steps'])!=(15,12,4) or
            config['model']['ranking_readout']!='ranking_head'):
            raise ValueError('Regularization readout, identity or budget changed')
        common=copy.deepcopy(reference[job['regime'],job['learning_rate_multiplier']]['config'])
        common['training'].update(weight_decay=ARMS[job['arm']],output_dir=training['output_dir'])
        if config!=common:raise ValueError('Decay arm changes more than weight decay')
    return jobs


def choose(decisions):
    """Inputs are native B-validation LR decisions, never selected test scores."""
    if len(decisions)!=3 or {d['arm'] for d in decisions}!=set(ARMS):raise ValueError('Need all three validation decisions')
    fields=('arm','learning_rate_multiplier','validation_macro_spearman','validation_defined_fraction')
    candidates=[{k:d[k] for k in fields} for d in decisions]
    for c in candidates:
        if (c['learning_rate_multiplier'] not in control_plan.LEARNING_RATE_MULTIPLIERS or
            c['validation_macro_spearman'] is None or not math.isfinite(c['validation_macro_spearman']) or
            c['validation_defined_fraction'] is None or not math.isfinite(c['validation_defined_fraction']) or
            not 0<c['validation_defined_fraction']<=1):raise ValueError('Each decay needs a finite defined validation winner')
    order=list(ARMS);best=max(candidates,key=lambda c:(c['validation_macro_spearman'],-order.index(c['arm'])))
    return dict(choice=dict(**best,weight_decay=ARMS[best['arm']]),
        within_setting={c['arm']:dict(**c,weight_decay=ARMS[c['arm']]) for c in candidates},
        family_tie_order=order,shared_across_checkpoint_rules=True,
        criterion='Maximum native B validation best-macro score at seed42 after each setting\'s existing three-LR selection; family ties prefer full, decay0.001, decay0.1',
        test_used_for_selection=False,new_fits_created=0,
        interpretation='One decay and each setting\'s B-selected LR are shared across checkpoint rules. These are checkpoint-rule contrasts on common training procedures, not independently tuned stopping algorithms. Defined coverage remains explicit; different defined cohorts are not silently equalized. The family has nine LR/decay candidates per split, not a globally optimal regularizer search.')


def build_selection():
    directory=WORK/'controlled';primary_path=directory/'plan.json';primary=read(primary_path)
    jobs=family_jobs(primary);decisions=[];evidence={};qualification=None
    for arm in ARMS:
        decision,path=postselection.verify_selection(directory,arm,primary);decisions.append(decision)
        evidence[str(path)]=sha256(path)
    for job in jobs:
        metadata=postselection.registered_job(directory,job['id'],primary)
        output=Path(metadata['output_dir']);runtime_path=output/'runtime_qualification.json';runtime=read(runtime_path)
        if runtime['world_size']!=4 or runtime['signature']!=control_plan.digest_json(dict(sources=runtime['sources'],plan_sha256=sha256(primary_path))):
            raise ValueError('Invalid training qualification')
        if qualification is None:
            qualification=runtime
            for name,digest in runtime['sources'].items():
                path=ROOT/name
                if not path.resolve().is_relative_to(ROOT.resolve()) or sha256(path)!=digest:raise ValueError('Qualified training source changed')
        elif qualification!=runtime:raise ValueError('Decay training qualifications differ')
        for path in (directory/'jobs'/f"{job['id']}.json",Path(metadata['config_path']),
            output/'controlled_training_complete.json',runtime_path):evidence[str(path)]=sha256(path)
    return dict(primary_plan_sha256=sha256(primary_path),software=software(),validation_evidence_sha256=evidence,
        training_qualification=qualification,**choose(decisions),
        accounting=dict(settings=3,unique_tuning_fits=18,existing_confirmation_fits=12,shared_full_reference_fits=10,total_primary_fits=30))


def selection_path():return WORK/'regularization_analysis/selection.json'


def prepare_selection():
    immutable(selection_path(),build_selection());return selection_path()


def verify_selection():
    result=read(selection_path())
    if result!=build_selection():raise ValueError('Frozen decay selection or validation evidence changed')
    return result


def verify_summary(path):
    path=Path(path);result=read(path);arm=result['summary']['arm']
    if arm not in ARMS:raise ValueError('Unregistered regularization summary')
    analyses=result['source']['analysis_sha256']
    for p,digest in analyses.items():
        if sha256(Path(p))!=digest:raise ValueError('Primary analysis changed')
    verified=controlled_summary.summarize(arm,sorted(analyses))
    if verified.resolve()!=path.resolve() or read(verified)!=result:raise ValueError('Use the verified canonical primary report')
    records=[(Path(p),read(Path(p))) for p in analyses]
    return result,{r['job']['id']:(p,r) for p,r in records}


def analyze(paths):
    selection=verify_selection();paths=[Path(p) for p in paths]
    if len(paths)!=3 or {read(p)['summary']['arm'] for p in paths}!=set(ARMS):raise ValueError('Need all three completed primary summaries')
    summaries={};analyses={};protocols=set();runtime_hashes={}
    for path in paths:
        result,records=verify_summary(path);summary=result['summary'];arm=summary['arm']
        if (result['source']['primary_plan_sha256']!=selection['primary_plan_sha256'] or
            result['source']['software']!=controlled_summary.software() or summary['ranking_readout']!='ranking_head' or
            summary['selected_learning_rate_multiplier']!=selection['within_setting'][arm]['learning_rate_multiplier']):
            raise ValueError('Selected LR, readout or summary implementation changed')
        summaries[arm]=result;analyses[arm]=records
        for _,record in records.values():
            protocols.add(control_plan.digest_json({k:record['source'][k] for k in ('software','design_sha256','noise_summary_sha256')}))
            runtime=Path(record['job']['output_dir'])/'runtime_qualification.json'
            if read(runtime)!=selection['training_qualification']:raise ValueError('Confirmation training qualification differs')
            runtime_hashes[str(runtime)]=sha256(runtime)
    if len(protocols)!=1:raise ValueError('Metrics, composition or noise protocols differ')
    source=dict(selection_sha256=sha256(selection_path()),summary_sha256={str(p):sha256(p) for p in paths},
        runtime_qualification_sha256=runtime_hashes,software=software())
    target=WORK/'regularization_analysis'/control_plan.digest_json(source)[:16]/'complete.json'
    if target.exists():
        result=read(target);payload={k:v for k,v in result.items() if k not in ('source','payload_sha256')}
        if result['source']!=source or result['payload_sha256']!=control_plan.digest_json(payload):raise ValueError('Completed regularization report changed')
        for path,digest in payload['group_tables_sha256'].items():
            if sha256(Path(path))!=digest:raise ValueError('Compared group table changed')
        return target
    settings={arm:dict(weight_decay=ARMS[arm],learning_rate_multiplier=r['summary']['selected_learning_rate_multiplier'],
        results={rule:{split:r['summary']['selected'][rule][split]['ranking_score'] for split in ('val','test')}
            for rule in controlled_summary.RULES},
        scalar_rule_changes={rule:{split:cells[split]['ranking_score'] for split in ('val','test')}
            for rule,cells in r['summary']['rule_minus_best_macro'].items()},
        paired_checkpoint_changes={job:{rule:{split:scores['ranking_score'] for split,scores in cells.items()}
            for rule,cells in comparisons.items()} for job,comparisons in r['paired_checkpoint_changes'].items()}) for arm,r in summaries.items()}
    comparisons={};spreads={};cache={};hashes={}
    for arm in list(ARMS)[1:]:
        comparisons[arm]={};spreads[arm]={}
        for rule in controlled_summary.RULES:
            comparisons[arm][rule]={};spreads[arm][rule]={}
            for split in ('val','test'):
                comparisons[arm][rule][split]={};values={}
                for regime in ('A','B'):
                    comparisons[arm][rule][split][regime]={};values[regime]={}
                    for seed in (42,43,44):
                        tables=[];inputs=[]
                        for setting in ('full',arm):
                            lr=settings[setting]['learning_rate_multiplier'];identity=control_plan.job_id(setting,regime,seed,lr)
                            path,record=analyses[setting][identity];checkpoint=record['rules'][rule]['checkpoint']
                            table=path.parent/checkpoint/split/'groups_ranking_score.parquet';tables.append(table);hashes[str(table)]=sha256(table)
                            inputs.append(dict(job_id=identity,checkpoint=checkpoint,learning_rate_multiplier=lr))
                        key=tuple(map(str,tables))
                        if key not in cache:cache[key]=controlled_summary.paired_group_change(*(pd.read_parquet(p) for p in tables))
                        comparison=dict(reference=inputs[0],alternative=inputs[1],**cache[key])
                        comparisons[arm][rule][split][regime][str(seed)]=comparison
                        values[regime][str(seed)]={'mean_change':comparison['mean_change']}
                spreads[arm][rule][split]={r:controlled_summary.seed_spread(v) for r,v in values.items()}
                spreads[arm][rule][split]['A_minus_B']=controlled_summary.seed_spread({str(s):controlled_summary.subtract(values['A'][str(s)],values['B'][str(s)]) for s in (42,43,44)})
    payload=dict(validation_choice=selection['choice'],settings=settings,decay_minus_reference=comparisons,
        paired_change_seed_spread=spreads,group_tables_sha256=hashes,
        accounting=selection['accounting'],test_used_for_selection=False,new_fits_created=0,study_complete=False,
        interpretation='All three decay settings and all five checkpoint rules remain visible. The selected decay is fixed using only native seed42 B validation and shared across checkpoint rules; each setting retains its existing B-selected LR. Pairwise decay contrasts match full-reference and alternative assays/units within regime and seed, using their mutually defined eligible cohort. They need not share a cohort across the two alternatives. Conditional whole-unit intervals do not include fitted-seed uncertainty; seed42 tuning exposure and seed43/44 confirmation spread are separate. A-minus-B is descriptive across different held-out populations with no cross-split interval. Prefix stopping uses the same full-horizon schedule and saves no training compute. These comparisons do not independently optimize every stopping algorithm or establish causality.')
    verify_selection()
    for path,digest in {str(selection_path()):source['selection_sha256'],**source['summary_sha256'],**runtime_hashes,**hashes}.items():
        if sha256(Path(path))!=digest:raise ValueError('Regularization evidence changed during reporting')
    immutable(target,dict(source=source,payload_sha256=control_plan.digest_json(payload),**payload));return target


def preflight():
    jobs=family_jobs(read(WORK/'controlled/plan.json'))
    return dict(arms=ARMS,tuning_jobs=[j['id'] for j in jobs],
        summary_candidates={a:[str(p) for p in sorted((WORK/'controlled_summaries'/a).glob('*/complete.json'))] for a in ARMS},
        selection_created=False,new_fits_created=0,scientific_comparison_created=False,
        interpretation='Read-only inventory. Neither completion markers nor report paths alone establish scientific completion.')


if __name__=='__main__':
    if ROOT!=Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Use CPU-only reporting in Atum Prot2Mol')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('select','analyze','preflight'))
    parser.add_argument('--summaries',nargs='+',type=Path);args=parser.parse_args()
    print(json.dumps(preflight(),indent=2) if args.action=='preflight' else prepare_selection() if args.action=='select' else analyze(args.summaries))
