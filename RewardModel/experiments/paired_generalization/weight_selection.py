"""Q34/E39: compare validation choices within the frozen five-setting weight sweep.

This is a read-only decision/report layer over existing primary tuning fits.
It creates no fits or confirmations and never reads test metrics or predictions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

if __package__:
    from .run import ROOT,WORK,sha256
    from . import control_plan,postselection,control_driver
else:
    from run import ROOT,WORK,sha256
    import control_plan,postselection,control_driver

read=postselection.read_json
immutable=postselection.immutable_json
WEIGHTS={
    'full':(.25,.25,1.),
    'bce_low':(.25,.0625,1.),
    'bce_high':(.25,1.,1.),
    'contrastive_low':(.25,.25,.25),
    'contrastive_high':(.25,.25,4.),
}
WEIGHT_KEYS=('ranking_loss_weight','classification_loss_weight','contrastive_loss_weight')


def software():
    from importlib.metadata import version
    paths=[Path(__file__),*[Path(__file__).with_name(n) for n in ('control_plan.py','postselection.py','control_driver.py','run.py')]]
    return dict(source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in paths},packages={'PyYAML':version('PyYAML')})


def family_jobs(primary):
    declared={a['name'] for a in primary['arms'] if a['family']=='static_weight_sweep'}
    if declared!=set(WEIGHTS)-{'full'} or primary['shared_reference']['original_weights']!='full':
        raise ValueError('Frozen weight-sweep membership changed')
    jobs=[j for j in primary['tuning_jobs'] if j['arm'] in WEIGHTS]
    expected={control_plan.job_id(a,r,42,lr) for a in WEIGHTS for r in ('A','B') for lr in control_plan.LEARNING_RATE_MULTIPLIERS}
    if len(jobs)!=30 or {j['id'] for j in jobs}!=expected:
        raise ValueError('Need the exact thirty existing A/B tuning jobs')
    for job in jobs:
        config=job['config'];model=config['model']
        if (job['seed']!=42 or job['phase']!='tuning' or tuple(model[k] for k in WEIGHT_KEYS)!=WEIGHTS[job['arm']] or
            model['ranking_readout']!='ranking_head' or config['training']['num_train_epochs']!=15):
            raise ValueError('Weight setting, readout or training budget differs from the declared sweep')
    return jobs


def choose(records):
    """Same LR rule as the primary protocol; fixed arm order breaks family ties."""
    expected={(a,r,lr) for a in WEIGHTS for r in ('A','B') for lr in control_plan.LEARNING_RATE_MULTIPLIERS}
    keys=[(r['arm'],r['regime'],r['learning_rate_multiplier']) for r in records]
    if len(records)!=30 or set(keys)!=expected:raise ValueError('Incomplete or duplicated weight/LR/split grid')
    decisions={};within={};candidates={}
    for regime in ('A','B'):
        within[regime]={};candidates[regime]=[]
        for arm,weights in WEIGHTS.items():
            items=[{k:r[k] for k in ('learning_rate_multiplier','validation_macro_spearman','validation_defined_fraction')}
                for r in records if r['regime']==regime and r['arm']==arm]
            for item in items:
                for key in ('validation_macro_spearman','validation_defined_fraction'):
                    if item[key] is not None and not math.isfinite(item[key]):item[key]=None
            selected=control_plan.select_learning_rate(items)
            selected['criterion']=f'Maximum {regime} validation macro Spearman; lower LR breaks a tie; defined coverage reported'
            within[regime][arm]=dict(arm=arm,weights=dict(zip(('R','B','C'),weights)),**selected)
            candidates[regime].extend(dict(arm=arm,**item) for item in items)
        order=list(WEIGHTS)
        best=max(within[regime].values(),key=lambda r:(r['validation_macro_spearman'],-order.index(r['arm'])))
        decisions[regime]={k:best[k] for k in ('arm','weights','learning_rate_multiplier','validation_macro_spearman','validation_defined_fraction')}
        decisions[regime]['job_ids']={r:control_plan.job_id(best['arm'],r,42,best['learning_rate_multiplier']) for r in ('A','B')}
    # Confirmations already follow each arm's B-selected LR. An alternative A
    # choice can lack repeated-seed coverage; never fabricate or schedule it.
    for regime,decision in decisions.items():
        covered=decision['learning_rate_multiplier']==within['B'][decision['arm']]['learning_rate_multiplier']
        decision['matches_existing_confirmation_lr']=covered
        decision['planned_confirmation_job_ids']=[control_plan.job_id(decision['arm'],r,s,decision['learning_rate_multiplier'])
            for s in (43,44) for r in ('A','B')] if covered else []
        decision['confirmation_status']='Existing primary-plan confirmations; completion not implied' if covered else 'Only seed42 covered at this A-selected LR; no additional fits created'
    return dict(choices=decisions,within_setting=within,all_candidates=candidates,
        different_weights=decisions['A']['arm']!=decisions['B']['arm'],
        different_learning_rates=decisions['A']['learning_rate_multiplier']!=decisions['B']['learning_rate_multiplier'],
        family_tie_order=list(WEIGHTS),test_used_for_selection=False,new_fits_created=0,
        accounting=dict(settings=5,lr_candidates_per_setting_per_split=3,tuning_fits_per_split=15,
            unique_tuning_fits=30,shared_full_reference_fits=6,other_weight_setting_fits=24,
            planned_existing_confirmations=20),
        interpretation='A-validation choice is a diagnostic comparator; B validation is the deployment-oriented selector. Same native best-macro epoch rule and seed42 for all candidates. Family ties prefer the listed fixed setting order, including the shared full reference first. The primary LR rule excludes undefined/nonfinite scores or zero defined coverage and reports coverage; it does not equalize the evaluated defined cohorts. These are tuning results, not independent test or seed evidence. The five settings form an axis sweep, not a Cartesian search or a claim of globally optimal weights. Family-level tuning uses 15 candidates per split; each individual setting uses three.')


def build_selection():
    directory=WORK/'controlled';path=directory/'plan.json';primary=read(path)
    jobs=family_jobs(primary);records=[];evidence=[];qualification=None
    for job in jobs:
        metadata=postselection.registered_job(directory,job['id'],primary)
        candidate,source=control_plan.validation_candidate(metadata)
        output=Path(metadata['output_dir']);runtime_path=output/'runtime_qualification.json'
        runtime=read(runtime_path)
        if runtime['world_size']!=4 or runtime['signature']!=control_plan.digest_json(dict(sources=runtime['sources'],plan_sha256=sha256(path))):
            raise ValueError('Training runtime qualification changed')
        if qualification is None:
            qualification=runtime
            for name,digest in runtime['sources'].items():
                if not (ROOT/name).resolve().is_relative_to(ROOT.resolve()):raise ValueError('Qualified source is outside the project')
                if sha256(ROOT/name)!=digest:raise ValueError('Qualified training source changed')
        elif runtime!=qualification:raise ValueError('Training qualifications differ across weight settings')
        records.append(dict(arm=metadata['arm'],regime=metadata['regime'],**candidate))
        evidence.append(dict(**source,config_sha256=metadata['config_sha256'],
            metadata_sha256=sha256(directory/'jobs'/f"{job['id']}.json"),
            training_complete_sha256=sha256(output/'controlled_training_complete.json'),
            runtime_qualification_sha256=sha256(runtime_path)))
    return dict(format_version=1,primary_plan_sha256=sha256(path),software=software(),
        training_qualification=qualification,validation_evidence=evidence,**choose(records),
        historical_limit='The original config supplies the reference weights; this sweep cannot establish how those weights were chosen before this study.',
        remaining='Scientific test/confirmation comparisons use the existing selected evaluations. This validation decision alone does not close E39 or Q34.')


def destination():
    return WORK/'weight_selection/static_weight_sweep/selection.json'


def prepare():
    result=build_selection();immutable(destination(),result);return destination()


def verify():
    if read(destination())!=build_selection():raise ValueError('Frozen weight comparison changed')
    return read(destination())


def preflight():
    directory=WORK/'controlled';primary=read(directory/'plan.json');jobs=family_jobs(primary)
    completed=[];missing=[]
    for job in jobs:
        metadata=read(directory/'jobs'/f"{job['id']}.json")
        if metadata['id']!=job['id'] or metadata['plan_sha256']!=sha256(directory/'plan.json'):
            raise ValueError('Weight-sweep job registration changed')
        if control_driver.completed_job(metadata):completed.append(job['id'])
        else:missing.append(job['id'])
    return dict(primary_plan_sha256=sha256(directory/'plan.json'),settings=list(WEIGHTS),
        verified_training_completions=completed,missing_training_completions=missing,
        selection_created=False,new_fits_created=0,test_metrics_read=False,
        interpretation='Read-only readiness; no weight/LR choice is made until all thirty registered tuning fits verify.')


if __name__=='__main__':
    if ROOT!=Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Use CPU-only comparison in Atum Prot2Mol')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','verify','preflight'))
    args=parser.parse_args()
    if args.action=='prepare':print(prepare())
    else:print(json.dumps(verify() if args.action=='verify' else preflight(),indent=2))
