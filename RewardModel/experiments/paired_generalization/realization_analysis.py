"""CPU diagnostics and descriptive seed/allocation summaries of frozen B models.

Reuse the primary metric definitions on allocation-specific rows and units.
The allocations share training support and are dependent; their score spread
is never used as an independent-split confidence interval or a selection rule.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT,WORK,sha256
    from . import selected_analysis,similarity_composition,realization_protocol
    from . import realization_postselection,realization_selection,control_plan
else:
    from run import ROOT,WORK,sha256
    import selected_analysis,similarity_composition,realization_protocol
    import realization_postselection,realization_selection,control_plan

read=selected_analysis.read
immutable=selected_analysis.immutable


def software():
    result=selected_analysis.software()
    for name in ('realization_analysis.py','realization_postselection.py','realization_losses.py','realization_selection.py'):
        path=Path(__file__).with_name(name)
        result['source_sha256'][str(path.relative_to(ROOT))]=sha256(path)
    return result


def references(contract_path, design_path):
    """Repartition existing descriptors by frozen whole-unit membership."""
    contract_path,design_path=Path(contract_path),Path(design_path)
    contract,layout,indices=realization_protocol.load_contract(contract_path.parent)
    design=similarity_composition.verify_design(design_path)
    if design['source_sha256'].get(contract['control_plan_path'])!=contract['control_plan_sha256']:
        raise ValueError('Allocation and descriptor designs use different primary plans')
    rows=pd.concat([pd.read_parquet(design_path/f'rows_B_{split}.parquet') for split in ('val','test')],ignore_index=True)
    groups=pd.concat([pd.read_parquet(design_path/f'groups_B_{split}.parquet') for split in ('val','test')],ignore_index=True)
    locations=realization_protocol.align_rows(layout,rows)
    rows=rows.iloc[np.argsort(locations)].reset_index(drop=True)
    if (not np.array_equal(rows.assay_id.astype(str),layout.assay_id.astype(str)) or
        not np.array_equal(rows.unit_id.to_numpy(),layout.component.to_numpy()) or
        groups.group_id.duplicated().any() or rows.max_similarity.any()):
        raise ValueError('Allocation scaffold, assay or indivisible-unit descriptors changed')
    a_groups={split:pd.read_parquet(design_path/f'groups_A_{split}.parquet') for split in ('val','test')}
    refs={};coverage={}
    for allocation,counts in contract['realizations'].items():
        refs[allocation]={};coverage[allocation]={}
        for split in ('val','test'):
            selected=rows.iloc[indices[f'{allocation}_{split}']].reset_index(drop=True)
            subset=groups[groups.group_id.isin(selected.group_id)].sort_values('group_id').reset_index(drop=True)
            sizes=selected.groupby('group_id').size().sort_index()
            if (not np.array_equal(subset.group_id,sizes.index) or
                not np.array_equal(subset['size'],sizes.to_numpy()) or
                dict(rows=len(selected),units=int(selected.unit_id.nunique()),groups=len(subset))!=counts[split]):
                raise ValueError('Allocation descriptors fragment an assay or change coverage')
            cells=similarity_composition.composition_weights(a_groups[split],subset)
            fingerprint=control_plan.digest_json(dict(contract_sha256=sha256(contract_path),allocation=allocation,role=split))
            refs[allocation][split]=dict(rows=selected,groups=subset,cells=cells,fingerprint=fingerprint)
            coverage[allocation][split]=dict(**counts[split],eligible_groups=int(subset.eligible.sum()),
                supported_target_mass=float(cells.loc[cells.common_support,'target_mass'].sum()),
                unsupported_cells=cells.loc[(cells.B_groups>0)&~cells.common_support,'cell'].tolist())
        selected_analysis.unit_map_for(refs[allocation])
    return refs,dict(contract_sha256=sha256(contract_path),design_sha256=sha256(design_path/'complete.json'),
        coverage=coverage,interpretation='Fixed descriptive cells recomputed for each B allocation; no A model score is inferred or fabricated. B training support and scaffold geometry are unchanged.')


def evaluation_inputs(identity, *, cpu_qualification=False):
    if Path(identity).name!=identity:raise ValueError('Expected a registered job ID')
    directory=WORK/'realization_postselection'/identity
    plan=read(directory/'plan.json');complete=read(directory/'complete.json')
    verified=realization_postselection.verify_complete(directory/'plan.json',cpu_qualification=cpu_qualification)
    if complete!=verified or plan['cpu_qualification']!=cpu_qualification:
        raise ValueError('Completed allocation evaluation changed or has wrong provenance')
    needed={(allocation,rule['checkpoint'],split) for allocation,rules in plan['rules'].items()
            for rule in rules.values() for split in ('val','test')}
    evaluations={}
    for name,source in complete['evidence'].items():
        checkpoint,allocation,split=name.split('/')
        key=(allocation,checkpoint,split)
        if key not in needed:raise ValueError('Unexpected selected-model evaluation')
        path=Path(source['output'])/'complete.json'
        if sha256(path)!=source['completion_sha256']:raise ValueError('Selected-model completion changed')
        evaluations[key]=read(path)
    if set(evaluations)!=needed:raise ValueError('Selected-model evaluation omits a required role')
    return plan,evaluations


def training_noise():
    primary=read(WORK/'controlled/plan.json');path=WORK/'tier1/replicates/noise_summary.json'
    digest=sha256(path)
    if digest!=primary['noise_summary_sha256']:raise ValueError('Training noise source changed')
    noise=read(path)['B']['train']['noise_scale']['rms_replicate_difference']
    if noise!=primary['training_noise_rms']['B']:raise ValueError('Training-only noise threshold changed')
    return noise,digest


def verify_statistics(output, result, plan):
    """Bind summary values to the exact selected, hash-checked child analyses."""
    selected_analysis.verify_artifacts(output,result)
    if result['job']!=plan['job'] or result['rules']!=plan['rules'] or set(result['allocations'])!=set(plan['rules']):
        raise ValueError('Analysis job or checkpoint-rule coverage changed')
    for allocation,rules in plan['rules'].items():
        expected={r['checkpoint'] for r in rules.values()}
        if set(result['allocations'][allocation])!=expected:
            raise ValueError('Analysis selected-checkpoint coverage changed')
        for checkpoint,statistics in result['allocations'][allocation].items():
            name=f'{allocation}/{checkpoint}/complete.json'
            if name not in result['artifacts_sha256']:
                raise ValueError('Selected checkpoint analysis is not hash-bound')
            child=read(output/name)
            if (child['statistics']!=statistics or child['source']!=result['source'] or
                child['allocation']!=allocation or child['checkpoint']!=plan['checkpoints'][checkpoint]):
                raise ValueError('Aggregated statistics differ from verified checkpoint artifacts')
            selected_analysis.verify_artifacts((output/name).parent,child)


def analyze_job(identity, *, cpu_qualification=False):
    import fcntl
    from reward_model.training.config import load_reward_training_config
    if Path(identity).name!=identity:raise ValueError('Expected a registered job ID')
    plan,evaluations=evaluation_inputs(identity,cpu_qualification=cpu_qualification)
    config=load_reward_training_config(plan['job']['config_path'])
    if config.model.ranking_min_pchembl_span!=.5 or config.data.ranking_min_pchembl_span!=.5:
        raise ValueError('Analysis requires the registered headline eligibility')
    refs,ref_info=references(plan['contract_path'],WORK/'tier1/similarity_composition/design')
    noise,noise_digest=training_noise()
    directory=WORK/'realization_postselection'/identity
    code=software()
    source=dict(evaluation_plan_sha256=sha256(directory/'plan.json'),
        evaluation_complete_sha256=sha256(directory/'complete.json'),references=ref_info,
        noise_summary_sha256=noise_digest,software=code,cpu_qualification=cpu_qualification)
    if ROOT==Path('/home/atabey/Prot2Mol'):
        if __package__:from .prepare_random import archive_directory
        else:from prepare_random import archive_directory
        archive_directory(WORK/'realization_analysis')
    output=WORK/'realization_analysis'/identity/control_plan.digest_json(source)[:16]
    output.mkdir(parents=True,exist_ok=True)
    with (output/'analysis.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        immutable(output/'analysis_plan.json',source)
        if (output/'complete.json').exists():
            result=read(output/'complete.json')
            if result['source']!=source:raise ValueError('Completed allocation analysis changed')
            verify_statistics(output,result,plan)
            return output/'complete.json'
        results={}
        for allocation,rules in plan['rules'].items():
            results[allocation]={}
            for checkpoint in sorted({r['checkpoint'] for r in rules.values()}):
                target=output/allocation/checkpoint
                if (target/'complete.json').exists():
                    result=read(target/'complete.json')
                    if (result['source']!=source or result['allocation']!=allocation or
                        result['checkpoint']!=plan['checkpoints'][checkpoint]):
                        raise ValueError('Cached allocation analysis changed')
                    selected_analysis.verify_artifacts(target,result)
                else:
                    if target.exists():raise RuntimeError('Partial allocation analysis requires inspection')
                    pair={split:evaluations[allocation,checkpoint,split] for split in ('val','test')}
                    statistics,tables=selected_analysis.diagnose_checkpoint(pair,refs[allocation],regime='B',
                        noise_rms=noise,ranking_readout=config.model.ranking_readout)
                    target.mkdir(parents=True)
                    for name,table in tables.items():
                        path=target/(name+'.parquet');path.parent.mkdir(exist_ok=True)
                        table.to_parquet(path,index=False)
                    result=dict(source=source,allocation=allocation,checkpoint=plan['checkpoints'][checkpoint],
                        statistics=statistics,artifacts_sha256={str(p.relative_to(target)):sha256(p) for p in sorted(target.rglob('*.parquet'))})
                    immutable(target/'complete.json',result)
                results[allocation][checkpoint]=result['statistics']
                print(json.dumps(dict(job=identity,allocation=allocation,checkpoint=checkpoint,phase='allocation_analysis_complete')),flush=True)
        loss_plan=read(plan['loss_plan_path'])
        result=dict(source=source,job=plan['job'],rules=plan['rules'],allocations=results,
            selection_sha256=loss_plan['selection_sha256'],analysis_protocol_sha256=control_plan.digest_json(code),
            seed_role='tuning_exposed' if plan['job']['seed']==42 else 'confirmation',
            cpu_qualification=cpu_qualification,
            artifacts_sha256={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*'))
                if p.is_file() and p.name not in ('analysis.lock','analysis_plan.json')},
            interpretation='Per-fit allocation diagnostics. Whole-unit intervals condition on fitted weights and assignments. Seed and dependent-allocation spread require their separate complete grid; no diagnostic selects a model.')
        immutable(output/'complete.json',result)
    return output/'complete.json'


def spread(values):
    finite=[float(v) for v in values if v is not None and math.isfinite(v)]
    return dict(total=len(values),defined=len(finite),mean=float(np.mean(finite)) if finite else None,
        sample_sd=float(np.std(finite,ddof=1)) if len(finite)>1 else None,
        minimum=min(finite) if finite else None,maximum=max(finite) if finite else None)


def summarize_records(records, selection):
    """Complete 3-seed x 3-dependent-allocation matrix; descriptive spread only."""
    if len(selection['selections'])!=3:raise ValueError('Need three frozen dependent allocations')
    expected={control_plan.job_id(selection['arm'],'B',seed,item['learning_rate_multiplier'])
              for item in selection['selections'].values() for seed in (42,43,44)}
    if len(records)!=len(expected) or {r['job']['id'] for r in records}!=expected:
        raise ValueError('Need exactly the complete selected seed/allocation grid')
    if len({r['analysis_protocol_sha256'] for r in records})!=1:
        raise ValueError('Metric implementations differ across the grid')
    matrix={};rule_names={'best_macro','best_loss','patience3_macro','patience3_loss','average_from_loss_minimum'};readouts=set()
    for record in records:
        job=record['job']
        allocations={a for a,s in selection['selections'].items() if s['learning_rate_multiplier']==job['learning_rate_multiplier']}
        if (record['cpu_qualification'] or job['regime']!='B' or job['arm']!=selection['arm'] or
            job['id']!=control_plan.job_id(selection['arm'],'B',job['seed'],job['learning_rate_multiplier']) or
            record['selection_sha256']!=selection['selection_sha256'] or
            set(record['allocations'])!=allocations or set(record['rules'])!=allocations or
            record['seed_role']!=('tuning_exposed' if job['seed']==42 else 'confirmation')):
            raise ValueError('Analysis has mismatched selection, allocation coverage or seed provenance')
        for allocation in allocations:
            rules=record['rules'][allocation]
            if set(rules)!=rule_names:raise ValueError('Checkpoint-rule coverage differs')
            for rule,choice in rules.items():
                statistics=record['allocations'][allocation][choice['checkpoint']]
                readouts.add(statistics['ranking_readout'])
                for scorer in selected_analysis.SCORERS:
                    metric=statistics['partitions']['test']['metrics'][scorer]
                    matrix[allocation,rule,scorer,job['seed']]=dict(job_id=job['id'],checkpoint=choice['checkpoint'],
                        learning_rate_multiplier=job['learning_rate_multiplier'],
                        macro_spearman=metric['original_eligible_macro_spearman'],
                        defined_fraction=metric['eligible_defined_fraction'],eligible_groups=metric['eligible_groups'])
    if len(readouts)!=1:raise ValueError('Configured ranking readout differs across the grid')
    by_allocation={};by_seed={}
    for allocation in sorted(selection['selections']):
        by_allocation[allocation]={}
        for rule in sorted(rule_names):
            by_allocation[allocation][rule]={}
            for scorer in selected_analysis.SCORERS:
                values={str(seed):matrix[allocation,rule,scorer,seed] for seed in (42,43,44)}
                by_allocation[allocation][rule][scorer]=dict(seeds=values,
                    all_three=spread([values[str(s)]['macro_spearman'] for s in (42,43,44)]),
                    confirmation_only=spread([values[str(s)]['macro_spearman'] for s in (43,44)]))
    for seed in (42,43,44):
        by_seed[str(seed)]={rule:{scorer:spread([matrix[a,rule,scorer,seed]['macro_spearman']
            for a in sorted(selection['selections'])]) for scorer in selected_analysis.SCORERS} for rule in sorted(rule_names)}
    return selected_analysis.finite_json(dict(arm=selection['arm'],ranking_readout=next(iter(readouts)),by_allocation=by_allocation,
        within_seed_allocation_sensitivity=by_seed,test_used_for_selection=False,
        interpretation='Descriptive means, sample SDs and ranges only; no confidence interval across seeds or allocations. Seed42 selected LR and is tuning-exposed; seeds43/44 are confirmations. Allocations share training support and are dependent. Their sensitivity includes different test groups and allocation-specific validation LR/checkpoint choices. Do not pool the nine cells as independent replicates or substitute these spreads for whole-unit intervals.'))


def summarize(paths):
    paths=[Path(p) for p in paths];records=[read(p) for p in paths]
    if not records:raise ValueError('Need completed analysis files')
    arm=records[0]['job']['arm'];selection=realization_selection.verify_selection(arm)
    selection=dict(selection,selection_sha256=sha256(realization_selection.destination(arm)/'selection.json'))
    _,noise_digest=training_noise();reference_cache={};code=software()
    for path,record in zip(paths,records):
        directory=WORK/'realization_postselection'/record['job']['id']
        plan,_=evaluation_inputs(record['job']['id'])
        contract=plan['contract_path']
        if contract not in reference_cache:
            _,reference_cache[contract]=references(contract,WORK/'tier1/similarity_composition/design')
        if (record['source']['evaluation_plan_sha256']!=sha256(directory/'plan.json') or
            record['source']['evaluation_complete_sha256']!=sha256(directory/'complete.json') or
            record['source']['software']!=code or record['source']['references']!=reference_cache[contract] or
            record['source']['noise_summary_sha256']!=noise_digest or record['source']['cpu_qualification'] or
            record['analysis_protocol_sha256']!=control_plan.digest_json(record['source']['software'])):
            raise ValueError('Analysis source changed before aggregation')
        verify_statistics(path.parent,record,plan)
    result=summarize_records(records,selection)
    evidence={str(p):sha256(p) for p in paths}
    target=WORK/'realization_analysis/summaries'/arm/control_plan.digest_json(evidence)[:16]/'summary.json'
    immutable(target,dict(**result,analysis_evidence_sha256=evidence))
    return target


if __name__=='__main__':
    if ROOT!=Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Scientific reporting requires CPU-only Atum Prot2Mol')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('analyze','summarize','qualify-references'))
    parser.add_argument('--job-id');parser.add_argument('--analyses',nargs='+',type=Path)
    args=parser.parse_args()
    if args.action=='qualify-references':
        _,info=references(WORK/'controlled/realizations/plan.json',WORK/'tier1/similarity_composition/design')
        print(json.dumps(dict(**info,model_loaded=False,predictions_used=False,scientific_scores_computed=False),indent=2))
    else:
        print(analyze_job(args.job_id) if args.action=='analyze' else summarize(args.analyses))
