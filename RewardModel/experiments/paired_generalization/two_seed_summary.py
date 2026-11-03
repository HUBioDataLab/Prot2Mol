"""Complete primary-arm reporting without test-driven selection.

User amendment: six seed42 tuning fits and two selected seed43 confirmations.
Seed spread, LR sensitivity and conditional whole-unit intervals stay separate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT,WORK,sha256
    from . import selected_analysis,postselection,control_plan,similarity_composition
    from .realization_analysis import spread
    from .analyze_predictions import unit_bootstrap
else:
    from run import ROOT,WORK,sha256
    import selected_analysis,postselection,control_plan,similarity_composition
    from realization_analysis import spread
    from analyze_predictions import unit_bootstrap

read=selected_analysis.read
immutable=selected_analysis.immutable
RULES=('best_macro','best_loss','patience3_macro','patience3_loss','average_from_loss_minimum')
SEEDS=(42,43)
METRICS=('original_eligible_macro_spearman','macro_spearman','size_weighted_spearman',
         'macro_kendall_tau_b','pooled_within_group_concordance','pooled_ordering_accuracy')


def software():
    result=selected_analysis.software()
    for name in ('two_seed_summary.py','realization_analysis.py','two_seed_policy.py'):
        path=Path(__file__).with_name(name)
        result['source_sha256'][str(path.relative_to(ROOT))]=sha256(path)
    return result


def expected_jobs(arm, multiplier):
    if multiplier not in control_plan.LEARNING_RATE_MULTIPLIERS:
        raise ValueError('Selection is outside the frozen LR grid')
    return {control_plan.job_id(arm,regime,seed,lr):(regime,seed,lr)
        for seed in SEEDS for regime in ('A','B')
        for lr in (control_plan.LEARNING_RATE_MULTIPLIERS if seed==42 else (multiplier,))}


def metric_point(record, rule, split, scorer):
    choice=record['rules'][rule]
    statistics=record['checkpoints'][choice['checkpoint']]
    metric=statistics['partitions'][split]['metrics'][scorer]
    return dict(job_id=record['job']['id'],checkpoint=choice['checkpoint'],
        metrics={**{name:metric[name] for name in METRICS},
            **{f'eligible_ndcg_at_{k}':metric['matched_eligible_ndcg'][str(k)] for k in (1,3,5)}},
        coverage={name:metric[name] for name in ('groups','defined_groups','defined_fraction',
            'eligible_groups','eligible_defined_fraction','all_pairs','comparable_pairs')},
        conditional_unit_interval=metric['unit_bootstrap_original_eligible'])


def subtract(left, right):
    if set(left)!=set(right):raise ValueError('Compared metrics differ')
    return {name:left[name]-right[name] if left[name] is not None and right[name] is not None else None for name in left}


def seed_spread(values):
    if set(values)!={str(s) for s in SEEDS}:raise ValueError('Need exactly the approved seeds42/43')
    names=set(values['42'])
    if any(set(v)!=names for v in values.values()):raise ValueError('Seed metrics differ')
    return dict(individual=values,
        both_seeds={name:spread([values[str(s)][name] for s in SEEDS]) for name in sorted(names)},
        single_confirmation={name:spread([values['43'][name]]) for name in sorted(names)},
        confirmation_seed_variance_available=False)


def summarize_records(records, decision, selection_sha256):
    """Describe all registered results; never pick a winner using these scores."""
    arm=decision['arm'];multiplier=decision['learning_rate_multiplier']
    expected=expected_jobs(arm,multiplier)
    if len(records)!=len(expected) or {r['job']['id'] for r in records}!=set(expected):
        raise ValueError('Need exactly six tuning and two selected confirmation analyses')
    protocols={control_plan.digest_json({k:r['source'][k] for k in ('software','design_sha256','noise_summary_sha256')}) for r in records}
    if len(protocols)!=1:raise ValueError('Analyses use different metrics, descriptors or noise thresholds')
    index={};readouts=set()
    for record in records:
        job=record['job'];regime,seed,lr=expected[job['id']];selected=lr==multiplier
        rules=set(RULES if selected else ('best_macro',))
        if (record['phase']!='selected_prediction_analysis_complete' or
            (job['arm'],job['regime'],job['seed'],job['learning_rate_multiplier'])!=(arm,regime,seed,lr) or
            job['phase']!=('tuning' if seed==42 else 'confirmation') or
            job['plan_sha256']!=decision['plan_sha256'] or record['selection_sha256']!=selection_sha256 or
            record['selected_learning_rate'] is not selected or set(record['rules'])!=rules or
            record['seed_role']!=('tuning_exposed' if seed==42 else 'confirmation') or
            set(record['checkpoints'])!={r['checkpoint'] for r in record['rules'].values()}):
            raise ValueError('Analysis differs from the registered selection, seed or checkpoint rules')
        for statistics in record['checkpoints'].values():
            readouts.add(statistics['ranking_readout'])
            if statistics['regime']!=regime or set(statistics['partitions'])!={'val','test'}:
                raise ValueError('Analysis partition/regime differs')
        index[regime,seed,lr]=record
    if len(readouts)!=1:raise ValueError('Readout differs within one arm')
    selected_results={};rule_changes={}
    for rule in RULES:
        selected_results[rule]={};rule_changes[rule]={}
        for split in ('val','test'):
            selected_results[rule][split]={};rule_changes[rule][split]={}
            for scorer in selected_analysis.SCORERS:
                points={r:{str(s):metric_point(index[r,s,multiplier],rule,split,scorer) for s in SEEDS} for r in ('A','B')}
                cell={r:dict(points=points[r],seed_spread=seed_spread({s:p['metrics'] for s,p in points[r].items()})) for r in ('A','B')}
                cell['matched_A_minus_B']=seed_spread({str(s):subtract(points['A'][str(s)]['metrics'],points['B'][str(s)]['metrics']) for s in SEEDS})
                selected_results[rule][split][scorer]=cell
                if rule!='best_macro':
                    rule_changes[rule][split][scorer]={r:seed_spread({str(s):subtract(points[r][str(s)]['metrics'],
                        metric_point(index[r,s,multiplier],'best_macro',split,scorer)['metrics']) for s in SEEDS}) for r in ('A','B')}
    candidates={str(lr):{split:{scorer:{r:metric_point(index[r,42,lr],'best_macro',split,scorer)
        for r in ('A','B')} for scorer in selected_analysis.SCORERS} for split in ('val','test')}
        for lr in control_plan.LEARNING_RATE_MULTIPLIERS}
    lr_spread={split:{scorer:{r:{name:spread([candidates[str(lr)][split][scorer][r]['metrics'][name]
        for lr in control_plan.LEARNING_RATE_MULTIPLIERS]) for name in candidates[str(multiplier)][split][scorer][r]['metrics']}
        for r in ('A','B')} for scorer in selected_analysis.SCORERS} for split in ('val','test')}
    rule_changes.pop('best_macro')
    return selected_analysis.finite_json(dict(arm=arm,ranking_readout=next(iter(readouts)),selected_learning_rate_multiplier=multiplier,
        selected=selected_results,rule_minus_best_macro=rule_changes,tuning_candidates=candidates,lr_sensitivity_seed42=lr_spread,
        test_used_for_selection=False,study_complete=False,
        interpretation='All LR choices come from frozen B validation. Seed42 is tuning-exposed; seed43 is the only confirmation. Confirmation-seed variance is unavailable; both-seed means/SDs/ranges and seed42 LR sensitivity are descriptive, not confidence intervals. A-minus-B differences pair the same arm/seed/LR/rule but different held-out populations, with no cross-split interval. Scalar rule differences retain their original defined cohorts; separate paired-unit contrasts restrict both checkpoints to mutually defined eligible assays. Undefined results remain null and coverage stays explicit. Per-fit intervals condition on fitted weights and units, not training-seed uncertainty. Diagnostic pathways may be inactive; ranking_score is the configured deployed readout.'))


def paired_group_change(first, second):
    columns=['group_id','component','size','original_ranking_eligible']
    if first.group_id.duplicated().any() or second.group_id.duplicated().any():
        raise ValueError('Duplicate comparison group')
    first=first.sort_values('group_id').reset_index(drop=True)
    second=second.sort_values('group_id').reset_index(drop=True)
    if not first[columns].equals(second[columns]):
        raise ValueError('Checkpoint comparison changed groups, units or eligibility')
    eligible=first.original_ranking_eligible
    first_defined=eligible&np.isfinite(first.spearman)
    second_defined=eligible&np.isfinite(second.spearman)
    matched=first_defined&second_defined
    delta=(second.spearman-first.spearman).where(matched)
    boot=unit_bootstrap(pd.DataFrame(dict(component=first.component,spearman=delta)))
    return selected_analysis.finite_json(dict(groups=len(first),eligible_groups=int(eligible.sum()),
        first_defined_groups=int(first_defined.sum()),second_defined_groups=int(second_defined.sum()),
        mutually_defined_groups=int(matched.sum()),first_macro=first.loc[matched,'spearman'].mean(),
        second_macro=second.loc[matched,'spearman'].mean(),mean_change=delta.mean(),unit_bootstrap=boot,
        interpretation='Second minus first on mutually defined eligible assays. Resample complete units, including zero-eligible units; conditional on these fitted checkpoints and assignments. This is not a seed interval or a cross-split comparison.'))


def verify_analysis(path, design_path):
    path=Path(path);record=read(path);identity=record['job']['id']
    if Path(identity).name!=identity:raise ValueError('Expected registered primary job ID')
    directory=WORK/'postselection'/identity
    plan,_=selected_analysis.verify_completed_evaluations(directory)
    for key in ('job','rules','seed_role','selected_learning_rate','selection_sha256'):
        if record[key]!=plan[key]:raise ValueError('Analysis differs from its completed native evaluation')
    source=record['source'];primary=read(WORK/'controlled/plan.json')
    noise=WORK/'tier1/replicates/noise_summary.json'
    if (record['phase']!='selected_prediction_analysis_complete' or
        source['evaluation_plan_sha256']!=sha256(directory/'plan.json') or
        source['evaluation_complete_sha256']!=sha256(directory/'complete.json') or
        source['software']!=selected_analysis.software() or
        source['design_sha256']!=sha256(design_path/'complete.json') or
        source['noise_summary_sha256']!=sha256(noise) or sha256(noise)!=primary['noise_summary_sha256']):
        raise ValueError('Analysis source changed before summary')
    selected_analysis.references(plan,design_path)
    selected_analysis.verify_artifacts(path.parent,record)
    checkpoints={c['id']:c for c in plan['checkpoints']}
    if set(record['checkpoints'])!=set(checkpoints):raise ValueError('Selected-checkpoint coverage changed')
    for name,metadata in checkpoints.items():
        relative=f'{name}/complete.json'
        if relative not in record['artifacts_sha256']:raise ValueError('Checkpoint analysis is not hash-bound')
        child=read(path.parent/relative)
        if child['source']!=source or child['checkpoint']!=metadata or child['statistics']!=record['checkpoints'][name]:
            raise ValueError('Reported statistics differ from checkpoint analysis')
        required={f'{split}/groups_{scorer}.parquet' for split in ('val','test') for scorer in selected_analysis.SCORERS}
        if not required<=set(child['artifacts_sha256']):raise ValueError('Checkpoint analysis omits group tables')
        selected_analysis.verify_artifacts(path.parent/name,child)
    return record


def summarize(arm, paths):
    if __package__:
        from . import two_seed_policy
    else:
        import two_seed_policy
    two_seed_policy.verify()
    directory=WORK/'controlled';primary=read(directory/'plan.json')
    decision,selection_path=postselection.verify_selection(directory,arm,primary)
    paths=[Path(p) for p in paths];design=WORK/'tier1/similarity_composition/design'
    similarity_composition.verify_design(design)
    records=[verify_analysis(p,design) for p in paths]
    result=summarize_records(records,decision,sha256(selection_path))
    locations={r['job']['id']:p for r,p in zip(records,paths)}
    source=dict(analysis_sha256={str(p):sha256(p) for p in paths},selection_sha256=sha256(selection_path),
        primary_plan_sha256=sha256(directory/'plan.json'),software=software(),
        seed_policy_sha256=sha256(two_seed_policy.destination()/'policy.json'))
    target=WORK/'controlled_summaries_two_seed'/arm/control_plan.digest_json(source)[:16]/'complete.json'
    if target.exists():
        saved=read(target)
        payload={k:saved[k] for k in ('summary','composition_pairs','paired_checkpoint_changes')}
        if (saved['source']!=source or saved['summary']!=result or
            saved['payload_sha256']!=control_plan.digest_json(payload)):
            raise ValueError('Completed summary changed')
        return target
    composition={};paired={}
    for seed in SEEDS:
        lrs=control_plan.LEARNING_RATE_MULTIPLIERS if seed==42 else (decision['learning_rate_multiplier'],)
        for lr in lrs:
            a,b=[locations[control_plan.job_id(arm,r,seed,lr)] for r in ('A','B')]
            composition[f'seed{seed}_lr{lr}']=selected_analysis.compare_analyses(a,b,design)
    for record,path in zip(records,paths):
        if not record['selected_learning_rate']:continue
        cache={};contrasts={}
        first=record['rules']['best_macro']['checkpoint']
        for rule in RULES[1:]:
            second=record['rules'][rule]['checkpoint'];contrasts[rule]={}
            for split in ('val','test'):
                contrasts[rule][split]={}
                for scorer in selected_analysis.SCORERS:
                    key=(first,second,split,scorer)
                    if key not in cache:
                        tables=[pd.read_parquet(path.parent/c/split/f'groups_{scorer}.parquet') for c in (first,second)]
                        cache[key]=paired_group_change(*tables)
                    contrasts[rule][split][scorer]=cache[key]
        paired[record['job']['id']]=contrasts
    payload=dict(summary=result,composition_pairs=composition,paired_checkpoint_changes=paired)
    immutable(target,dict(source=source,payload_sha256=control_plan.digest_json(payload),**payload))
    return target


def preflight(arm):
    """Observe available completions without fabricating selection or reports."""
    directory=WORK/'controlled';primary=read(directory/'plan.json')
    if arm not in {a['name'] for a in primary['arms']}:raise ValueError('Unregistered arm')
    jobs=[j for j in primary['tuning_jobs'] if j['arm']==arm]
    if len(jobs)!=6:raise ValueError('Expected the six frozen tuning fits')
    candidates={j['id']:(directory/'runs'/j['id']/'controlled_training_complete.json').exists() for j in jobs}
    decision_path=directory/'selections'/f'{arm}.json'
    decision=None;analyses={}
    if decision_path.exists():
        decision,_=postselection.verify_selection(directory,arm,primary)
        for identity in expected_jobs(arm,decision['learning_rate_multiplier']):
            analyses[identity]=[str(p) for p in sorted((WORK/'postselection_analysis'/identity).glob('*/complete.json'))]
    return dict(arm=arm,primary_plan_sha256=sha256(directory/'plan.json'),tuning_completion_markers=candidates,
        selection_verified=decision is not None,analysis_candidates=analyses,
        scientific_summary_created=False,interpretation='Read-only inventory; marker presence alone is not verified training or analysis completion. Summary requires all eight verified analyses.')


if __name__=='__main__':
    if ROOT!=Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Use CPU-only reporting in Atum Prot2Mol')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('summarize','preflight'))
    parser.add_argument('--arm',required=True);parser.add_argument('--analyses',nargs='+',type=Path)
    args=parser.parse_args()
    print(json.dumps(preflight(args.arm),indent=2) if args.action=='preflight' else summarize(args.arm,args.analyses))
