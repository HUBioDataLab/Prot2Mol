"""Validation-only decay choice, native receipt gates and paired report IO."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization import two_seed_regularization_analysis as module
from experiments.paired_generalization.run import sha256,write_json
from test_paired_generalization_control_plan import ROOT,noise_fixture


def decisions():
    return [dict(arm=arm,learning_rate_multiplier=lr,validation_macro_spearman=score,validation_defined_fraction=1.)
        for arm,lr,score in [('full',1.,.3),('weight_decay_0p001',3.,.35),('weight_decay_0p1',.3,.2)]]


def test_family_choice_uses_only_validation_and_is_shared_across_rules():
    records=decisions();choice=module.choose(records)
    assert choice['choice']['arm']=='weight_decay_0p001' and choice['choice']['learning_rate_multiplier']==3.
    assert choice['shared_across_checkpoint_rules'] and not choice['test_used_for_selection'] and choice['new_fits_created']==0
    for r in records:r['test_score']=999 if r['arm']=='full' else -999
    assert module.choose(records)==choice
    for r in records:r['validation_macro_spearman']=.4
    assert module.choose(records)['choice']['arm']=='full'


@pytest.mark.parametrize('damage',['missing','duplicate','nonfinite','zero_coverage','other_lr'])
def test_invalid_or_incomplete_validation_choices_are_rejected(damage):
    records=decisions()
    if damage=='missing':records.pop()
    elif damage=='duplicate':records[-1]=records[0]
    elif damage=='nonfinite':records[0]['validation_macro_spearman']=float('nan')
    elif damage=='zero_coverage':records[0]['validation_defined_fraction']=0.
    else:records[0]['learning_rate_multiplier']=10.
    with pytest.raises(ValueError):module.choose(records)


@pytest.fixture
def frozen(tmp_path,monkeypatch):
    root=tmp_path/'project';work=root/'outputs';directory=work/'controlled';root.mkdir()
    monkeypatch.setattr(module,'ROOT',root);monkeypatch.setattr(module,'WORK',work)
    monkeypatch.setattr(module.control_plan,'WORK',work)
    monkeypatch.setattr(module.two_seed_policy,'WORK',work)
    monkeypatch.setattr(module.two_seed_policy,'verify',lambda:dict(seeds=[42,43]))
    write_json(module.two_seed_policy.destination()/'policy.json',dict(seeds=[42,43]))
    monkeypatch.setattr(module,'software',lambda:dict(synthetic_source=True))
    primary=module.control_plan.build_plan(ROOT,work,noise_fixture());write_json(directory/'plan.json',primary)
    code=root/'qualified.py';code.write_text('synthetic qualified source')
    sources={'qualified.py':sha256(code)}
    runtime=dict(world_size=4,sources=sources,signature=module.control_plan.digest_json(dict(sources=sources,plan_sha256=sha256(directory/'plan.json'))))
    winners={d['arm']:d for d in decisions()};metadata=[]
    for job in module.family_jobs(primary):
        item=module.control_plan.materialize_job(job,directory,sha256(directory/'plan.json'));metadata.append(item)
        out=Path(item['output_dir']);winner=winners[item['arm']]
        score=winner['validation_macro_spearman']-.01*abs(item['learning_rate_multiplier']-winner['learning_rate_multiplier'])
        if item['regime']=='A':score=.8-.01*abs(item['learning_rate_multiplier']-.3)
        write_json(out/'experiment_summary.json',dict(config_sha256=item['config_sha256'],test_evaluated=False,test_metrics={}))
        for rank in range(4):write_json(out/f'rank_{rank}_complete.json',dict(rank=rank,world_size=4,config_sha256=item['config_sha256']))
        write_json(out/'retained/retention.json',dict(config_sha256=item['config_sha256'],selections={'best_macro':dict(step=10,metric=score)}))
        path=out/'prediction_exports/trajectory/eval_step00000010.json';path.parent.mkdir(parents=True)
        path.with_suffix('.npz').write_bytes(b'synthetic native validation payload')
        write_json(path,dict(predictions_sha256=sha256(path.with_suffix('.npz')),metrics=dict(eval_macro_spearman=score,eval_spearman_defined_fraction=1.)))
        write_json(out/'runtime_qualification.json',runtime)
        write_json(out/'controlled_training_complete.json',dict(config_sha256=item['config_sha256'],plan_sha256=item['plan_sha256'],
            evidence_sha256={'experiment_summary.json':sha256(out/'experiment_summary.json')}))
    confirmations={arm:module.control_plan.confirm(arm) for arm in module.ARMS}
    for selection in confirmations.values():selection['confirmation_jobs']=[j for j in selection['confirmation_jobs'] if j['seed']==43]
    return dict(root=root,work=work,directory=directory,primary=primary,metadata=metadata,code=code,
        runtime=runtime,confirmations=confirmations)


def test_actual_eighteen_fit_validation_chain_freezes_choice_without_test_reads(frozen):
    before={str(p):sha256(p) for p in frozen['work'].rglob('*') if p.is_file()}
    result=module.build_selection()
    assert result['choice']['arm']=='weight_decay_0p001' and result['choice']['learning_rate_multiplier']==3.
    assert result['accounting']['unique_tuning_fits']==18 and result['accounting']['total_primary_fits']==24
    assert {str(p):sha256(p) for p in frozen['work'].rglob('*') if p.is_file()}==before
    for job in frozen['metadata']:write_json(Path(job['output_dir'])/'unrelated_test.json',dict(score=999))
    assert module.build_selection()==result
    path=module.prepare_selection();assert module.verify_selection()==result
    changed=module.read(path);changed['choice']['arm']='full';write_json(path,changed)
    with pytest.raises(ValueError,match='Frozen decay selection'):module.verify_selection()


@pytest.mark.parametrize('damage',['extra_setting','missing_fit','budget','model','full_decay'])
def test_scope_and_common_configuration_are_fixed(frozen,damage):
    primary=copy.deepcopy(frozen['primary'])
    job=next(j for j in primary['tuning_jobs'] if j['arm']=='weight_decay_0p001')
    if damage=='extra_setting':primary['arms'].append(copy.deepcopy(next(a for a in primary['arms'] if a['family']=='regularization')))
    elif damage=='missing_fit':primary['tuning_jobs'].remove(job)
    elif damage=='budget':job['config']['training']['num_train_epochs']=12
    elif damage=='model':job['config']['model']['ranking_loss_weight']=.5
    else:next(j for j in primary['tuning_jobs'] if j['arm']=='full')['config']['training']['weight_decay']=.2
    with pytest.raises(ValueError):module.family_jobs(primary)


@pytest.mark.parametrize('damage',['missing_fit','validation_payload','runtime','source','exposed_test'])
def test_changed_native_evidence_cannot_choose_decay(frozen,damage):
    job=frozen['metadata'][-1];out=Path(job['output_dir'])
    if damage=='missing_fit':(out/'controlled_training_complete.json').unlink()
    elif damage=='validation_payload':(out/'prediction_exports/trajectory/eval_step00000010.npz').write_bytes(b'changed')
    elif damage=='runtime':
        r=module.read(out/'runtime_qualification.json');r['world_size']=2;write_json(out/'runtime_qualification.json',r)
    elif damage=='source':frozen['code'].write_text('changed')
    else:
        r=module.read(out/'experiment_summary.json');r['test_evaluated']=True;write_json(out/'experiment_summary.json',r)
    with pytest.raises((ValueError,RuntimeError,FileNotFoundError)):module.prepare_selection()
    assert not module.selection_path().exists()


@pytest.fixture
def reports(frozen,monkeypatch):
    """Synthetic report assembly; native scientific gates have separate tests."""
    module.prepare_selection();work=frozen['work'];paths=[];locations={};records={};calls=[]
    for arm in module.ARMS:
        choice=frozen['confirmations'][arm];lr=choice['selection']['learning_rate_multiplier'];analyses={}
        for identity,(regime,seed,multiplier) in module.controlled_summary.expected_jobs(arm,lr).items():
            out=work/'controlled/runs'/identity;write_json(out/'runtime_qualification.json',frozen['runtime'])
            path=work/'postselection_analysis'/identity/'synthetic/complete.json'
            rules={rule:dict(checkpoint='early' if rule in ('best_loss','patience3_loss') else 'late') for rule in module.controlled_summary.RULES}
            record=dict(job=dict(id=identity,output_dir=str(out)),rules=rules,
                source=dict(software=dict(native_fixture=True),design_sha256='design',noise_summary_sha256='noise'))
            write_json(path,record);analyses[str(path)]=sha256(path);records[identity]=path
            for checkpoint in ('early','late'):
                for split in ('val','test'):
                    value=.2+(.15 if regime=='A' else 0.)+.01*(seed-42)+{'full':0.,'weight_decay_0p001':.05,'weight_decay_0p1':-.03}[arm]
                    if checkpoint=='early':value-=.02
                    frame=pd.DataFrame(dict(group_id=['a','b','c'],component=[1,1,2],size=[3,4,1],original_ranking_eligible=[True,True,False],spearman=[value,value+.1,np.nan]))
                    table=path.parent/checkpoint/split/'groups_ranking_score.parquet';table.parent.mkdir(parents=True);frame.to_parquet(table,index=False)
            selected={rule:{split:dict(ranking_score=dict(synthetic_score=999 if arm=='full' else -999)) for split in ('val','test')} for rule in module.controlled_summary.RULES}
        path=work/'controlled_summaries_two_seed'/arm/'synthetic/complete.json'
        write_json(path,dict(source=dict(analysis_sha256=analyses,primary_plan_sha256=sha256(frozen['directory']/'plan.json'),software=module.controlled_summary.software()),
            summary=dict(arm=arm,ranking_readout='ranking_head',selected_learning_rate_multiplier=lr,selected=selected,
                rule_minus_best_macro={rule:{split:dict(ranking_score=dict(synthetic=True)) for split in ('val','test')} for rule in module.controlled_summary.RULES[1:]}),paired_checkpoint_changes={}))
        locations[arm]=path;paths.append(path)
    def canonical(arm,analyses):
        calls.append(arm);assert set(analyses)==set(module.read(locations[arm])['source']['analysis_sha256'])
        return locations[arm]
    monkeypatch.setattr(module.controlled_summary,'summarize',canonical)
    frozen.update(paths=paths,locations=locations,records=records,calls=calls);return frozen


def test_three_setting_report_preserves_validation_choice_and_paired_seed_evidence(reports):
    path=module.analyze(reports['paths']);result=module.read(path)
    assert result['validation_choice']['arm']=='weight_decay_0p001'
    assert len(result['source']['runtime_qualification_sha256'])==24 and len(result['group_tables_sha256'])==48
    change=result['decay_minus_reference']['weight_decay_0p001']['best_loss']['test']['B']['43']
    assert change['mean_change']==pytest.approx(.05) and change['unit_bootstrap']['units']==2
    assert change['reference']['learning_rate_multiplier']==1. and change['alternative']['learning_rate_multiplier']==3.
    spread=result['paired_change_seed_spread']['weight_decay_0p1']['best_macro']['test']
    assert spread['B']['single_confirmation']['mean_change']['mean']==pytest.approx(-.03)
    assert spread['B']['single_confirmation']['mean_change']['total']==1
    assert spread['B']['single_confirmation']['mean_change']['sample_sd'] is None
    assert not spread['B']['confirmation_seed_variance_available']
    assert spread['A_minus_B']['both_seeds']['mean_change']['mean']==pytest.approx(0.)
    assert not result['test_used_for_selection'] and not result['study_complete'] and result['new_fits_created']==0
    assert module.analyze(reports['paths'])==path and len(reports['calls'])==6
    changed=copy.deepcopy(result);changed['validation_choice']['arm']='full';write_json(path,changed)
    with pytest.raises(ValueError,match='Completed regularization report changed'):module.analyze(reports['paths'])


@pytest.mark.parametrize('damage',['missing_summary','readout','selected_lr','confirmation_runtime','analysis_hash','group_table'])
def test_incomplete_or_changed_primary_reports_are_rejected(reports,damage):
    paths=list(reports['paths'])
    if damage=='missing_summary':paths.pop()
    elif damage in ('readout','selected_lr'):
        r=module.read(paths[-1]);r['summary']['ranking_readout' if damage=='readout' else 'selected_learning_rate_multiplier']='cosine' if damage=='readout' else 1.;write_json(paths[-1],r)
    elif damage=='confirmation_runtime':
        identity=module.control_plan.job_id('full','B',43,1.);record=module.read(reports['records'][identity])
        path=Path(record['job']['output_dir'])/'runtime_qualification.json';r=module.read(path);r['world_size']=2;write_json(path,r)
    elif damage=='analysis_hash':next(iter(reports['records'].values())).write_text('{}')
    else:
        result=module.read(module.analyze(paths));Path(next(iter(result['group_tables_sha256']))).write_bytes(b'changed')
    with pytest.raises(ValueError):module.analyze(paths)


def test_upstream_native_device_gate_is_not_bypassed(reports,monkeypatch):
    def reject(*args):raise ValueError('Native device provenance is not scientific')
    monkeypatch.setattr(module.controlled_summary,'summarize',reject)
    with pytest.raises(ValueError,match='device provenance'):module.analyze(reports['paths'])
    assert not list((reports['work']/'regularization_analysis_two_seed').glob('*/complete.json'))


def test_preflight_creates_no_selection_or_scientific_result(frozen):
    before={str(p):sha256(p) for p in frozen['work'].rglob('*') if p.is_file()}
    result=module.preflight()
    assert len(result['tuning_jobs'])==18 and all(not v for v in result['summary_candidates'].values())
    assert not result['selection_created'] and not result['scientific_comparison_created'] and result['new_fits_created']==0
    assert {str(p):sha256(p) for p in frozen['work'].rglob('*') if p.is_file()}==before



def test_reporting_does_not_accept_unverified_user_seed_policy(frozen,monkeypatch):
    def fail():raise ValueError('Two-seed policy changed')
    monkeypatch.setattr(module.two_seed_policy,'verify',fail)
    with pytest.raises(ValueError,match='policy changed'):module.prepare_selection()
    assert not module.selection_path().exists()


def test_report_selection_binds_current_seed_policy(frozen):
    result=module.build_selection()
    assert result['seed_policy_sha256']==sha256(module.two_seed_policy.destination()/'policy.json')
    assert result['accounting']==dict(settings=3,unique_tuning_fits=18,existing_confirmation_fits=6,shared_full_reference_fits=8,total_primary_fits=24)
