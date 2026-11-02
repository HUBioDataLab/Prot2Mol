"""Loss-system scope, common-cohort interactions and report evidence integrity."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization import loss_subset_analysis as module
from experiments.paired_generalization.run import sha256,write_json
from test_paired_generalization_control_plan import ROOT,noise_fixture


def tables_fixture():
    base=pd.DataFrame(dict(group_id=list('abcdef'),component=[1,1,2,3,4,5],
        size=[3,4,3,3,1,3],original_ranking_eligible=[True,True,True,True,False,True]))
    scores=dict(R=[.1,.2,.3,.4,.8,.5],RB=[.2,.5,.4,np.nan,.8,.6],
        RC=[.0,.4,.5,.5,.9,np.nan],RBC=[.4,.6,.9,.6,.7,.7])
    return {name:base.assign(spearman=values).iloc[::-1] for name,values in scores.items()}


def test_common_cohort_effects_match_independent_whole_unit_resampling():
    result=module.paired_effects(tables_fixture())
    assert result['groups']==6 and result['eligible_groups']==5
    assert result['mutually_defined_groups']==3
    assert result['defined_groups']==dict(R=5,RB=4,RC=4,RBC=5)
    assert result['common_cohort_means']==pytest.approx(dict(R=.2,RB=1.1/3,RC=.3,RBC=1.9/3))
    # Only a/b/c have scores in every arm. Unit 1 contributes both a and b;
    # unit 2 contributes c. Units 3/4/5 remain in the sampling universe.
    differences=dict(B_given_R=[.1,.3,.1],B_given_RC=[.4,.2,.4],
        C_given_R=[-.1,.2,.2],C_given_RB=[.2,.1,.5],B_by_C_given_R=[.3,-.1,.3])
    draws=np.random.default_rng(42).integers(1,6,size=(2000,5))
    multiplicities=np.stack([(draws==unit).sum(axis=1) for unit in (1,1,2)],axis=1)
    denominator=multiplicities.sum(axis=1);defined=denominator>0
    for name,delta in differences.items():
        expected=(multiplicities[defined]@delta)/denominator[defined]
        actual=result['contrasts'][name]
        assert actual['score']==pytest.approx(np.mean(delta))
        assert actual['interval']==pytest.approx(np.quantile(expected,[.025,.975]))
        assert actual['units']==5 and actual['defined_repetitions']==int(defined.sum())<2000
        assert actual['defined_weight_fraction']==pytest.approx(3/5)
        assert actual['effective_groups']==3
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('damage',['missing_arm','extra_arm','duplicate_group','different_unit','different_size','different_eligibility','empty'])
def test_incompatible_comparison_cohorts_are_rejected(damage):
    tables=tables_fixture()
    if damage=='missing_arm':tables.pop('RC')
    elif damage=='extra_arm':tables['B']=tables['RB']
    elif damage=='empty':tables={name:frame.iloc[:0] for name,frame in tables.items()}
    else:
        column,value=dict(duplicate_group=('group_id','b'),different_unit=('component',99),
            different_size=('size',10),different_eligibility=('original_ranking_eligible',False))[damage]
        tables['RB'].loc[0,column]=value
    with pytest.raises(ValueError):module.paired_effects(tables)


def test_undefined_common_cohort_is_null_and_not_a_zero_effect():
    tables=tables_fixture();tables['RC']['spearman']=np.nan
    result=module.paired_effects(tables)
    assert result['mutually_defined_groups']==0 and result['defined_groups']['R']==5
    assert all(value is None for value in result['common_cohort_means'].values())
    assert all(v['score'] is None and v['interval'] is None and v['defined_weight_fraction']==0.
        for v in result['contrasts'].values())
    json.dumps(result,allow_nan=False)
    for frame in tables.values():frame['original_ranking_eligible']=False
    assert all(v['defined_weight_fraction'] is None for v in module.paired_effects(tables)['contrasts'].values())


def test_all_seeds_rules_and_partitions_are_required_and_spreads_stay_separate():
    effects={(rule,split,regime,seed):dict(contrasts={name:dict(score=(seed-42)*.1+(.2 if regime=='A' else 0.))
        for name in module.CONTRASTS}) for rule in module.controlled_summary.RULES
        for split in ('val','test') for regime in ('A','B') for seed in (42,43,44)}
    cell=module.summarize_effects(effects)['best_macro']['test']
    assert cell['B']['all_three']['B_by_C_given_R']['mean']==pytest.approx(.1)
    assert cell['B']['all_three']['B_by_C_given_R']['sample_sd']==pytest.approx(.1)
    assert cell['B']['confirmation_only']['B_by_C_given_R']['mean']==pytest.approx(.15)
    assert cell['A_minus_B']['all_three']['B_by_C_given_R']['mean']==pytest.approx(.2)
    assert 'interval' not in cell['A_minus_B']['all_three']['B_by_C_given_R']
    effects.pop(('best_macro','test','B',44))
    with pytest.raises(ValueError,match='every rule'):module.summarize_effects(effects)


@pytest.mark.parametrize('damage',['duplicate_arm','wrong_family','weights','readout','horizon','missing_job','job_identity'])
def test_frozen_seven_loss_grid_and_budgets_cannot_change(tmp_path,damage):
    primary=module.control_plan.build_plan(ROOT,tmp_path,noise_fixture());module.validate_grid(primary)
    job=next(j for j in primary['tuning_jobs'] if j['arm']=='loss_BC')
    if damage=='duplicate_arm':primary['arms'].append(copy.deepcopy(next(a for a in primary['arms'] if a['name']=='loss_BC')))
    elif damage=='wrong_family':next(a for a in primary['arms'] if a['name']=='loss_C')['family']='other'
    elif damage=='weights':job['config']['model']['ranking_loss_weight']=.25
    elif damage=='readout':job['config']['model']['ranking_readout']='ranking_head'
    elif damage=='horizon':job['config']['training']['num_train_epochs']=10
    elif damage=='missing_job':primary['tuning_jobs'].remove(job)
    else:job['regime']='A' if job['regime']=='B' else 'B'
    with pytest.raises(ValueError):module.validate_grid(primary)


@pytest.fixture
def reports(tmp_path,monkeypatch):
    """Synthetic IO fixture; native training/evaluation gates are not simulated science."""
    root=tmp_path/'project';work=root/'outputs';root.mkdir()
    monkeypatch.setattr(module,'ROOT',root);monkeypatch.setattr(module,'WORK',work)
    monkeypatch.setattr(module,'software',lambda:dict(synthetic_reporting_fixture=True))
    primary=work/'controlled/plan.json'
    write_json(primary,module.control_plan.build_plan(ROOT,work,noise_fixture()))
    code=root/'qualified.py';code.write_text('synthetic source')
    sources={'qualified.py':sha256(code)}
    runtime=dict(world_size=4,sources=sources,signature=module.control_plan.digest_json(dict(sources=sources,plan_sha256=sha256(primary))))
    locations={};all_records={};paths=[];tables=tables_fixture()
    for subset,arm in module.SUBSETS.items():
        lr={'RB':.3,'RC':3.}.get(subset,1.);records={}
        for identity,(regime,seed,multiplier) in module.controlled_summary.expected_jobs(arm,lr).items():
            output=work/'controlled/runs'/identity
            write_json(output/'runtime_qualification.json',runtime)
            path=work/'postselection_analysis'/identity/'synthetic/complete.json'
            record=dict(job=dict(id=identity,output_dir=str(output)),
                source=dict(software=dict(native_fixture=True),design_sha256='same_design',noise_summary_sha256='same_noise'),
                rules={rule:dict(checkpoint='selected') for rule in module.controlled_summary.RULES})
            write_json(path,record);records[str(path)]=sha256(path);all_records[identity]=path
            for split in ('val','test'):
                table=path.parent/'selected'/split/'groups_ranking_score.parquet';table.parent.mkdir(parents=True)
                tables.get(subset,tables['R']).to_parquet(table,index=False)
        selected={rule:{split:dict(ranking_score=dict(synthetic=True)) for split in ('val','test')}
            for rule in module.controlled_summary.RULES}
        path=work/'controlled_summaries'/arm/'synthetic/complete.json'
        write_json(path,dict(source=dict(analysis_sha256=records,primary_plan_sha256=sha256(primary),software=module.controlled_summary.software()),
            summary=dict(arm=arm,ranking_readout=module.READOUTS[subset],selected_learning_rate_multiplier=lr,selected=selected)))
        locations[arm]=path;paths.append(path)
    calls=[]
    def canonical_summary(arm,analyses):
        calls.append(arm)
        assert set(analyses)==set(module.read(locations[arm])['source']['analysis_sha256'])
        return locations[arm]
    # Existing controlled_summary tests exercise the actual native-device,
    # frozen-selection, checkpoint and artifact gates. This fixture isolates
    # seven-arm assembly, provenance consistency and cached-report verification.
    monkeypatch.setattr(module.controlled_summary,'summarize',canonical_summary)
    return dict(root=root,work=work,paths=paths,locations=locations,records=all_records,calls=calls,code=code)


def test_seven_arm_report_preserves_declared_readouts_lrs_and_evidence(reports):
    path=module.analyze(reports['paths']);result=module.read(path)
    assert set(result['scoring_systems'])==set(module.SUBSETS)
    assert result['scoring_systems']['BC']['readout']=='activity_logit'
    assert result['scoring_systems']['C']['readout']=='cosine'
    assert len(result['source']['runtime_qualification_sha256'])==70
    assert result['accounting']['unique_tuning_fits']==42 and result['accounting']['shared_full_reference_fits']==10
    assert len(result['group_tables_sha256'])==48
    effects=result['conditional_auxiliary_effects']['best_loss']['test']['B']['44']
    assert effects['inputs']['RB']['learning_rate_multiplier']==.3
    assert effects['inputs']['RC']['learning_rate_multiplier']==3.
    assert effects['contrasts']['B_by_C_given_R']['score']==pytest.approx(1/6)
    assert result['new_fits_created']==0 and not result['test_used_for_selection'] and not result['study_complete']
    assert module.analyze(reports['paths'])==path
    assert len(reports['calls'])==14
    changed=copy.deepcopy(result);changed['conditional_auxiliary_effects']['best_loss']['test']['B']['44']['contrasts']['B_by_C_given_R']['score']=.9
    write_json(path,changed)
    with pytest.raises(ValueError,match='Completed loss report changed'):module.analyze(reports['paths'])


@pytest.mark.parametrize('damage',['missing_arm','readout','metric_protocol','runtime','qualified_source','analysis_hash','group_table'])
def test_incomplete_or_inconsistent_evidence_cannot_enter_comparison(reports,damage):
    paths=list(reports['paths']);record_path=next(iter(reports['records'].values()))
    if damage=='missing_arm':paths.pop()
    elif damage=='readout':
        path=paths[-1];r=module.read(path);r['summary']['ranking_readout']='cosine';write_json(path,r)
    elif damage=='metric_protocol':
        r=module.read(record_path);r['source']['noise_summary_sha256']='different';write_json(record_path,r)
        path=reports['locations']['loss_R'];s=module.read(path);s['source']['analysis_sha256'][str(record_path)]=sha256(record_path);write_json(path,s)
    elif damage=='runtime':
        r=module.read(record_path);path=Path(r['job']['output_dir'])/'runtime_qualification.json';runtime=module.read(path)
        runtime['world_size']=2;write_json(path,runtime)
    elif damage=='qualified_source':reports['code'].write_text('changed source')
    elif damage=='analysis_hash':record_path.write_text('{}')
    else:
        result=module.read(module.analyze(paths));Path(next(iter(result['group_tables_sha256']))).write_bytes(b'changed')
    with pytest.raises(ValueError):module.analyze(paths)


def test_primary_native_gate_failures_are_propagated_before_any_loss_report(reports,monkeypatch):
    def reject(*args):raise ValueError('Native evaluation device provenance is not scientific')
    monkeypatch.setattr(module.controlled_summary,'summarize',reject)
    with pytest.raises(ValueError,match='device provenance'):module.analyze(reports['paths'])
    assert not (reports['work']/'loss_subset_analysis').exists()


def test_preflight_only_inventories_paths_without_verifying_or_creating_science(reports):
    reports['paths'][-1].unlink()
    before={str(p):sha256(p) for p in reports['work'].rglob('*') if p.is_file()}
    result=module.preflight()
    assert sum(len(v) for v in result['summary_candidates'].values())==6
    assert not result['scientific_comparison_created'] and result['new_fits_created']==0
    assert reports['calls']==[]
    assert {str(p):sha256(p) for p in reports['work'].rglob('*') if p.is_file()}==before
