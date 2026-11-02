"""Primary-grid accounting, paired units and summary-to-artifact integrity."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization import controlled_summary as module
from experiments.paired_generalization.run import sha256,write_json
from test_paired_generalization_postselection import grid


def records_fixture():
    decision=dict(arm='full',learning_rate_multiplier=1.,plan_sha256='primary')
    records=[]
    for identity,(regime,seed,lr) in module.expected_jobs('full',1.).items():
        selected=lr==1.
        rules={r:dict(checkpoint='early' if r in ('best_loss','patience3_loss') else 'late')
            for r in (module.RULES if selected else ('best_macro',))}
        checkpoints={}
        for checkpoint in {r['checkpoint'] for r in rules.values()}:
            value=(.4 if regime=='A' else .2)+.01*(seed-42)+.01*(lr-1)-(.02 if checkpoint=='early' else 0.)
            metric=dict(**{name:value for name in module.METRICS},matched_eligible_ndcg={str(k):value+.1 for k in (1,3,5)},
                groups=100,defined_groups=90,defined_fraction=.9,eligible_groups=80,eligible_defined_fraction=1.,
                all_pairs=300,comparable_pairs=250,unit_bootstrap_original_eligible=dict(units=40,interval=[value-.1,value+.1]))
            checkpoints[checkpoint]=dict(regime=regime,ranking_readout='ranking_head',
                partitions={split:dict(metrics={scorer:copy.deepcopy(metric) for scorer in module.selected_analysis.SCORERS}) for split in ('val','test')})
        records.append(dict(phase='selected_prediction_analysis_complete',
            job=dict(id=identity,arm='full',regime=regime,seed=seed,learning_rate_multiplier=lr,
                phase='tuning' if seed==42 else 'confirmation',plan_sha256='primary'),
            rules=rules,checkpoints=checkpoints,selected_learning_rate=selected,selection_sha256='selection',
            seed_role='tuning_exposed' if seed==42 else 'confirmation',
            source=dict(software={'metric':'v1'},design_sha256='design',noise_summary_sha256='noise')))
    return records,decision


def test_three_seed_spread_matched_splits_and_lr_sensitivity_are_separate():
    records,decision=records_fixture();result=module.summarize_records(records,decision,'selection')
    cell=result['selected']['best_macro']['test']['ranking_score']
    score=cell['B']['seed_spread']
    assert score['all_three']['original_eligible_macro_spearman']['mean']==pytest.approx(.21)
    assert score['all_three']['original_eligible_macro_spearman']['sample_sd']==pytest.approx(.01)
    assert score['confirmation_only']['original_eligible_macro_spearman']['mean']==pytest.approx(.215)
    assert score['confirmation_only']['original_eligible_macro_spearman']['total']==2
    assert cell['matched_A_minus_B']['all_three']['original_eligible_macro_spearman']['mean']==pytest.approx(.2)
    assert cell['B']['points']['42']['coverage']['eligible_groups']==80
    delta=result['rule_minus_best_macro']['best_loss']['test']['ranking_score']['B']
    assert delta['confirmation_only']['original_eligible_macro_spearman']['mean']==pytest.approx(-.02)
    assert set(result['tuning_candidates'])=={'0.3','1.0','3.0'}
    assert result['lr_sensitivity_seed42']['test']['ranking_score']['B']['original_eligible_macro_spearman']['total']==3
    assert not result['test_used_for_selection'] and not result['study_complete']
    # An extreme test value changes the report, never the frozen LR decision.
    records[0]['checkpoints']['late']['partitions']['test']['metrics']['ranking_score']['original_eligible_macro_spearman']=.99
    assert module.summarize_records(records,decision,'selection')['selected_learning_rate_multiplier']==1.


@pytest.mark.parametrize('damage',['missing','duplicate','other_lr_confirmation','phase','selection','primary',
    'seed_role','selected_flag','rules','extra_checkpoint','readout','metric_code','design','noise','regime'])
def test_incomplete_or_mismatched_records_cannot_enter_summary(damage):
    records,decision=records_fixture();record=next(r for r in records if r['job']['seed']==43 and r['job']['regime']=='A')
    if damage=='missing':records.pop()
    elif damage=='duplicate':records[-1]=records[0]
    elif damage=='other_lr_confirmation':record['job']['id']=module.control_plan.job_id('full','A',43,.3)
    elif damage=='phase':record['job']['phase']='allocation_confirmation'
    elif damage=='selection':record['selection_sha256']='other'
    elif damage=='primary':record['job']['plan_sha256']='other'
    elif damage=='seed_role':record['seed_role']='tuning_exposed'
    elif damage=='selected_flag':record['selected_learning_rate']=False
    elif damage=='rules':record['rules'].pop('best_loss')
    elif damage=='extra_checkpoint':record['checkpoints']['extra']=record['checkpoints']['late']
    elif damage=='readout':record['checkpoints']['late']['ranking_readout']='cosine'
    elif damage=='metric_code':record['source']['software']={'metric':'v2'}
    elif damage=='design':record['source']['design_sha256']='other'
    elif damage=='noise':record['source']['noise_summary_sha256']='other'
    elif damage=='regime':record['checkpoints']['late']['regime']='B'
    with pytest.raises(ValueError):module.summarize_records(records,decision,'selection')


def test_undefined_scores_preserve_coverage_and_do_not_become_zero():
    records,decision=records_fixture()
    for record in records:
        if record['job']['regime']=='B':
            for checkpoint in record['checkpoints'].values():
                for partition in checkpoint['partitions'].values():
                    metric=partition['metrics']['ranking_score']
                    metric['original_eligible_macro_spearman']=None;metric['eligible_defined_fraction']=0.
    result=module.summarize_records(records,decision,'selection')
    cell=result['selected']['best_macro']['test']['ranking_score']
    assert cell['B']['seed_spread']['all_three']['original_eligible_macro_spearman']['defined']==0
    assert cell['matched_A_minus_B']['all_three']['original_eligible_macro_spearman']['mean'] is None
    assert cell['B']['points']['44']['coverage']['eligible_defined_fraction']==0.
    json.dumps(result,allow_nan=False)


def group_pair():
    first=pd.DataFrame(dict(group_id=['a','b','c','d'],component=[1,1,2,3],size=[3,4,3,1],
        original_ranking_eligible=[True,True,True,False],spearman=[.1,.4,np.nan,np.nan]))
    second=first.assign(spearman=[.3,np.nan,.5,np.nan]).iloc[::-1]
    return first,second


def test_paired_checkpoint_change_uses_mutually_defined_assays_and_all_units():
    first,second=group_pair();result=module.paired_group_change(first,second)
    assert result['first_defined_groups']==result['second_defined_groups']==2
    assert result['mutually_defined_groups']==1 and result['eligible_groups']==3
    assert result['mean_change']==pytest.approx(.2)
    assert result['unit_bootstrap']['units']==3
    assert result['unit_bootstrap']['interval']==pytest.approx([.2,.2])
    empty=module.paired_group_change(first,second.assign(spearman=np.nan))
    assert empty['mean_change'] is None and empty['unit_bootstrap']['interval'] is None


@pytest.mark.parametrize('column',['group_id','component','size','original_ranking_eligible'])
def test_group_or_unit_changes_cannot_be_treated_as_paired(column):
    first,second=group_pair();second=second.copy()
    second.loc[0,column]={'group_id':'b','component':99,'size':12,'original_ranking_eligible':False}[column]
    with pytest.raises(ValueError):module.paired_group_change(first,second)


def persist_records(tmp_path,monkeypatch):
    records,decision=records_fixture();monkeypatch.setattr(module,'WORK',tmp_path)
    noise=tmp_path/'tier1/replicates/noise_summary.json';write_json(noise,{'fixture':True})
    primary=tmp_path/'controlled/plan.json';write_json(primary,dict(noise_summary_sha256=sha256(noise)))
    decision['plan_sha256']=sha256(primary)
    selection=tmp_path/'controlled/selections/full.json';write_json(selection,decision)
    design=tmp_path/'tier1/similarity_composition/design'
    write_json(design/'complete.json',dict(design=module.similarity_composition.DESIGN,source_sha256={},artifacts_sha256={}))
    plans={};paths=[]
    for record in records:
        record['job']['plan_sha256']=sha256(primary);record['selection_sha256']=sha256(selection)
        identity=record['job']['id'];evaluation=tmp_path/'postselection'/identity
        plan={k:record[k] for k in ('job','rules','seed_role','selected_learning_rate','selection_sha256')}
        plan['checkpoints']=[dict(id=c,weights_sha256='fixture') for c in record['checkpoints']]
        plans[identity]=plan;write_json(evaluation/'plan.json',plan);write_json(evaluation/'complete.json',{'fixture':True})
        source=dict(software=module.selected_analysis.software(),design_sha256=sha256(design/'complete.json'),
            noise_summary_sha256=sha256(noise),evaluation_plan_sha256=sha256(evaluation/'plan.json'),
            evaluation_complete_sha256=sha256(evaluation/'complete.json'))
        output=tmp_path/'postselection_analysis'/identity/'fixture'
        for metadata in plan['checkpoints']:
            checkpoint=metadata['id'];directory=output/checkpoint;statistics=record['checkpoints'][checkpoint]
            for split in ('val','test'):
                for scorer in module.selected_analysis.SCORERS:
                    value=statistics['partitions'][split]['metrics'][scorer]['original_eligible_macro_spearman']
                    table=pd.DataFrame(dict(group_id=['a','b','c'],component=[0,1,2],size=[3,3,1],
                        original_ranking_eligible=[True,True,False],spearman=[value-.01,value+.01,np.nan]))
                    path=directory/split/f'groups_{scorer}.parquet';path.parent.mkdir(parents=True,exist_ok=True);table.to_parquet(path,index=False)
            child=dict(source=source,checkpoint=metadata,statistics=statistics,
                artifacts_sha256={str(p.relative_to(directory)):sha256(p) for p in directory.rglob('*.parquet')})
            write_json(directory/'complete.json',child)
        record['source']=source
        record['artifacts_sha256']={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()}
        path=output/'complete.json';write_json(path,record);paths.append(path)
    # Upstream native-device and selection checks are exercised separately with
    # actual tiny inference below; these synthetic receipts test reporting IO.
    monkeypatch.setattr(module.selected_analysis,'verify_completed_evaluations',lambda directory:(plans[Path(directory).name],None))
    monkeypatch.setattr(module.selected_analysis,'references',lambda *args:None)
    monkeypatch.setattr(module.postselection,'verify_selection',lambda *args:(decision,selection))
    monkeypatch.setattr(module.selected_analysis,'compare_analyses',lambda *args:dict(fixture_matched=True))
    return paths,design


def test_complete_reporting_keeps_artifact_evidence_and_reuses_identical_summary(tmp_path,monkeypatch):
    paths,design=persist_records(tmp_path,monkeypatch)
    result_path=module.summarize('full',paths);result=module.read(result_path)
    assert len(result['composition_pairs'])==5
    assert len(result['paired_checkpoint_changes'])==6
    changes=result['paired_checkpoint_changes']['full__B__seed43__lr1p0']['best_loss']['test']['ranking_score']
    assert changes['mean_change']==pytest.approx(-.02)
    assert result['source']['analysis_sha256']=={str(p):sha256(p) for p in paths}
    assert module.summarize('full',paths)==result_path
    saved=result_path.read_bytes();changed=copy.deepcopy(result)
    changed['paired_checkpoint_changes']['full__B__seed43__lr1p0']['best_loss']['test']['ranking_score']['mean_change']=.8
    write_json(result_path,changed)
    with pytest.raises(ValueError,match='Completed summary changed'):module.summarize('full',paths)
    result_path.write_bytes(saved)
    path=paths[0];record=module.read(path)
    record['checkpoints']['late']['partitions']['test']['metrics']['ranking_score']['original_eligible_macro_spearman']=.99
    write_json(path,record)
    with pytest.raises(ValueError,match='statistics differ'):module.verify_analysis(path,design)


def test_changed_evaluation_receipt_blocks_summary_before_reading_diagnostics(tmp_path,monkeypatch):
    paths,design=persist_records(tmp_path,monkeypatch)
    identity=module.read(paths[0])['job']['id']
    (tmp_path/'postselection'/identity/'complete.json').write_text('{}')
    with pytest.raises(ValueError,match='source changed'):module.verify_analysis(paths[0],design)


def test_actual_cpu_native_outputs_cannot_enter_scientific_summary(grid,monkeypatch):
    from reward_model import model as model_api
    from reward_model.model import LoadedEncoder
    from experiments.paired_generalization.qualify_controls import make_model
    directory,_,metadata,_=grid
    monkeypatch.setattr(module,'WORK',directory.parent)
    monkeypatch.setattr(module.selected_analysis,'WORK',directory.parent)
    loader=model_api.load_reward_model
    def tiny_loader(checkpoint,**kwargs):
        model=make_model(None,{})
        return loader(checkpoint,**kwargs,protein_bundle=LoadedEncoder('tiny/protein',None,model.protein_encoder,12),
            molecule_bundle=LoadedEncoder('tiny/molecule',None,model.molecule_encoder,16))
    monkeypatch.setattr(model_api,'load_reward_model',tiny_loader)
    path=module.postselection.prepare(metadata[0]['id'])
    module.postselection.evaluate_worker(path,0,1,cpu_qualification=True)
    plan=module.read(path)
    evidence={f"{c['id']}/{s}":sha256(path.parent/c['id']/s/'complete.json') for c in plan['checkpoints'] for s in ('val','test')}
    write_json(path.parent/'complete.json',dict(phase='fixed_weight_evaluation_complete',plan_sha256=sha256(path),evidence_sha256=evidence))
    analysis=directory.parent/'analysis.json';write_json(analysis,dict(job=dict(id=metadata[0]['id'])))
    with pytest.raises(ValueError,match='device provenance'):module.verify_analysis(analysis,directory.parent/'unused_design')


def test_read_only_preflight_does_not_invent_selection_from_completion_markers(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'WORK',tmp_path)
    jobs=[dict(id=identity,arm='full') for identity,(_,seed,_) in module.expected_jobs('full',1.).items() if seed==42]
    write_json(tmp_path/'controlled/plan.json',dict(arms=[dict(name='full')],tuning_jobs=jobs))
    write_json(tmp_path/'controlled/runs'/jobs[0]['id']/'controlled_training_complete.json',{'marker_only':True})
    before={str(p):sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    result=module.preflight('full')
    assert sum(result['tuning_completion_markers'].values())==1
    assert not result['selection_verified'] and not result['scientific_summary_created']
    assert not result['analysis_candidates']
    assert {str(p):sha256(p) for p in tmp_path.rglob('*') if p.is_file()}==before
    with pytest.raises(ValueError,match='Unregistered'):module.preflight('not_an_arm')
