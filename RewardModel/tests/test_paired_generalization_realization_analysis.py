"""Allocation membership, actual native diagnostics and honest spread accounting."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from experiments.paired_generalization import realization_analysis as module
from experiments.paired_generalization.run import sha256,write_json
from test_paired_generalization_realization_postselection import ready,evaluation_plan
from test_paired_generalization_realization_losses import native,views


def make_design(views):
    root=views['root'];directory=root/'tier1/similarity_composition/design';directory.mkdir(parents=True)
    rows=views['rows'];units=rows[['assay_id','component']].drop_duplicates().set_index('assay_id').component
    scaffolds=pd.Series({**{m:'heldout_'+str(i) for i,m in enumerate(rows.compound_id)},'training_molecule':'training_scaffold'})
    for role in ('val','test'):
        raw=rows[rows.original_split==role].copy()
        desc,groups,_=module.similarity_composition.descriptors(raw,['training_molecule'],scaffolds,units)
        for regime in ('A','B'):
            desc.to_parquet(directory/f'rows_{regime}_{role}.parquet',index=False)
            groups.to_parquet(directory/f'groups_{regime}_{role}.parquet',index=False)
    write_json(directory/'complete.json',dict(design=module.similarity_composition.DESIGN,
        source_sha256={str(views['primary_path']):sha256(views['primary_path'])},
        artifacts_sha256={p.name:sha256(p) for p in directory.glob('*.parquet')}))
    return directory


def test_descriptor_views_preserve_whole_groups_labels_units_and_recompute_cells(views):
    design=make_design(views)
    refs,info=module.references(views['contract_path'],design)
    for allocation,roles in refs.items():
        for role,ref in roles.items():
            positions=views['indices'][allocation+'_'+role]
            expected=views['rows'].iloc[positions]
            assert ref['rows'].compound_id.tolist()==expected.compound_id.tolist()
            assert ref['rows'].unit_id.tolist()==expected.component.tolist()
            assert ref['groups']['size'].sum()==len(expected)
            assert not ref['rows'].max_similarity.any()
            a=pd.read_parquet(design/f'groups_A_{role}.parquet')
            pd.testing.assert_frame_equal(ref['cells'],module.similarity_composition.composition_weights(a,ref['groups']))
            assert info['coverage'][allocation][role]['eligible_groups']==2
        assert set(roles['val']['rows'].unit_id).isdisjoint(roles['test']['rows'].unit_id)


@pytest.mark.parametrize('damage',['labels','units','assay_size','duplicate_group'])
def test_inconsistent_descriptor_layout_is_rejected_even_with_updated_artifact_hash(views,damage):
    design=make_design(views)
    path=design/('rows_B_val.parquet' if damage in ('labels','units') else 'groups_B_val.parquet')
    frame=pd.read_parquet(path)
    if damage=='labels':frame.loc[0,'pchembl_value']+=.25
    elif damage=='units':frame.loc[0,'unit_id']=999
    elif damage=='assay_size':frame.loc[0,'size']+=1
    else:frame=pd.concat([frame,frame.iloc[:1]],ignore_index=True)
    frame.to_parquet(path,index=False)
    p=design/'complete.json';record=module.read(p);record['artifacts_sha256'][path.name]=sha256(path);write_json(p,record)
    with pytest.raises(ValueError):module.references(views['contract_path'],design)


def test_actual_selected_predictions_flow_into_allocated_diagnostics_and_cached_artifacts(ready,monkeypatch):
    root=ready['views']['root'];monkeypatch.setattr(module,'WORK',root)
    path,plan=evaluation_plan(ready)
    # Complete the actual tiny inference pipeline before the analysis gate.
    for index in range(2):module.realization_postselection.evaluate_worker(path,index,cpu_qualification=True)
    module.realization_postselection.verify_complete(path,cpu_qualification=True)
    make_design(ready['views'])
    noise=root/'tier1/replicates/noise_summary.json'
    write_json(noise,dict(B=dict(train=dict(noise_scale=dict(rms_replicate_difference=.7)))))
    write_json(root/'controlled/plan.json',dict(noise_summary_sha256=sha256(noise),training_noise_rms=dict(B=.7)))
    ready['plan']['selection_sha256']='fixture-allocation-selection'
    # The upstream loss plan is synthetic. Keep its existing bytes unchanged:
    # the analysis reads a selection hash, while the actual evaluation gate is
    # independently exercised above and in the upstream tests.
    original_read=module.read
    def read_with_selection(value):
        record=original_read(value)
        if Path(value)==ready['path']:record=dict(record,selection_sha256='fixture-allocation-selection')
        return record
    monkeypatch.setattr(module,'read',read_with_selection)
    ready['plan'].pop('selection_sha256')
    # Real job identity/config binding is tested upstream; this fixture adds the
    # two reporting metadata fields without altering native inference receipts.
    original_inputs=module.evaluation_inputs
    def inputs(identity,**kwargs):
        p,values=original_inputs(identity,**kwargs)
        p=copy.deepcopy(p);p['job'].update(seed=43,arm='full',regime='B',learning_rate_multiplier=1.)
        return p,values
    monkeypatch.setattr(module,'evaluation_inputs',inputs)
    before={str(p):sha256(p) for p in path.parent.rglob('*') if p.is_file()}
    result_path=module.analyze_job('fixture',cpu_qualification=True);result=module.read(result_path)
    assert result['cpu_qualification'] and result['seed_role']=='confirmation'
    statistics=result['allocations']['equal_size_seed43']['checkpoint-30']
    source=module.read(path.parent/'checkpoint-30/equal_size_seed43/test/complete.json')
    frame,_=module.realization_postselection.checkpoint_evaluation.read_native_predictions(source['prediction_manifest'])
    values=[spearmanr(g.pchembl_value,g.ranking_score).statistic for _,g in frame.groupby('group_id')]
    actual=statistics['partitions']['test']['metrics']['ranking_score']
    assert actual['original_eligible_macro_spearman']==pytest.approx(np.nanmean(values))
    assert actual['unit_bootstrap_original_eligible']['units']==2
    assert statistics['calibration']['fit_partition']=='validation only'
    assert statistics['unit_definition']=='scaffold-assay component'
    assert module.analyze_job('fixture',cpu_qualification=True)==result_path
    # Even unchanged tables/child receipts cannot legitimize edited top-level
    # scores or a missing selected-checkpoint binding in the cached report.
    saved=result_path.read_bytes()
    changed=copy.deepcopy(result)
    changed['allocations']['equal_size_seed43']['checkpoint-30']['partitions']['test']['metrics']['ranking_score']['eligible_groups']+=1
    write_json(result_path,changed)
    with pytest.raises(ValueError,match='statistics differ'):module.analyze_job('fixture',cpu_qualification=True)
    result_path.write_bytes(saved)
    changed=copy.deepcopy(result)
    changed['artifacts_sha256'].pop('equal_size_seed43/checkpoint-30/complete.json')
    with pytest.raises(ValueError,match='not hash-bound'):module.verify_statistics(result_path.parent,changed,inputs('fixture',cpu_qualification=True)[0])
    assert {str(p):sha256(p) for p in path.parent.rglob('*') if p.is_file()}==before
    json.dumps(result,allow_nan=False)
    table=next(result_path.parent.rglob('*.parquet'));table.write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):module.analyze_job('fixture',cpu_qualification=True)


def summary_fixture():
    selection=dict(arm='full',selection_sha256='selection',selections={
        'original':dict(learning_rate_multiplier=.3),'second':dict(learning_rate_multiplier=1.),'third':dict(learning_rate_multiplier=.3)})
    rules=('best_macro','best_loss','patience3_macro','patience3_loss','average_from_loss_minimum')
    records=[]
    for seed in (42,43,44):
        for lr in (.3,1.):
            allocations={};choices={}
            for allocation,item in selection['selections'].items():
                if item['learning_rate_multiplier']!=lr:continue
                value={'original':.1,'second':.3,'third':.2}[allocation]+.01*(seed-42)
                stats=dict(ranking_readout='ranking_head',partitions={'test':dict(metrics={s:dict(
                    original_eligible_macro_spearman=value,eligible_defined_fraction=1.,eligible_groups=10)
                    for s in module.selected_analysis.SCORERS})})
                allocations[allocation]={'checkpoint':stats};choices[allocation]={r:dict(checkpoint='checkpoint') for r in rules}
            records.append(dict(job=dict(id=module.control_plan.job_id('full','B',seed,lr),arm='full',regime='B',seed=seed,
                learning_rate_multiplier=lr),allocations=allocations,rules=choices,selection_sha256='selection',
                analysis_protocol_sha256='protocol',cpu_qualification=False,seed_role='tuning_exposed' if seed==42 else 'confirmation'))
    return records,selection


def test_seed_spread_and_dependent_allocation_sensitivity_are_separate_and_not_confidence_intervals():
    records,selection=summary_fixture();result=module.summarize_records(records,selection)
    cell=result['by_allocation']['original']['best_macro']['ranking_score']
    assert cell['all_three']['mean']==pytest.approx(.11)
    assert cell['all_three']['sample_sd']==pytest.approx(.01)
    assert cell['confirmation_only']['mean']==pytest.approx(.115)
    assert cell['confirmation_only']['total']==2
    sensitivity=result['within_seed_allocation_sensitivity']['42']['best_macro']['ranking_score']
    assert sensitivity['mean']==pytest.approx(.2) and sensitivity['minimum']==.1 and sensitivity['maximum']==.3
    assert 'confidence_interval' not in json.dumps(result)
    assert 'dependent' in result['interpretation'] and 'tuning-exposed' in result['interpretation']
    assert not result['test_used_for_selection']


@pytest.mark.parametrize('damage',['missing_seed','duplicate','cpu','selection','protocol','allocation','rule','readout','seed_label'])
def test_incomplete_or_incomparable_grid_cannot_produce_spread(damage):
    records,selection=summary_fixture()
    if damage=='missing_seed':records.pop()
    elif damage=='duplicate':records[-1]=records[0]
    elif damage=='cpu':records[0]['cpu_qualification']=True
    elif damage=='selection':records[0]['selection_sha256']='other'
    elif damage=='protocol':records[0]['analysis_protocol_sha256']='other'
    elif damage=='allocation':records[0]['allocations'].pop('third')
    elif damage=='rule':records[0]['rules']['original'].pop('best_loss')
    elif damage=='readout':records[0]['allocations']['original']['checkpoint']['ranking_readout']='cosine'
    elif damage=='seed_label':records[0]['seed_role']='confirmation'
    with pytest.raises(ValueError):module.summarize_records(records,selection)


def test_undefined_scores_keep_coverage_and_are_not_replaced_by_zero():
    records,selection=summary_fixture()
    for record in records:
        if 'original' in record['allocations']:
            metric=record['allocations']['original']['checkpoint']['partitions']['test']['metrics']['ranking_score']
            metric['original_eligible_macro_spearman']=None;metric['eligible_defined_fraction']=0.
    result=module.summarize_records(records,selection)
    score=result['by_allocation']['original']['best_macro']['ranking_score']
    assert score['all_three']['defined']==0 and score['all_three']['mean'] is None
    assert score['confirmation_only']['sample_sd'] is None
    assert score['seeds']['43']['defined_fraction']==0.


def test_final_summary_rechecks_current_sources_and_child_statistics(tmp_path,monkeypatch):
    records,selection=summary_fixture();monkeypatch.setattr(module,'WORK',tmp_path)
    selection_path=tmp_path/'selection/selection.json';write_json(selection_path,selection)
    monkeypatch.setattr(module.realization_selection,'destination',lambda arm:selection_path.parent)
    monkeypatch.setattr(module.realization_selection,'verify_selection',lambda arm:selection)
    code={'metric_version':'fixture'};reference={'contract_sha256':'contract'}
    monkeypatch.setattr(module,'software',lambda:code)
    monkeypatch.setattr(module,'training_noise',lambda:(.7,'noise'))
    monkeypatch.setattr(module,'references',lambda *args:(None,reference))
    plans={};paths=[]
    for record in records:
        identity=record['job']['id'];directory=tmp_path/'realization_postselection'/identity
        plan=dict(job=record['job'],rules=record['rules'],contract_path='contract',checkpoints={'checkpoint':{'epoch':1}})
        write_json(directory/'plan.json',plan);write_json(directory/'complete.json',{'phase':'fixture'})
        plans[identity]=plan
        source=dict(evaluation_plan_sha256=sha256(directory/'plan.json'),evaluation_complete_sha256=sha256(directory/'complete.json'),
            software=code,references=reference,noise_summary_sha256='noise',cpu_qualification=False)
        output=tmp_path/'realization_analysis'/identity/'fixture'
        artifacts={}
        for allocation,checkpoints in record['allocations'].items():
            for checkpoint,statistics in checkpoints.items():
                child=output/allocation/checkpoint/'complete.json'
                write_json(child,dict(source=source,statistics=statistics,allocation=allocation,
                    checkpoint=plan['checkpoints'][checkpoint],artifacts_sha256={}))
                artifacts[str(child.relative_to(output))]=sha256(child)
        record.update(source=source,analysis_protocol_sha256=module.control_plan.digest_json(code),
            selection_sha256=sha256(selection_path),artifacts_sha256=artifacts)
        path=output/'complete.json';write_json(path,record);paths.append(path)
    monkeypatch.setattr(module,'evaluation_inputs',lambda identity:(plans[identity],None))
    summary=module.read(module.summarize(paths))
    assert summary['by_allocation']['original']['best_macro']['ranking_score']['confirmation_only']['mean']==pytest.approx(.115)
    assert summary['analysis_evidence_sha256']=={str(p):sha256(p) for p in paths}
    monkeypatch.setattr(module,'references',lambda *args:(None,dict(reference,contract_sha256='changed')))
    with pytest.raises(ValueError,match='source changed'):module.summarize(paths)
    monkeypatch.setattr(module,'references',lambda *args:(None,reference))
    damaged=module.read(paths[0]);damaged['allocations']['original']['checkpoint']['partitions']['test']['metrics']['ranking_score']['original_eligible_macro_spearman']=.99
    write_json(paths[0],damaged)
    with pytest.raises(ValueError,match='statistics differ'):module.summarize(paths)
