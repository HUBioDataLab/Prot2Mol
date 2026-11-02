"""Actual tiny native predictions plus independent checks of selected diagnostics.

These fixtures qualify the analysis path; they are not scientific results.
"""
import copy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import kendalltau, spearmanr
from datasets import load_from_disk

from experiments.paired_generalization import selected_analysis as audit, postselection
from experiments.paired_generalization.checkpoint_evaluation import evaluate_fixed_model, read_native_predictions
from experiments.paired_generalization.qualify_controls import make_model
from experiments.paired_generalization.similarity_composition import descriptors, composition_weights, DESIGN
from experiments.paired_generalization.label_precision import trainer_label_frame
from experiments.paired_generalization.run import sha256, write_json
from experiments.paired_generalization.control_plan import token_cache_hashes
from test_paired_generalization_checkpoint_evaluation import examples_fixture, config_fixture
from test_paired_generalization_postselection import grid


@pytest.fixture
def native_pair(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setenv('ACCELERATE_USE_CPU','true')
    monkeypatch.setenv('WANDB_MODE','disabled')
    model = make_model(None,{})
    examples = examples_fixture(model)
    test = examples.map(lambda row:dict(assay_id='z'+row['assay_id'],
        target_chembl_id='z'+row['target_chembl_id'],compound_id='z'+row['compound_id'],
        group_id='z'+row['target_chembl_id']+'__z'+row['assay_id']))
    config = config_fixture(model,tmp_path)
    evaluations,refs = {},{}
    for split,dataset,prefix in (('val',examples,'eval'),('test',test,'test')):
        cache=tmp_path/'cache'/f'{split}_examples'
        dataset.save_to_disk(str(cache));dataset=load_from_disk(str(cache))
        evaluations[split] = evaluate_fixed_model(model,dataset,config,tmp_path/split,prefix=prefix,cpu=True)
        raw = trainer_label_frame(dataset.select_columns(audit.native_analysis.SOURCE_COLUMNS+['group_id']).to_pandas())
        source=tmp_path/f'{split}.parquet';raw.to_parquet(source,index=False)
        raw = raw.iloc[::-1].reset_index(drop=True)
        units = pd.Series({a:i+(0 if split=='val' else 3) for i,a in enumerate(sorted(raw.assay_id.unique()))})
        scaffolds = pd.Series({m:'S' if m in ('m0','m1','m2') else 'U' for m in raw.compound_id})
        scaffolds.loc['m0']='S'
        rows,groups,_ = descriptors(raw,['m0'],scaffolds,units)
        refs[split] = dict(rows=rows,groups=groups,cells=composition_weights(groups,groups),fingerprint=dataset._fingerprint,
            source=dict(source=str(source),source_sha256=sha256(source),cache=str(cache),cache_sha256=token_cache_hashes(cache)))
    return evaluations,refs


def test_native_predictions_match_independent_rank_and_pair_metrics(native_pair):
    evaluations,refs = native_pair
    result,tables = audit.diagnose_checkpoint(evaluations,refs,regime='A',noise_rms=.7,ranking_readout='ranking_head')
    for split in ('val','test'):
        frame,_ = read_native_predictions(evaluations[split]['prediction_manifest'])
        statistics = result['partitions'][split]
        for scorer in audit.SCORERS:
            rhos,taus,concordant,discordant,pairs = [],[],0,0,0
            for _,group in frame.groupby('group_id'):
                y,s = group.pchembl_value.to_numpy(),group[scorer].to_numpy()
                rhos.append(spearmanr(y,s).statistic);taus.append(kendalltau(y,s).statistic)
                for i in range(len(y)):
                    for j in range(i):
                        product=(float(y[i])-float(y[j]))*(float(s[i])-float(s[j]))
                        concordant += product>0;discordant += product<0;pairs += 1
            metrics = statistics['metrics'][scorer]
            assert metrics['original_eligible_macro_spearman'] == pytest.approx(np.nanmean(rhos))
            assert metrics['macro_kendall_tau_b'] == pytest.approx(np.nanmean(taus))
            assert metrics['pooled_within_group_concordance'] == pytest.approx((concordant-discordant)/pairs)
            assert metrics['unit_bootstrap_original_eligible']['units']==3
            assert metrics['eligible_defined_fraction']==1.
            assert statistics['A_composition_standardization'][scorer]['standardized']['score'] == pytest.approx(np.nanmean(rhos))
            assert len(tables[f'{split}/composition_{scorer}'])==3
        assert len(statistics['noise_bands'])==6
    assert result['unit_definition']=='assay'
    assert result['calibration']['validation_fit_logloss_after'] <= result['calibration']['validation_logloss_before']+1e-8
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('damage',['fingerprint','labels','unit_overlap','readout','native_metric'])
def test_wrong_partition_labels_units_readout_and_metric_are_rejected(native_pair,damage):
    evaluations,refs = native_pair
    readout = 'ranking_head'
    if damage=='fingerprint':refs['test']['fingerprint']='wrong'
    elif damage=='labels':refs['test']['rows'].loc[0,'pchembl_value']+=.2
    elif damage=='unit_overlap':refs['test']['rows']['unit_id']=0
    elif damage=='readout':readout='activity_logit'
    else:
        path=Path(evaluations['test']['prediction_manifest']);record=json.loads(path.read_text())
        record['metrics']['test_macro_spearman']+=.1;write_json(path,record)
    with pytest.raises((ValueError,AssertionError)):
        audit.diagnose_checkpoint(evaluations,refs,regime='A',noise_rms=.7,ranking_readout=readout)


def test_calibration_and_bbse_estimation_do_not_fit_test_labels():
    val=pd.DataFrame(dict(group_id=['p__a','p__a','q__b','q__b'],binary_label=[0,1,0,1],activity_logits=[-1.,1.,-2.,2.]))
    test=pd.DataFrame(dict(group_id=['r__c','r__c','s__d','s__d'],binary_label=[0,0,0,1],activity_logits=[-1.,1.,1.,2.]))
    units=dict(a=0,b=1,c=2,d=3)
    first=audit.calibration_and_shift(val,test,units)
    second=audit.calibration_and_shift(val,test.assign(binary_label=1-test.binary_label),units)
    assert first['calibration']['temperature']==second['calibration']['temperature']
    assert first['bbse']['estimated_positive_prior']==second['bbse']['estimated_positive_prior']==.75
    assert first['bbse']['unit_bootstrap_interval']==second['bbse']['unit_bootstrap_interval']
    assert first['bbse']['actual_test_positive_rate_diagnostic_only']==.25
    assert second['bbse']['actual_test_positive_rate_diagnostic_only']==.75


def test_references_match_loaded_cache_fingerprints_and_frozen_design(tmp_path,native_pair):
    _,refs=native_pair
    directory=tmp_path/'design';directory.mkdir()
    for split,ref in refs.items():
        ref['rows'].to_parquet(directory/f'rows_A_{split}.parquet',index=False)
        ref['groups'].to_parquet(directory/f'groups_A_{split}.parquet',index=False)
        ref['cells'].to_parquet(directory/f'weights_{split}.parquet',index=False)
    write_json(directory/'complete.json',dict(design=DESIGN,
        source_sha256={r['source']['source']:r['source']['source_sha256'] for r in refs.values()},
        artifacts_sha256={p.name:sha256(p) for p in directory.glob('*.parquet')}))
    plan=dict(job=dict(regime='A'),data={split:ref['source'] for split,ref in refs.items()})
    loaded=audit.references(plan,directory)
    for split in refs:
        assert loaded[split]['fingerprint']==refs[split]['fingerprint']
        pd.testing.assert_frame_equal(loaded[split]['rows'],refs[split]['rows'],check_dtype=False)
    state=Path(refs['val']['source']['cache'])/'state.json'
    state.write_text(state.read_text()+'\n')
    with pytest.raises(ValueError,match='cache metadata changed'):
        audit.references(plan,directory)


def test_completed_evaluation_gate_checks_real_receipts_and_rejects_cpu_science(grid,monkeypatch):
    from reward_model import model as model_api
    from reward_model.model import LoadedEncoder
    monkeypatch.setattr(audit,'WORK',postselection.WORK)
    path=postselection.prepare(grid[2][0]['id'])
    loader=model_api.load_reward_model
    def tiny_loader(checkpoint,**kwargs):
        model=make_model(None,{})
        return loader(checkpoint,**kwargs,
            protein_bundle=LoadedEncoder('tiny/protein',None,model.protein_encoder,12),
            molecule_bundle=LoadedEncoder('tiny/molecule',None,model.molecule_encoder,16))
    monkeypatch.setattr(model_api,'load_reward_model',tiny_loader)
    postselection.evaluate_worker(path,0,1,cpu_qualification=True)
    plan=audit.read(path)
    with pytest.raises(FileNotFoundError):audit.verify_completed_evaluations(path.parent,cpu_qualification=True)
    evidence={f"{c['id']}/{s}":sha256(path.parent/c['id']/s/'complete.json') for c in plan['checkpoints'] for s in ('val','test')}
    complete=dict(phase='fixed_weight_evaluation_complete',plan_sha256=sha256(path),evidence_sha256=evidence)
    write_json(path.parent/'complete.json',complete)
    verified,_=audit.verify_completed_evaluations(path.parent,cpu_qualification=True)
    assert verified==plan
    with pytest.raises(ValueError,match='device provenance'):
        audit.verify_completed_evaluations(path.parent)
    changed=copy.deepcopy(complete);changed['evidence_sha256'].pop(next(iter(evidence)))
    write_json(path.parent/'complete.json',changed)
    with pytest.raises(ValueError,match='omits or adds'):
        audit.verify_completed_evaluations(path.parent,cpu_qualification=True)
    write_json(path.parent/'complete.json',complete)
    target=path.parent/next(iter(evidence))/'complete.json';target.write_text('{}')
    with pytest.raises(ValueError,match='receipt changed'):
        audit.verify_completed_evaluations(path.parent,cpu_qualification=True)


def test_analysis_persists_and_checks_results_after_completion_gate(tmp_path,native_pair,monkeypatch):
    evaluations,refs=native_pair
    work=tmp_path/'work';monkeypatch.setattr(audit,'WORK',work)
    identity='full__A__seed42__lr1p0'
    directory=work/'postselection'/identity
    config_path=tmp_path/'config.yaml'
    config_path.write_text(yaml.safe_dump(json.loads(json.dumps(asdict(config_fixture(make_model(None,{}),tmp_path))))))
    plan=dict(job=dict(id=identity,regime='A',config_path=str(config_path)),
        data={split:ref['source'] for split,ref in refs.items()},
        checkpoints=[dict(id='checkpoint-1',weights_sha256='synthetic')],
        rules=dict(best_macro=dict(checkpoint='checkpoint-1')),seed_role='tuning_exposed',
        selected_learning_rate=True,selection_sha256='synthetic')
    write_json(directory/'plan.json',plan);write_json(directory/'complete.json',dict(fixture=True))
    # Receipt/selection validation is exercised separately with real completed
    # tiny-grid exports above. This test covers downstream persistence/reuse.
    monkeypatch.setattr(audit,'verify_completed_evaluations',lambda path:(plan,{'checkpoint-1':evaluations}))
    noise_path=work/'tier1/replicates/noise_summary.json'
    write_json(noise_path,dict(A=dict(train=dict(noise_scale=dict(rms_replicate_difference=.7)))))
    write_json(work/'controlled/plan.json',dict(noise_summary_sha256=sha256(noise_path),training_noise_rms=dict(A=.7)))
    design=work/'tier1/similarity_composition/design';design.mkdir(parents=True)
    for split,ref in refs.items():
        ref['rows'].to_parquet(design/f'rows_A_{split}.parquet',index=False)
        ref['groups'].to_parquet(design/f'groups_A_{split}.parquet',index=False)
        ref['cells'].to_parquet(design/f'weights_{split}.parquet',index=False)
    write_json(design/'complete.json',dict(design=DESIGN,
        source_sha256={r['source']['source']:r['source']['source_sha256'] for r in refs.values()},
        artifacts_sha256={p.name:sha256(p) for p in design.glob('*.parquet')}))
    before={s:sha256(Path(e['prediction_manifest'])) for s,e in evaluations.items()}
    result_path=audit.analyze_job(identity)
    result=audit.read(result_path)
    assert result['phase']=='selected_prediction_analysis_complete'
    assert result['rules']==plan['rules'] and set(result['checkpoints'])=={'checkpoint-1'}
    assert audit.analyze_job(identity)==result_path
    assert {s:sha256(Path(e['prediction_manifest'])) for s,e in evaluations.items()}==before
    target=result_path.parent/'checkpoint-1/test/composition_ranking_score.parquet'
    target.write_bytes(b'changed')
    with pytest.raises(ValueError,match='analysis artifact changed'):
        audit.analyze_job(identity)


def test_matched_comparison_uses_the_rule_not_equal_checkpoint_numbers(tmp_path,native_pair):
    evaluations,refs=native_pair
    _,tables=audit.diagnose_checkpoint(evaluations,refs,regime='A',noise_rms=.7,ranking_readout='ranking_head')
    design=tmp_path/'design';design.mkdir()
    for split in ('val','test'):refs[split]['cells'].to_parquet(design/f'weights_{split}.parquet',index=False)
    write_json(design/'complete.json',dict(design=DESIGN,source_sha256={},
        artifacts_sha256={p.name:sha256(p) for p in design.glob('*.parquet')}))
    paths=[]
    for regime,checkpoint,delta in [('A','checkpoint-14',0.),('B','checkpoint-6',-.1)]:
        output=tmp_path/regime;output.mkdir()
        for split in ('val','test'):
            directory=output/checkpoint/split;directory.mkdir(parents=True)
            for scorer in audit.SCORERS:
                table=tables[f'{split}/composition_{scorer}'].copy();table['spearman']+=delta
                table.to_parquet(directory/f'composition_{scorer}.parquet',index=False)
        result=dict(phase='selected_prediction_analysis_complete',
            job=dict(regime=regime,arm='full',seed=42,phase='tuning',learning_rate_multiplier=1.,plan_sha256='primary'),
            source=dict(design_sha256=sha256(design/'complete.json')),selection_sha256='selection',
            rules=dict(best_macro=dict(checkpoint=checkpoint)),
            artifacts_sha256={str(p.relative_to(output)):sha256(p) for p in output.rglob('*.parquet')})
        path=output/'complete.json';write_json(path,result);paths.append(path)
    compared=audit.compare_analyses(*paths,design)
    for scorer in audit.SCORERS:
        assert compared['comparisons']['best_macro']['test'][scorer]['standardized_common_A_minus_B']==pytest.approx(.1)
    b=audit.read(paths[1]);b['job']['seed']=43;write_json(paths[1],b)
    with pytest.raises(ValueError,match='not matched'):
        audit.compare_analyses(*paths,design)


def test_nonfinite_undefined_diagnostics_are_serialized_as_null():
    assert audit.finite_json(dict(condition=float('inf'),score=np.float64('nan'),coverage=0.))==dict(condition=None,score=None,coverage=0.)
