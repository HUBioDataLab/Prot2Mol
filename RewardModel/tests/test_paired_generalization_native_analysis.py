from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from transformers import TrainingArguments

from experiments.paired_generalization import native_analysis as audit
from experiments.paired_generalization.qualify_controls import make_model
from experiments.paired_generalization.run import sha256
from experiments.paired_generalization.control_plan import token_cache_hashes
from experiments.paired_generalization.similarity_composition import descriptors, composition_weights
from reward_model.training.data import RewardAssayListCollator, RewardEvaluationDataset
from reward_model.training.trainer import RewardModelTrainer
from test_paired_generalization_checkpoint_evaluation import examples_fixture, config_fixture


def snapshot(tmp_path,prefix='eval'):
    model = make_model(None,{})
    examples = examples_fixture(model)
    args = TrainingArguments(output_dir=str(tmp_path/'native'),use_cpu=True,report_to=[],
        per_device_train_batch_size=12,per_device_eval_batch_size=3,remove_unused_columns=False)
    args.reward_export_evaluation_predictions = True
    trainer = RewardModelTrainer(model=model,args=args,
        eval_dataset=RewardEvaluationDataset(examples,protein_shuffle_sensitivity=False),
        data_collator=RewardAssayListCollator())
    trainer.state.global_step,trainer.state.epoch = 10,1.
    trainer.evaluate(metric_key_prefix=prefix)
    path = tmp_path/'native/prediction_exports/trajectory'/f'{prefix}_step00000010.json'
    frame = examples.select_columns(audit.IDENTITY+audit.LABELS+['assay_id']).to_pandas()
    return path,frame,examples,model


@pytest.mark.parametrize('prefix',['eval','train_eval'])
def test_actual_native_snapshot_audit_matches_independent_metrics_and_calibrates_only_validation(tmp_path,prefix):
    path,frame,examples,_ = snapshot(tmp_path,prefix)
    unit_map = dict(a0=0,a1=1,a2=2)
    before = {p.name:sha256(p) for p in path.parent.iterdir()}
    result = audit.audit_snapshot(path,frame,examples._fingerprint,unit_map,tmp_path/'audit',
        prefix=prefix,noise_rms=.7,provenance={'fixture':True},regime='A')
    predictions,_ = audit.native_frame(path,frame,examples._fingerprint,prefix)
    for scorer in audit.SCORERS:
        correlations = [spearmanr(g.pchembl_value,g[scorer]).statistic for _,g in predictions.groupby('group_id')]
        assert result['metrics'][scorer]['original_eligible_macro_spearman'] == pytest.approx(np.nanmean(correlations))
        assert result['metrics'][scorer]['eligible_groups'] == 3
        assert result['metrics'][scorer]['unit_bootstrap_original_eligible']['units'] == 3
        assert set(result['metrics'][scorer]['matched_eligible_ndcg']) == {'1','3','5'}
    assert result['native_metrics'][prefix+'_macro_spearman'] == result['metrics']['ranking_score']['original_eligible_macro_spearman']
    assert result['unit_definition'] == 'assay'
    assert bool(result['calibration']) == (prefix=='eval')
    if prefix == 'eval':
        assert result['calibration']['validation_fit_logloss_after'] <= result['calibration']['validation_logloss_before']+1e-10
    else:
        assert 'fixed eligible-assay probe' in result['probe_limit']
        assert 'descriptive resampling of the fixed training probe' in result['metrics']['ranking_score']['unit_bootstrap_original_eligible']['interpretation']
    assert len(result['noise_bands']) == 6
    assert {p.name:sha256(p) for p in path.parent.iterdir()} == before
    assert audit.audit_snapshot(path,frame,examples._fingerprint,unit_map,tmp_path/'audit',
        prefix=prefix,noise_rms=.7,provenance={'fixture':True},regime='A') == result
    (tmp_path/'audit/groups_ranking_score.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):
        audit.audit_snapshot(path,frame,examples._fingerprint,unit_map,tmp_path/'audit',
            prefix=prefix,noise_rms=.7,provenance={'fixture':True},regime='A')


def test_native_snapshot_is_aligned_to_scaffold_design_without_assuming_same_row_order(tmp_path):
    path,frame,examples,_ = snapshot(tmp_path)
    scaffolds = pd.Series(['S']*3+['U']*6,index=frame.compound_id)
    unit_map = pd.Series(dict(a0=0,a1=1,a2=2))
    # Deliberately permute the design rows, as raw parquet/cache order can differ.
    raw = audit.trainer_label_frame(frame.iloc[::-1].reset_index(drop=True))
    rows,groups,_ = descriptors(raw,['m0'],scaffolds,unit_map)
    cells = composition_weights(groups,groups)
    result = audit.audit_snapshot(path,frame,examples._fingerprint,unit_map.to_dict(),tmp_path/'audit',
        prefix='eval',noise_rms=.7,provenance={},regime='A',similarity_design=(rows,groups,cells))
    assert result['similarity']['whole_groups']['all_shared']['eligible_groups'] == 1
    assert result['similarity']['whole_groups']['all_unseen']['eligible_groups'] == 2
    standardized = result['composition_standardization']
    assert standardized['target_supported_mass'] == 1.
    assert standardized['scores']['standardized']['score'] == pytest.approx(result['metrics']['ranking_score']['original_eligible_macro_spearman'])


def test_wrong_identity_label_fingerprint_or_metric_is_rejected(tmp_path):
    path,frame,examples,_ = snapshot(tmp_path)
    with pytest.raises(ValueError,match='fingerprint'):
        audit.native_frame(path,frame,'different','eval')
    with pytest.raises(AssertionError):
        audit.native_frame(path,frame.iloc[::-1].reset_index(drop=True),examples._fingerprint,'eval')
    changed = frame.copy();changed.loc[0,'pchembl_value'] += .1
    with pytest.raises(AssertionError):
        audit.native_frame(path,changed,examples._fingerprint,'eval')
    record = json.loads(path.read_text())
    record['metrics']['eval_macro_spearman'] += .2
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='parity failed'):
        audit.audit_snapshot(path,frame,examples._fingerprint,dict(a0=0,a1=1,a2=2),tmp_path/'audit',
            prefix='eval',noise_rms=.7,provenance={},regime='A')
    assert not (tmp_path/'audit').exists()


def test_test_partition_is_refused_before_reading_any_prediction(tmp_path):
    with pytest.raises(ValueError,match='Only validation'):
        audit.audit_snapshot(tmp_path/'not_read',None,None,None,tmp_path/'output',
            prefix='test',noise_rms=.7,provenance={},regime='A')


def test_reference_preparation_reconstructs_exact_fixed_probe_and_checks_raw_cache_labels(tmp_path):
    model = make_model(None,{})
    examples = examples_fixture(model)
    frame = examples.select_columns(audit.SOURCE_COLUMNS).to_pandas()
    hashes,caches,paths = {},{},{}
    for split in ('train','val'):
        path = tmp_path/f'{split}.parquet'
        frame.to_parquet(path,index=False)
        cache = tmp_path/'cache'/f'{split}_examples'
        examples.save_to_disk(str(cache))
        hashes[split],caches[split],paths[split] = sha256(path),token_cache_hashes(cache),str(path)
    config = config_fixture(model,tmp_path)
    config = replace(config,data=replace(config.data,train_parquet_path=paths['train'],val_parquet_path=paths['val'],tokenized_dataset_dir=str(tmp_path/'cache')),
                     training=replace(config.training,fixed_train_eval_assays=2))
    plan = dict(source_data_sha256={'A':hashes},tokenized_data_sha256={'A':caches})
    refs,record = audit.prepare_references(config,plan,'A',tmp_path/'references')
    assert record['probe']['actual_assays'] == 2 and record['probe']['rows'] == 6
    assert len(refs['eval'][0]) == 9
    assert refs['train_eval'][0].groupby('group_id').size().tolist() == [3,3]
    assert not record['token_payloads_used'] and not record['token_payloads_rehashed']
    again,second = audit.prepare_references(config,plan,'A',tmp_path/'references')
    assert second == record
    pd.testing.assert_frame_equal(again['train_eval'][0],refs['train_eval'][0])
    changed = frame.copy();changed.loc[0,'pchembl_value'] += .000000001
    changed.to_parquet(tmp_path/'train.parquet',index=False)
    plan['source_data_sha256']['A']['train'] = sha256(tmp_path/'train.parquet')
    with pytest.raises(ValueError,match='Raw cache labels'):
        audit.prepare_references(config,plan,'A',tmp_path/'another_reference')
