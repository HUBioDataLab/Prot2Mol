import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr
import torch
from datasets import Dataset

from experiments.paired_generalization.checkpoint_evaluation import evaluate_fixed_model, read_native_predictions
from experiments.paired_generalization.qualify_controls import make_batch, make_model
from reward_model.training.config import RewardTrainerConfig, RewardTrainingConfigBundle, RewardTrainingDataConfig
from reward_model.training.known_positives import KnownPositiveIndex
from reward_model.training.trainer import RewardModelTrainer


def examples_fixture(model):
    batch = make_batch(model,torch.device('cpu'),False)
    rows = []
    for i in range(9):
        rows.append(dict(group_id=f'p{i//3}__a{i//3}',target_chembl_id=f'p{i//3}',compound_id=f'm{i}',
            assay_id=f'a{i//3}',activity_type='IC50',pchembl_value=float(batch['pchembl_values'][i]),
            binary_label=int(batch['activity_labels'][i]),
            **{key:batch[key][i].tolist() for key in ['protein_input_ids','protein_attention_mask','molecule_input_ids','molecule_attention_mask']}))
    return Dataset.from_list(rows)


def config_fixture(model,tmp_path):
    return RewardTrainingConfigBundle(model=model.config,
        data=RewardTrainingDataConfig(train_parquet_path='unused_train',val_parquet_path='unused_val',
            test_parquet_path='unused_test',tokenized_dataset_dir='unused_cache',
            enforce_binary_label_threshold=False,evaluation_ranking_partitions=3),
        training=RewardTrainerConfig(output_dir=str(tmp_path/'unused_training'),training_mode='multi_gpu',
            per_device_train_batch_size=12,per_device_eval_batch_size=2,seed=43,
            protein_shuffle_sensitivity=False,report_to=['wandb']))


@pytest.mark.parametrize('readout',['ranking_head','activity_logit','cosine'])
def test_native_fixed_evaluation_preserves_weights_and_exports_correct_readout(tmp_path,monkeypatch,readout):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setenv('ACCELERATE_USE_CPU','true')
    monkeypatch.setenv('WANDB_MODE','disabled')
    def no_training(*args,**kwargs):raise AssertionError('Evaluation must not start training')
    monkeypatch.setattr(RewardModelTrainer,'train',no_training)
    monkeypatch.setattr(RewardModelTrainer,'create_optimizer',no_training)
    changes = dict(ranking_readout=readout)
    if readout!='ranking_head':changes['ranking_loss_weight']=0.
    if readout=='cosine':changes['classification_loss_weight']=0.
    model = make_model(None,changes)
    examples = examples_fixture(model)
    config = config_fixture(model,tmp_path)
    original = copy.deepcopy(config)
    weights = {name:value.clone() for name,value in model.state_dict().items()}
    result = evaluate_fixed_model(model,examples,config,tmp_path/'evaluation',cpu=True,source={'fixture':True})
    assert config==original
    assert result['parameters_unchanged'] and not result['optimizer_created']
    assert result['contrastive_assay_batch_size']==12
    assert result['ranking_partition_seed']==43 and result['ranking_num_partitions']==3
    for name,value in model.state_dict().items():torch.testing.assert_close(value,weights[name],rtol=0,atol=0)
    frame,manifest = read_native_predictions(result['prediction_manifest'])
    assert frame.group_id.tolist()==examples['group_id']
    assert frame.compound_id.tolist()==examples['compound_id']
    np.testing.assert_array_equal(frame.pchembl_value,np.asarray(examples['pchembl_value'],dtype=np.float32))
    correlations = [spearmanr(g.pchembl_value,g.ranking_score).statistic for _,g in frame.groupby('group_id')]
    assert result['metrics']['test_macro_spearman']==pytest.approx(np.nanmean(correlations))
    if readout=='activity_logit':np.testing.assert_array_equal(frame.ranking_score,frame.activity_logits)
    if readout=='cosine':np.testing.assert_array_equal(frame.ranking_score,frame.cosine_similarity)
    assert manifest['ranking_readout']==readout
    assert not list((tmp_path/'evaluation').glob('checkpoint-*'))
    with pytest.raises(RuntimeError,match='Existing evaluation'):
        evaluate_fixed_model(model,examples,config,tmp_path/'evaluation',cpu=True)


def test_known_positive_mask_requires_training_index_and_never_builds_one_from_test(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setenv('ACCELERATE_USE_CPU','true')
    model = make_model(None,dict(contrastive_mask_policy='training_known'))
    examples = examples_fixture(model)
    config = config_fixture(model,tmp_path)
    with pytest.raises(ValueError,match='original training-positive index'):
        evaluate_fixed_model(model,examples,config,tmp_path/'missing_index',cpu=True)
    assert not (tmp_path/'missing_index').exists()
    index = KnownPositiveIndex.from_training_dataset(examples.select([0,1,2]),active_threshold=6.)
    before = copy.deepcopy(index.provenance)
    result = evaluate_fixed_model(model,examples,config,tmp_path/'with_index',known_positive_index=index,cpu=True)
    assert result['known_positive_index']==before
    assert result['known_positive_index']['source_rows']==3
    assert result['rows']==9


def test_native_prediction_reader_detects_changed_payload(tmp_path):
    payload = tmp_path/'predictions.npz';np.savez(payload,row_index=np.arange(3))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(predictions_file=payload.name,predictions_sha256='0'*64,rows=3)))
    with pytest.raises(ValueError,match='payload differs'):
        read_native_predictions(manifest)
