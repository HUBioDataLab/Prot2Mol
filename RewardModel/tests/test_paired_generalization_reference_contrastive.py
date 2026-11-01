from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import TrainingArguments

from experiments.paired_generalization import reference_contrastive as audit
from experiments.paired_generalization.qualify_controls import make_model
from reward_model.model import LoadedEncoder, load_reward_model, save_reward_model
from reward_model.training.data import RewardAssayListCollator, RewardEvaluationDataset
from reward_model.training.trainer import RewardModelTrainer
from test_paired_generalization_checkpoint_evaluation import examples_fixture, config_fixture


def test_saved_checkpoint_accepts_only_encoder_resolved_widths_without_changing_config_or_weights(tmp_path):
    model = make_model(None,{})
    launch = replace(model.config, protein_hidden_size=None, molecule_hidden_size=None)
    before = launch.to_dict()
    save_reward_model(model, str(tmp_path))
    restored = load_reward_model(str(tmp_path), device=torch.device('cpu'),
        protein_bundle=LoadedEncoder('tiny/protein', None, model.protein_encoder, 12),
        molecule_bundle=LoadedEncoder('tiny/molecule', None, model.molecule_encoder, 16))
    # Exercise real JSON serialization and the model loader's canonical config.
    assert isinstance(launch.ranking_head_hidden_dims, tuple)
    assert json.loads((tmp_path/'config.json').read_text())['ranking_head_hidden_dims'] == [8]
    assert restored.config.ranking_head_hidden_dims == launch.ranking_head_hidden_dims
    weights = audit.parameter_digest(restored)
    audit.verify_loaded_config(restored, launch)
    assert launch.to_dict() == before
    assert audit.parameter_digest(restored) == weights
    audit.verify_loaded_config(restored, replace(launch, protein_hidden_size=12, molecule_hidden_size=16))


@pytest.mark.parametrize('change', [
    {'protein_hidden_size':15}, {'molecule_hidden_size':20},
    {'dropout':.05}, {'contrastive_loss_weight':.5},
    {'ranking_head_hidden_dims':[9]}, {'molecule_model_name_or_path':'other/model'},
])
def test_loaded_config_rejects_explicit_width_architecture_or_objective_changes(change):
    model = make_model(None,{})
    launch = replace(model.config, protein_hidden_size=None, molecule_hidden_size=None)
    with pytest.raises(ValueError, match='differs from its launch'):
        audit.verify_loaded_config(model, replace(launch, **change))


def test_saved_width_must_agree_with_the_actual_loaded_encoder():
    model = make_model(None,{})
    launch = replace(model.config, protein_hidden_size=None, molecule_hidden_size=None)
    model.config.protein_hidden_size = 15
    with pytest.raises(ValueError, match='loaded encoder'):
        audit.verify_loaded_config(model, launch)


def snapshot(tmp_path, prefix='eval'):
    model = make_model(None,{})
    model.protein_tokenizer = model.molecule_tokenizer = SimpleNamespace(pad_token_id=1)
    examples = examples_fixture(model)
    config = config_fixture(model,tmp_path)
    config = replace(config,training=replace(config.training,seed=42,per_device_eval_batch_size=32))
    args = TrainingArguments(output_dir=str(tmp_path/'native'),use_cpu=True,report_to=[],seed=42,
        per_device_train_batch_size=12,per_device_eval_batch_size=32,remove_unused_columns=False)
    args.reward_export_evaluation_predictions = True
    trainer = RewardModelTrainer(model=model,args=args,
        eval_dataset=RewardEvaluationDataset(examples,protein_shuffle_sensitivity=False),
        data_collator=RewardAssayListCollator(protein_pad_token_id=1,molecule_pad_token_id=1))
    trainer.state.global_step,trainer.state.epoch = 10,1.
    trainer.evaluate(metric_key_prefix=prefix)
    path = tmp_path/'native/prediction_exports/trajectory'/f'{prefix}_step00000010.json'
    return model,examples,config,path


@pytest.mark.parametrize('prefix',['eval','train_eval'])
@pytest.mark.parametrize('unresolved_widths',[False,True])
def test_actual_model_export_reproduces_native_loss_without_training(tmp_path,monkeypatch,prefix,unresolved_widths):
    model,examples,config,native = snapshot(tmp_path,prefix)
    if unresolved_widths:
        config = replace(config,model=replace(config.model,protein_hidden_size=None,molecule_hidden_size=None))
    original_config = config.model.to_dict()
    def refuse(*args,**kwargs):raise AssertionError('No training or optimizer is allowed')
    monkeypatch.setattr(RewardModelTrainer,'train',refuse)
    monkeypatch.setattr(RewardModelTrainer,'create_optimizer',refuse)
    weights = {name:value.clone() for name,value in model.state_dict().items()}
    result = audit.evaluate_checkpoint(model,examples,config,native,tmp_path/'audit',{'fixture':True},cpu=True)
    assert config.model.to_dict() == original_config
    assert result['parameters_unchanged'] and result['cpu_qualification']
    assert abs(result['analysis']['export_minus_native_contrastive_loss']) < 1e-5
    assert result['analysis']['export_macro_spearman'] == pytest.approx(result['analysis']['native_macro_spearman'])
    for losses in result['analysis']['losses'].values():
        assert losses['total'] == pytest.approx(losses['protein_to_molecule']+losses['molecule_to_protein'],abs=1e-6)
    for name,value in model.state_dict().items():torch.testing.assert_close(value,weights[name],rtol=0,atol=0)
    assert audit.verify_export(tmp_path/'audit',{'fixture':True},cpu=True)==result
    with pytest.raises(ValueError,match='execution mode'):
        audit.verify_export(tmp_path/'audit',{'fixture':True})
    with pytest.raises(RuntimeError,match='Existing embedding'):
        audit.evaluate_checkpoint(model,examples,config,native,tmp_path/'audit',{'fixture':True},cpu=True)
    # Corruption is refused instead of being accepted as an already finished job.
    (tmp_path/'audit/protein.npy').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):
        audit.verify_export(tmp_path/'audit',{'fixture':True},cpu=True)


@pytest.mark.parametrize('change',[{'protein_hidden_size':15},{'dropout':.05}])
def test_export_rejects_config_changes_before_inference(tmp_path,monkeypatch,change):
    model,examples,config,native = snapshot(tmp_path)
    config = replace(config,model=replace(config.model,**change))
    def refuse(*args,**kwargs):raise AssertionError('Inference must not start')
    monkeypatch.setattr(audit,'score_dataset',refuse)
    with pytest.raises(ValueError,match='differs from its launch'):
        audit.evaluate_checkpoint(model,examples,config,native,tmp_path/'changed',{},cpu=True)
    assert not (tmp_path/'changed').exists()


def test_test_partition_and_changed_probe_are_refused_before_inference(tmp_path,monkeypatch):
    model,examples,config,native = snapshot(tmp_path)
    def refuse(*args,**kwargs):raise AssertionError('Inference must not start')
    monkeypatch.setattr(audit,'score_dataset',refuse)
    with pytest.raises(ValueError,match='dataset differs'):
        audit.evaluate_checkpoint(model,examples.select([0,1,2]),config,native,tmp_path/'changed',{},cpu=True)
    record = json.loads(native.read_text());record['metric_prefix']='test'
    native.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='only native validation'):
        audit.evaluate_checkpoint(model,examples,config,native,tmp_path/'test',{},cpu=True)
    # Also refuse test manifests before reading their prediction payloads.
    monkeypatch.setattr(audit,'read_native_predictions',refuse)
    with pytest.raises(ValueError,match='partition or dataset'):
        audit.analyze_export(tmp_path/'absent',native,config,examples._fingerprint)


@pytest.mark.parametrize('change',[
    {'contrastive_active_threshold':5.}, {'contrastive_mask_policy':'training_known'},
    {'cosine_scale_init':10.}, {'contrastive_strict_active_only':False}])
def test_unsupported_objective_cannot_silently_reuse_reference_direction_policy(tmp_path,change):
    model=make_model(None,change)
    config=config_fixture(model,tmp_path)
    config=replace(config,training=replace(config.training,seed=42))
    with pytest.raises(ValueError,match='fixed matched A reference'):
        audit.check_policy(config)


def test_material_native_loss_drift_and_misaligned_rows_are_rejected(tmp_path):
    model,examples,config,native = snapshot(tmp_path)
    audit.evaluate_checkpoint(model,examples,config,native,tmp_path/'audit',{},cpu=True)
    original=native.read_text();record=json.loads(original)
    record['metrics']['eval_contrastive_loss']+=.1
    native.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='differs materially'):
        audit.analyze_export(tmp_path/'audit',native,config,examples._fingerprint)
    native.write_text(original)
    import pandas as pd
    frame=pd.read_parquet(tmp_path/'audit/predictions.parquet')
    frame.iloc[::-1].to_parquet(tmp_path/'audit/predictions.parquet',index=False)
    with pytest.raises(AssertionError):
        audit.analyze_export(tmp_path/'audit',native,config,examples._fingerprint)


def test_incomplete_training_blocks_planning_before_checkpoint_access(tmp_path,monkeypatch):
    (tmp_path/'plan.json').write_text('{}')
    def incomplete(*args):raise RuntimeError('Training/retention is not complete')
    def no_protection_read(*args):raise AssertionError('Must check training completion first')
    monkeypatch.setattr(audit,'registered_job',incomplete)
    monkeypatch.setattr(audit,'verify_complete',no_protection_read)
    with pytest.raises(RuntimeError,match='Training/retention'):
        audit.build_plan(tmp_path)


def test_complete_plan_requires_all_fifteen_intact_protected_epochs(tmp_path,monkeypatch):
    """Storage/provenance fixture; it does not claim real scientific training."""
    import yaml
    from experiments.paired_generalization.protect_trajectory import hold_retention, capture_checkpoint, release_retention
    from experiments.paired_generalization.run import write_json, sha256
    model = make_model(None,{})
    config = config_fixture(model,tmp_path)
    config = replace(config,training=replace(config.training,seed=42,num_train_epochs=15,
        fixed_train_eval_assays=2048,bf16=True,fp16=False,dynamic_padding=True))
    config_path=tmp_path/'config.yaml'
    config_path.write_text(yaml.safe_dump(dict(model=config.model.to_dict(),data=asdict(config.data),training=asdict(config.training))))
    run=tmp_path/'runs'/audit.REFERENCE;run.mkdir(parents=True)
    metadata=dict(id=audit.REFERENCE,regime='A',arm='full',seed=42,learning_rate_multiplier=1.,
        config_path=str(config_path),config_sha256=sha256(config_path),output_dir=str(run))
    write_json(tmp_path/'plan.json',{})
    write_json(run/'controlled_run.json',dict(format_version=1,allow_checkpoint_retention=True,**metadata))
    write_json(run/'controlled_training_complete.json',{'synthetic_fixture':True})
    monkeypatch.setattr(audit,'registered_job',lambda *args:metadata)
    protected=tmp_path/'protected_trajectories'/audit.REFERENCE
    hold_retention(run,protected,metadata['config_sha256'],15)
    for epoch in range(1,16):
        checkpoint=run/f'checkpoint-{epoch*10}';checkpoint.mkdir()
        torch.save(model.state_dict(),checkpoint/'pytorch_model.bin')
        write_json(checkpoint/'config.json',model.config.to_dict())
        write_json(checkpoint/'trainer_state.json',dict(epoch=epoch,global_step=epoch*10))
        capture_checkpoint(run,protected,checkpoint,15)
        for prefix in ('eval','train_eval'):
            write_json(run/'prediction_exports/trajectory'/f'{prefix}_step{epoch*10:08d}.json',
                dict(phase='trajectory',epoch=epoch,step=epoch*10,metric_prefix=prefix,
                     ranking_readout='ranking_head',dataset_fingerprint='fixture'))
    release_retention(protected)
    result=audit.build_plan(tmp_path)
    assert len(result['checkpoints'])==15
    assert [x['epoch'] for x in result['checkpoints']]==list(range(1,16))
    assert all(set(x['native'])=={'eval','train_eval'} for x in result['checkpoints'])
    assert not result['test_data_used']
    (protected/'epoch07/pytorch_model.bin').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='payload changed'):
        audit.build_plan(tmp_path)
