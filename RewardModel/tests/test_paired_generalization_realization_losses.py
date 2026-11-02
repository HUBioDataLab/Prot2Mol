"""Whole-unit cache views and real native inference; no scientific training."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from datasets import Dataset, load_from_disk

from experiments.paired_generalization import realization_losses as module
from experiments.paired_generalization import realization_protocol, control_plan
from experiments.paired_generalization.qualify_controls import make_model
from experiments.paired_generalization.run import sha256,write_json
from test_paired_generalization_checkpoint_evaluation import examples_fixture,config_fixture
from test_paired_generalization_realization_protocol import fixture
from test_paired_generalization_realization_selection import prepared as selected_fixture

ORIGINAL_BUILD_PLAN=module.build_plan


@pytest.fixture
def views(tmp_path):
    model=make_model(None,{})
    template=examples_fixture(model)
    rows,indices,counts=realization_protocol.make_views(*fixture())
    sources={};caches={};fingerprints={}
    for role in ('val','test'):
        frame=rows[rows.original_split==role].reset_index(drop=True).copy()
        frame['activity_type']='IC50'
        for column in ('protein_input_ids','protein_attention_mask','molecule_input_ids','molecule_attention_mask'):
            frame[column]=[template[i%len(template)][column] for i in range(len(frame))]
        cache=tmp_path/'cache'/f'{role}_examples'
        Dataset.from_pandas(frame,preserve_index=False).save_to_disk(str(cache))
        source=tmp_path/f'{role}.parquet';frame.to_parquet(source,index=False)
        sources[role]=sha256(source);caches[role]=control_plan.token_cache_hashes(cache)
        fingerprints[role]=load_from_disk(str(cache))._fingerprint
    primary=dict(bases={'B':dict(data=dict(tokenized_dataset_dir=str(tmp_path/'cache'),
        val_parquet_path=str(tmp_path/'val.parquet'),test_parquet_path=str(tmp_path/'test.parquet')))},
        source_data_sha256={'B':sources},tokenized_data_sha256={'B':caches})
    primary_path=tmp_path/'primary.json';write_json(primary_path,primary)
    directory=tmp_path/'contract';directory.mkdir()
    rows.to_parquet(directory/'rows.parquet',index=False)
    np.savez_compressed(directory/'views.npz',**indices)
    write_json(directory/'audit.json',{})
    contract=dict(control_plan_path=str(primary_path),control_plan_sha256=sha256(primary_path),
        assignment_audit_path=str(directory/'audit.json'),assignment_audit_sha256=sha256(directory/'audit.json'),
        artifacts_sha256={name:sha256(directory/name) for name in ('rows.parquet','views.npz')},
        training_identical=True,realizations=counts,dataset_fingerprints=fingerprints)
    path=directory/'plan.json';write_json(path,contract)
    return dict(root=tmp_path,model=model,rows=rows,indices=indices,primary=primary,
        primary_path=primary_path,contract_path=path,contract=contract)


def test_views_preserve_tokens_labels_exact_order_and_whole_units(views):
    loaded=module.AllocationViews(views['contract_path'],views['primary_path'])
    for allocation in views['contract']['realizations']:
        selected=loaded.validation(allocation)
        positions=views['indices'][allocation+'_val']
        assert selected['compound_id']==views['rows'].iloc[positions].compound_id.tolist()
        assert selected['pchembl_value']==views['rows'].iloc[positions].pchembl_value.tolist()
        assert selected['binary_label']==views['rows'].iloc[positions].binary_label.tolist()
        for i,p in enumerate(positions):
            assert selected[i]['protein_input_ids']==loaded.union[int(p)]['protein_input_ids']
            assert selected[i]['molecule_input_ids']==loaded.union[int(p)]['molecule_input_ids']
        assert set(selected['component']).isdisjoint(set(views['rows'].iloc[views['indices'][allocation+'_test']].component))
    with pytest.raises(ValueError,match='Unknown'):loaded.validation('unknown')


@pytest.mark.parametrize('damage',['cache','fingerprint','row_order','fragment','training_support'])
def test_changed_cache_or_allocation_layout_is_rejected(views,damage):
    path=views['contract_path'];contract=module.read(path)
    if damage=='cache':
        (views['root']/'cache/val_examples/dataset_info.json').write_text('{}')
    elif damage=='fingerprint':contract['dataset_fingerprints']['val']='different'
    elif damage=='row_order':
        rows=views['rows'].copy();rows.iloc[[0,1]]=rows.iloc[[1,0]].to_numpy()
        rows.to_parquet(path.parent/'rows.parquet',index=False)
        contract['artifacts_sha256']['rows.parquet']=sha256(path.parent/'rows.parquet')
    elif damage=='fragment':
        indices={k:v.copy() for k,v in views['indices'].items()}
        a,b='original_seed42_val','original_seed42_test'
        indices[a][0],indices[b][0]=indices[b][0],indices[a][0]
        for key in (a,b):indices[key].sort()
        np.savez_compressed(path.parent/'views.npz',**indices)
        contract['artifacts_sha256']['views.npz']=sha256(path.parent/'views.npz')
    elif damage=='training_support':contract['training_identical']=False
    write_json(path,contract)
    with pytest.raises(ValueError):module.AllocationViews(path,views['primary_path'])


def test_registered_job_requires_frozen_allocation_selection_and_b_regime(selected_fixture,monkeypatch):
    monkeypatch.setattr(module,'WORK',selected_fixture['work'])
    module.realization_selection.prepare('full')
    job=next(j for j in selected_fixture['jobs'] if j['regime']=='B' and j['learning_rate_multiplier']==1.)
    metadata,_,_,allocations=module.registered_selected_job(job['id'])
    assert metadata==job and allocations==['equal_size_seed43']
    a=next(j for j in selected_fixture['jobs'] if j['regime']=='A')
    with pytest.raises(ValueError,match='selected B'):module.registered_selected_job(a['id'])
    identity=control_plan.job_id('full','B',43,1.)
    with pytest.raises(RuntimeError,match='not complete'):module.registered_selected_job(identity)


@pytest.fixture(params=[False,True],ids=['reference_mask','training_known_mask'])
def native(views,monkeypatch,request):
    import reward_model.model as model_api
    from reward_model.model import LoadedEncoder
    from reward_model.training.known_positives import KnownPositiveIndex
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setenv('ACCELERATE_USE_CPU','true')
    monkeypatch.setenv('WANDB_MODE','disabled')
    model=make_model(None,dict(contrastive_mask_policy='training_known') if request.param else {})
    config=config_fixture(model,views['root'])
    config_path=views['root']/'config.yaml'
    config_path.write_text(yaml.safe_dump(dict(model=model.config.to_dict(),data=asdict(config.data),training=asdict(config.training))))
    checkpoints=[]
    for epoch in range(1,16):
        path=views['root']/f'checkpoint-{epoch*10}';path.mkdir()
        write_json(path/'config.json',model.config.to_dict());torch.save(model.state_dict(),path/'pytorch_model.bin')
        checkpoints.append(dict(id=path.name,path=str(path),epoch=epoch,step=epoch*10,
            weights_sha256=sha256(path/'pytorch_model.bin'),model_config_sha256=sha256(path/'config.json')))
    allocation='equal_size_seed43'
    history=[dict(epoch=e,step=e*10,macro_spearman=.5 if e==3 else .1,defined_fraction=1.) for e in range(1,16)]
    uniform=views['root']/'uniform.json'
    write_json(uniform,dict(selections={allocation:dict(validation_history=history,selected=history[2])}))
    known=None
    if request.param:
        train=examples_fixture(model).select([0,1,2])
        cache=views['root']/'train_cache';train.save_to_disk(str(cache))
        source=views['root']/'train.parquet';train.to_pandas().to_parquet(source,index=False)
        known=dict(source=str(source),source_sha256=sha256(source),cache=str(cache),
            cache_sha256=control_plan.token_cache_hashes(cache),
            provenance=KnownPositiveIndex.from_training_dataset(load_from_disk(str(cache)),active_threshold=6.).provenance)
    plan=dict(job=dict(id='fixture',config_path=str(config_path)),workers=2,allocations=[allocation],
        checkpoints=checkpoints,contract_path=str(views['contract_path']),primary_plan_path=str(views['primary_path']),
        uniform_validation_path=str(uniform),uniform_validation_sha256=sha256(uniform),known_positive_source=known)
    path=views['root']/'evaluation/plan.json';write_json(path,plan)
    monkeypatch.setattr(module,'build_plan',lambda identity,workers:plan)
    original=model_api.load_reward_model;loads=[]
    def loader(checkpoint,**kwargs):
        tiny=make_model(None,{})
        loads.append(str(checkpoint))
        return original(checkpoint,**kwargs,
            protein_bundle=LoadedEncoder('tiny/protein',None,tiny.protein_encoder,12),
            molecule_bundle=LoadedEncoder('tiny/molecule',None,tiny.molecule_encoder,16))
    monkeypatch.setattr(model_api,'load_reward_model',loader)
    from reward_model.training.trainer import RewardModelTrainer
    def forbidden(*args,**kwargs):raise AssertionError('Validation must never train')
    monkeypatch.setattr(RewardModelTrainer,'train',forbidden)
    monkeypatch.setattr(RewardModelTrainer,'create_optimizer',forbidden)
    return dict(path=path,plan=plan,loads=loads,known=request.param,views=views)


def test_fifteen_native_validation_passes_preserve_macro_choice_and_weights(native):
    path=native['path'];plan=native['plan']
    before={c['path']:sha256(Path(c['path'])/'pytorch_model.bin') for c in plan['checkpoints']}
    module.evaluate_worker(path,0,cpu_qualification=True)
    with pytest.raises(FileNotFoundError):module.collect_rules(path,cpu_qualification=True)
    assert not (path.parent/'validation_complete.json').exists()
    module.evaluate_worker(path,1,cpu_qualification=True)
    result=module.collect_rules(path,cpu_qualification=True)
    assert len(native['loads'])==15
    assert not result['test_roles_evaluated'] and result['cpu_qualification']
    selected=result['selections']['equal_size_seed43']
    assert selected['rules']['best_macro']['epoch']==3
    assert selected['rules']['best_loss']['epoch']==1  # Identical fixed weights and pools.
    assert selected['averaging_epochs']==list(range(1,16))
    assert result['measured_cost']['sum_worker_pass_seconds']>0 and result['measured_cost']['evaluation_bytes']>0
    assert {c['path']:sha256(Path(c['path'])/'pytorch_model.bin') for c in plan['checkpoints']}==before
    output=path.parent/'checkpoint-10/equal_size_seed43/val'
    complete=module.read(output/'complete.json')
    frame,_=module.checkpoint_evaluation.read_native_predictions(complete['prediction_manifest'])
    expected=native['views']['rows'].iloc[native['views']['indices']['equal_size_seed43_val']]
    assert frame.compound_id.tolist()==expected.compound_id.tolist()
    assert complete['prefix']=='eval' and complete['parameters_unchanged']
    if native['known']:assert complete['known_positive_index']['source_rows']==3
    else:assert complete['known_positive_index'] is None
    module.evaluate_worker(path,0,cpu_qualification=True)
    assert len(native['loads'])==15
    averages=module.create_averages(path,cpu_qualification=True)
    assert len(averages['targets'])==1 and not averages['weights_released']
    target=averages['targets']['epochs01_15']
    assert target['epochs']==list(range(1,16)) and not target['singleton']
    tensors=torch.load(Path(target['path'])/'pytorch_model.bin',weights_only=True)
    source=torch.load(Path(plan['checkpoints'][0]['path'])/'pytorch_model.bin',weights_only=True)
    assert all(torch.equal(tensors[k],v) for k,v in source.items())
    assert module.create_averages(path,cpu_qualification=True)==averages
    (Path(target['path'])/'pytorch_model.bin').write_bytes(b'changed')
    with pytest.raises(ValueError,match='average changed'):module.create_averages(path,cpu_qualification=True)
    with pytest.raises(ValueError,match='provenance'):module.collect_rules(path)
    Path(complete['prediction_manifest']).with_suffix('.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='payload'):module.collect_rules(path,cpu_qualification=True)


def test_worker_rejects_wrong_device_and_changed_checkpoint(native,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    with pytest.raises(RuntimeError,match='authorized GPU'):module.evaluate_worker(native['path'],0)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    checkpoint=native['plan']['checkpoints'][0]
    (Path(checkpoint['path'])/'pytorch_model.bin').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Checkpoint changed'):module.evaluate_worker(native['path'],0,cpu_qualification=True)
    assert not (native['path'].parent/'checkpoint-10').exists()


@pytest.mark.parametrize('damage',[None,'retention_anchor','checkpoint_config','missing_epoch','checkpoint_path'])
def test_plan_binds_full_checkpoint_trajectory_to_selection_and_retention(native,monkeypatch,damage):
    root=native['views']['root'];existing=native['plan']
    metadata=dict(id='fixture',arm='full',regime='B',seed=42,phase='tuning',
        config_path=existing['job']['config_path'],config_sha256=sha256(Path(existing['job']['config_path'])),output_dir=str(root))
    primary=native['views']['primary']
    selection=dict(primary_plan_sha256=sha256(native['views']['primary_path']),
        contract_sha256=sha256(native['views']['contract_path']))
    monkeypatch.setattr(module,'WORK',root)
    monkeypatch.setattr(module.realization_selection,'WORK',root)
    selection_path=module.realization_selection.destination('full')/'selection.json';write_json(selection_path,selection)
    monkeypatch.setattr(module,'registered_selected_job',lambda identity:(metadata,primary,selection,['equal_size_seed43']))
    scalar=root/'scalar';monkeypatch.setattr(module.realization_predictions,'destination',lambda run:scalar)
    jobs=[dict(checkpoint=c['path'],**{k:c[k] for k in ('epoch','step','weights_sha256','model_config_sha256')})
          for c in existing['checkpoints']]
    weights=[dict(epoch=c['epoch'],step=c['step'],path=str((Path(c['path'])/'pytorch_model.bin').relative_to(root)),
        sha256=c['weights_sha256']) for c in existing['checkpoints']]
    uniform=dict(config_sha256=metadata['config_sha256'],contract_sha256=selection['contract_sha256'],source_weights=weights)
    write_json(scalar/'validation_complete.json',uniform);write_json(scalar/'jobs.json',dict(jobs=jobs))
    monkeypatch.setattr(module.realization_predictions,'verify_validation',lambda run:uniform)
    write_json(root/'retained/retention.json',dict(additional_realizations=dict(
        path=str(scalar/'validation_complete.json'),sha256=sha256(scalar/'validation_complete.json'))))
    write_json(root/'controlled_training_complete.json',dict(fixture=True))
    write_json(root/'config.json',module.read(Path(existing['checkpoints'][0]['path'])/'config.json'))
    if native['known']:
        source=existing['known_positive_source']
        primary['source_data_sha256']['B']['train']=source['source_sha256']
        primary['tokenized_data_sha256']['B']['train']=source['cache_sha256']
        write_json(root/'known_positive_index.json',source['provenance'])
    if damage=='retention_anchor':write_json(scalar/'validation_complete.json',dict(changed=True))
    elif damage=='checkpoint_config':
        p=Path(existing['checkpoints'][1]['path'])/'config.json';r=module.read(p);r['dropout']=.123;write_json(p,r)
    elif damage=='missing_epoch':
        jobs.pop();uniform['source_weights'].pop();write_json(scalar/'jobs.json',dict(jobs=jobs))
    elif damage=='checkpoint_path':
        jobs[1]['checkpoint']=str(root/'other');write_json(scalar/'jobs.json',dict(jobs=jobs))
    if damage:
        with pytest.raises(ValueError):ORIGINAL_BUILD_PLAN('fixture',2)
    else:
        plan=ORIGINAL_BUILD_PLAN('fixture',2)
        assert len(plan['checkpoints'])==15 and plan['allocations']==['equal_size_seed43']
        assert plan['uniform_validation_sha256']==sha256(scalar/'validation_complete.json')
        assert not plan['test_roles_evaluated']
        assert (plan['known_positive_source'] is not None)==native['known']


@pytest.mark.parametrize('condition',['busy','space','lock'])
def test_runner_does_not_prepare_or_launch_while_resources_are_unavailable(tmp_path,monkeypatch,condition):
    import fcntl
    monkeypatch.setattr(module,'WORK',tmp_path);(tmp_path/'controlled').mkdir()
    monkeypatch.setattr(module.control_queue,'runtime_guard',lambda:None)
    monkeypatch.setattr(module.control_queue,'gpu_processes',lambda:[dict(pid=1)] if condition=='busy' else [])
    monkeypatch.setattr(module.control_queue,'storage_gate',lambda:dict(ready=condition!='space'))
    def forbidden(*args,**kwargs):raise AssertionError('Must not prepare or start a worker')
    monkeypatch.setattr(module,'prepare',forbidden)
    with (tmp_path/'controlled/driver.lock').open('a') as lock:
        if condition=='lock':fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        result=module.run('fixture')
    assert result['phase']==dict(busy='waiting_for_authorized_gpus',space='waiting_for_storage',lock='waiting_for_global_driver_lock')[condition]
