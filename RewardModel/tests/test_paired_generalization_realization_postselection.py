"""Actual tiny-model validation/averaging/test flow, with synthetic frozen choices."""
from pathlib import Path

import pytest

from experiments.paired_generalization import realization_postselection as module
from experiments.paired_generalization.run import sha256,write_json
from test_paired_generalization_realization_losses import native,views


@pytest.fixture
def ready(native,monkeypatch):
    root=native['views']['root']
    monkeypatch.setattr(module,'WORK',root)
    path=root/'realization_losses/fixture/plan.json'
    write_json(path,native['plan'])
    for index in range(2):module.realization_losses.evaluate_worker(path,index,cpu_qualification=True)
    module.realization_losses.create_averages(path,cpu_qualification=True)
    native['path']=path
    return native


def evaluation_plan(ready):
    plan=module.build_plan('fixture',2,cpu_qualification=True)
    path=ready['views']['root']/'realization_postselection/fixture/plan.json';write_json(path,plan)
    return path,plan


def test_plan_reuses_epoch_validation_and_never_reads_test_metrics_for_choices(ready):
    before={str(p):sha256(p) for p in ready['path'].parent.rglob('*') if p.is_file()}
    path,plan=evaluation_plan(ready)
    assert len(plan['checkpoints'])==3 and len(plan['validation_references'])==2
    assert len(plan['tasks'])==4
    assert all(t['split']=='test' for t in plan['tasks'] if t['checkpoint'].startswith('checkpoint-'))
    assert {t['split'] for t in plan['tasks'] if t['checkpoint'].startswith('average_')}=={'val','test'}
    assert plan['rules']['equal_size_seed43']['best_macro']['checkpoint']=='checkpoint-30'
    assert plan['rules']['equal_size_seed43']['best_loss']['checkpoint']=='checkpoint-10'
    write_json(path.parent/'unrelated_test_metrics.json',dict(best_checkpoint='checkpoint-150',test_macro_spearman=999.))
    assert module.build_plan('fixture',2,cpu_qualification=True)==plan
    assert {str(p):sha256(p) for p in ready['path'].parent.rglob('*') if p.is_file()}==before
    with pytest.raises(ValueError,match='provenance'):module.build_plan('fixture',2)


def test_selected_epochs_and_average_evaluate_exact_roles_without_training_or_upstream_mutation(ready):
    path,plan=evaluation_plan(ready)
    upstream={str(p):sha256(p) for p in ready['path'].parent.rglob('*') if p.is_file()}
    old_loads=len(ready['loads'])
    module.evaluate_worker(path,0,cpu_qualification=True)
    with pytest.raises(FileNotFoundError):module.verify_complete(path,cpu_qualification=True)
    assert not (path.parent/'complete.json').exists()
    module.evaluate_worker(path,1,cpu_qualification=True)
    result=module.verify_complete(path,cpu_qualification=True)
    assert len(result['evidence'])==6 and len(result['costs'])==4
    assert sum(r['reused_validation'] for r in result['evidence'].values())==2
    assert result['cpu_qualification'] and not result['test_used_for_selection'] and not result['study_complete']
    assert len(ready['loads'])==old_loads+4
    output=path.parent/'checkpoint-30/equal_size_seed43/test'
    complete=module.read(output/'complete.json')
    frame,_=module.checkpoint_evaluation.read_native_predictions(complete['prediction_manifest'])
    expected=ready['views']['rows'].iloc[ready['views']['indices']['equal_size_seed43_test']]
    assert frame.compound_id.tolist()==expected.compound_id.tolist()
    val_ids=set(ready['views']['rows'].iloc[ready['views']['indices']['equal_size_seed43_val']].compound_id)
    assert val_ids.isdisjoint(frame.compound_id)
    if ready['known']:assert complete['known_positive_index']['source_rows']==3
    else:assert complete['known_positive_index'] is None
    assert {str(p):sha256(p) for p in ready['path'].parent.rglob('*') if p.is_file()}==upstream
    module.evaluate_worker(path,0,cpu_qualification=True)
    assert len(ready['loads'])==old_loads+4
    assert module.verify_complete(path,cpu_qualification=True)==result
    with pytest.raises(ValueError):module.verify_complete(path)
    Path(complete['prediction_manifest']).with_suffix('.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='payload'):module.verify_complete(path,cpu_qualification=True)


@pytest.mark.parametrize('damage',['missing_average_stage','average_weights','selected_weights','validation_payload','window'])
def test_incomplete_or_changed_selection_evidence_prevents_test_plan(ready,damage):
    parent=ready['path'].parent
    if damage=='missing_average_stage':(parent/'averages_complete.json').unlink()
    elif damage=='average_weights':(parent/'averages/epochs01_15/pytorch_model.bin').write_bytes(b'changed')
    elif damage=='selected_weights':
        (Path(ready['plan']['checkpoints'][0]['path'])/'pytorch_model.bin').write_bytes(b'changed')
    elif damage=='validation_payload':
        complete=module.read(parent/'checkpoint-10/equal_size_seed43/val/complete.json')
        Path(complete['prediction_manifest']).with_suffix('.npz').write_bytes(b'changed')
    elif damage=='window':
        p=parent/'averages_complete.json';r=module.read(p);r['allocations']['equal_size_seed43']='epochs02_15';write_json(p,r)
    with pytest.raises((ValueError,FileNotFoundError)):module.build_plan('fixture',2,cpu_qualification=True)
    assert not (ready['views']['root']/'realization_postselection').exists()


def test_wrong_device_or_partial_output_cannot_start_or_repeat_inference(ready,monkeypatch):
    path,plan=evaluation_plan(ready)
    old_loads=len(ready['loads'])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    # Preserve the CPU provenance check: cannot relabel a CPU plan as GPU science.
    with pytest.raises(ValueError,match='qualification mode'):module.evaluate_worker(path,0)
    with pytest.raises(RuntimeError,match='CPU-only'):module.evaluate_worker(path,0,cpu_qualification=True)
    changed=dict(plan,cpu_qualification=False);write_json(path,changed)
    with pytest.raises(RuntimeError,match='authorized GPU'):module.evaluate_worker(path,0)
    write_json(path,plan)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    task=plan['tasks'][0]
    (path.parent/task['checkpoint']/task['allocation']/task['split']).mkdir(parents=True)
    with pytest.raises(RuntimeError,match='Partial'):module.evaluate_worker(path,0,cpu_qualification=True)
    assert len(ready['loads'])==old_loads


@pytest.mark.parametrize('condition',['busy','space','lock'])
def test_runner_preserves_existing_training_and_waits_for_resources(tmp_path,monkeypatch,condition):
    import fcntl
    monkeypatch.setattr(module,'WORK',tmp_path);(tmp_path/'controlled').mkdir()
    monkeypatch.setattr(module.control_queue,'runtime_guard',lambda:None)
    monkeypatch.setattr(module.control_queue,'gpu_processes',lambda:[dict(pid=1)] if condition=='busy' else [])
    monkeypatch.setattr(module.control_queue,'storage_gate',lambda:dict(ready=condition!='space'))
    def forbidden(*args,**kwargs):raise AssertionError('Must not prepare or launch')
    monkeypatch.setattr(module,'prepare',forbidden)
    with (tmp_path/'controlled/driver.lock').open('a') as lock:
        if condition=='lock':fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=module.run('fixture')
    assert result['phase']==dict(busy='waiting_for_authorized_gpus',space='waiting_for_storage',lock='waiting_for_global_driver_lock')[condition]
