"""Real scalar export/selection fixtures; training is simulated, never scientific."""
from pathlib import Path

import numpy as np
import pytest

from experiments.paired_generalization import realization_selection as module
from experiments.paired_generalization import control_plan, control_driver, control_queue
from experiments.paired_generalization import realization_predictions as predictions
from experiments.paired_generalization import realization_protocol
from experiments.paired_generalization.run import sha256, write_json
from test_paired_generalization_control_driver import completed_tuning_fixture
from test_paired_generalization_realization_protocol import fixture


def mark_complete(metadata):
    run = Path(metadata['output_dir'])
    evidence = ['experiment_summary.json','retained/retention.json',
                *(f'rank_{rank}_complete.json' for rank in range(4))]
    write_json(run/'controlled_training_complete.json',dict(job_id=metadata['id'],
        config_sha256=metadata['config_sha256'],plan_sha256=metadata['plan_sha256'],
        evidence_sha256={name:sha256(run/name) for name in evidence}))


def export_fixture(metadata, contract_path, rows):
    run = Path(metadata['output_dir'])
    output = predictions.destination(run)
    write_json(run/'launch_provenance.json',dict(config_sha256=metadata['config_sha256']))
    jobs = [dict(epoch=epoch,step=10*epoch,weights_sha256=f'weights-{epoch}') for epoch in range(1,16)]
    frozen = dict(run=str(run),config_sha256=metadata['config_sha256'],
        contract_path=str(contract_path),contract_sha256=sha256(contract_path),jobs=jobs,source_weights=[],
        launch_provenance_sha256=sha256(run/'launch_provenance.json'))
    write_json(output/'jobs.json',frozen)
    # Different allocation winners: original=.3, allocation43=1, allocation44=3.
    positive = {0,{.3:1,1.:2,3.:3}[metadata['learning_rate_multiplier']]}
    for job in jobs:
        for split in ('val','test'):
            frame = rows[rows.original_split==split].reset_index(drop=True).copy()
            frame['row_index'] = np.arange(len(frame))
            frame['ranking_score'] = frame.pchembl_value * np.where(frame.component.isin(positive),1.,-1.)
            location = output/f"epoch{job['epoch']:02d}_{split}"
            location.mkdir(parents=True)
            frame.to_parquet(location/'predictions.parquet',index=False)
            write_json(location/'complete.json',dict(job=dict(**job,split=split,jobs_sha256=sha256(output/'jobs.json')),
                rows=len(frame),predictions_sha256=sha256(location/'predictions.parquet')))
    result = predictions.select_validation(run)
    retention_path = run/'retained/retention.json'
    retained = module.read(retention_path)
    selected_path = output/'validation_complete.json'
    retained['additional_realizations'] = dict(path=str(selected_path),sha256=sha256(selected_path))
    write_json(retention_path,retained)
    return result


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    plan = completed_tuning_fixture(tmp_path,monkeypatch)
    directory = tmp_path/'controlled'
    for item in (module,predictions,realization_protocol):monkeypatch.setattr(item,'WORK',tmp_path)
    rows,views,counts = realization_protocol.make_views(*fixture())
    target = directory/'realizations';target.mkdir()
    rows.to_parquet(target/'rows.parquet',index=False)
    np.savez_compressed(target/'views.npz',**views)
    write_json(target/'audit.json',dict(fixture=True))
    contract = dict(control_plan_path=str(directory/'plan.json'),control_plan_sha256=sha256(directory/'plan.json'),
        assignment_audit_path=str(target/'audit.json'),assignment_audit_sha256=sha256(target/'audit.json'),
        artifacts_sha256={name:sha256(target/name) for name in ('rows.parquet','views.npz')},
        training_identical=True,realizations=counts,
        inference=dict(batch_size=32,autocast='bfloat16',scalar_exports_only=True),limitations='dependent fixture allocations')
    write_json(target/'plan.json',contract)
    jobs = []
    for job in plan['tuning_jobs'][:6]:
        metadata = module.read(directory/'jobs'/f"{job['id']}.json")
        if job['regime']=='B':export_fixture(metadata,target/'plan.json',rows)
        mark_complete(metadata);jobs.append(metadata)
    primary = control_plan.confirm('full')
    return dict(work=tmp_path,directory=directory,plan=plan,rows=rows,contract_path=target/'plan.json',
        jobs=jobs,primary=primary)


def test_independent_lr_choices_earliest_epochs_union_and_primary_unchanged(prepared):
    path = prepared['directory']/'selections/full.json';before=path.read_bytes()
    selection = module.prepare('full')
    assert {name:s['learning_rate_multiplier'] for name,s in selection['selections'].items()} == dict(
        original_seed42=.3,equal_size_seed43=1.,equal_size_seed44=3.)
    assert all(s['checkpoint']['epoch']==1 for s in selection['selections'].values())
    assert selection['winning_lr_union']==[.3,1.,3.]
    assert selection['confirmation_counts']==dict(required=6,reused_primary=2,additional=4)
    assert [r['id'] for r in selection['confirmation_requests'] if r['reuse_primary']] == [
        'full__B__seed43__lr1p0','full__B__seed44__lr1p0']
    assert module.verify_selection('full')==selection
    assert module.prepare('full')==selection
    assert path.read_bytes()==before
    assert not selection['test_metrics_computed']
    assert not list((prepared['directory']/'configs').glob('*seed43__lr0p3*'))


def test_lr_ties_and_nonwinning_test_fields_cannot_select(prepared):
    candidates=[]
    for job in prepared['jobs']:
        if job['regime']!='B':continue
        result=predictions.verify_validation(Path(job['output_dir']))
        for item in result['selections'].values():
            for row in item['validation_history']:row['macro_spearman']=.2
            item['selected']=predictions.choose_epochs(item['validation_history'])
        candidates.append(dict(job_id=job['id'],learning_rate_multiplier=job['learning_rate_multiplier'],
            test_metrics_computed=False,selections=result['selections'],test_macro_spearman=999*job['learning_rate_multiplier']))
    allocations=candidates[0]['selections']
    selected=module.select_allocations(candidates,allocations)
    assert all(r['learning_rate_multiplier']==.3 for r in selected.values())
    candidates[-1]['test_macro_spearman']=-1e9
    assert module.select_allocations(candidates,allocations)==selected
    candidates[0]['test_metrics_computed']=True
    with pytest.raises(ValueError,match='validation-only'):module.select_allocations(candidates,allocations)


@pytest.mark.parametrize('damage',['payload','retention_anchor','missing_rank','test_exposure','missing_receipt','contract','epoch_history'])
def test_changed_or_incomplete_evidence_never_freezes_selection(prepared,damage):
    job=prepared['jobs'][1];run=Path(job['output_dir']);output=predictions.destination(run)
    if damage=='payload':(output/'epoch15_test/predictions.parquet').write_bytes(b'corrupt')
    elif damage=='retention_anchor':
        p=output/'validation_complete.json';r=module.read(p);r['remaining']='changed';write_json(p,r)
    elif damage=='missing_rank':(run/'rank_3_complete.json').unlink()
    elif damage=='test_exposure':
        p=run/'experiment_summary.json';r=module.read(p);r['test_evaluated']=True;write_json(p,r)
    elif damage=='missing_receipt':(run/'controlled_training_complete.json').unlink()
    elif damage=='contract':
        p=prepared['contract_path'];r=module.read(p);r['training_identical']=False;write_json(p,r)
    elif damage=='epoch_history':
        p=output/'validation_complete.json';r=module.read(p)
        r['selections']['original_seed42']['validation_history'].pop();write_json(p,r)
    with pytest.raises((ValueError,RuntimeError,FileNotFoundError)):module.prepare('full')
    assert not (module.destination('full')/'selection.json').exists()


def finish_primary(prepared):
    for metadata in prepared['primary']['confirmation_jobs']:
        run=Path(metadata['output_dir'])
        write_json(run/'experiment_summary.json',dict(config_sha256=metadata['config_sha256'],test_evaluated=False,test_metrics={}))
        for rank in range(4):write_json(run/f'rank_{rank}_complete.json',dict(rank=rank,world_size=4,config_sha256=metadata['config_sha256']))
        write_json(run/'retained/retention.json',dict(config_sha256=metadata['config_sha256'],selections={}))
        if metadata['regime']=='B':export_fixture(metadata,prepared['contract_path'],prepared['rows'])
        mark_complete(metadata)


def runtime(monkeypatch):
    monkeypatch.setattr(control_queue,'runtime_guard',lambda:None)
    monkeypatch.setenv('ACCELERATE_USE_CPU','false')
    monkeypatch.setattr(control_queue,'gpu_processes',lambda:[])
    monkeypatch.setattr(control_queue,'storage_gate',lambda:dict(ready=True))
    monkeypatch.setattr(control_driver,'baseline_gate',lambda *args:dict(ready=True))


def test_waits_primary_confirmations_and_respects_global_lock(prepared,monkeypatch):
    import fcntl
    runtime(monkeypatch);module.prepare('full')
    assert module.run_next('full')['phase']=='waiting_for_primary_confirmations'
    assert not list((prepared['directory']/'configs').glob('*seed43__lr0p3*'))
    with (prepared['directory']/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert module.run_next('full')['phase']=='waiting_for_global_driver_lock'


def test_exact_supplements_run_once_and_union_completion_checks_exports(prepared,monkeypatch):
    runtime(monkeypatch);module.prepare('full');finish_primary(prepared)
    primary_before={Path(j['config_path']):Path(j['config_path']).read_bytes() for j in prepared['primary']['confirmation_jobs']}
    calls=[]
    def execute(metadata,plan,gate):
        calls.append(metadata)
        assert metadata['regime']=='B' and metadata['phase']=='allocation_confirmation'
        run=Path(metadata['output_dir'])
        write_json(run/'experiment_summary.json',dict(config_sha256=metadata['config_sha256'],test_evaluated=False,test_metrics={}))
        for rank in range(4):write_json(run/f'rank_{rank}_complete.json',dict(rank=rank,world_size=4,config_sha256=metadata['config_sha256']))
        write_json(run/'retained/retention.json',dict(config_sha256=metadata['config_sha256'],selections={}))
        export_fixture(metadata,prepared['contract_path'],prepared['rows']);mark_complete(metadata)
        return dict(phase='training_complete_test_not_scored')
    monkeypatch.setattr(control_driver,'execute_job',execute)
    for _ in range(4):assert module.run_next('full')['phase']=='one_allocation_confirmation_complete'
    assert [(j['seed'],j['learning_rate_multiplier']) for j in calls]==[(43,.3),(43,3.),(44,.3),(44,3.)]
    result=module.run_next('full')
    assert result['phase']=='allocation_confirmation_union_complete_other_work_pending'
    assert len(result['evidence'])==6 and not result['study_complete']
    assert module.run_next('full')==result and len(calls)==4
    assert all(p.read_bytes()==b for p,b in primary_before.items())
    output=predictions.destination(Path(calls[-1]['output_dir']))
    (output/'epoch01_val/predictions.parquet').write_bytes(b'corrupt')
    with pytest.raises(ValueError):module.run_next('full')


@pytest.mark.parametrize('condition',['busy','space','cpu','partial_output','changed_selection'])
def test_supplement_launch_guards(prepared,monkeypatch,condition):
    runtime(monkeypatch);selection=module.prepare('full');finish_primary(prepared)
    calls=[];monkeypatch.setattr(control_driver,'execute_job',lambda *args:calls.append(args))
    if condition=='busy':monkeypatch.setattr(control_queue,'gpu_processes',lambda:[dict(pid=100)])
    elif condition=='space':monkeypatch.setattr(control_queue,'storage_gate',lambda:dict(ready=False))
    elif condition=='cpu':monkeypatch.setenv('ACCELERATE_USE_CPU','true')
    elif condition=='partial_output':
        identity=next(r['id'] for r in selection['confirmation_requests'] if not r['reuse_primary'])
        (prepared['directory']/'runs'/identity).mkdir()
    elif condition=='changed_selection':
        p=module.destination('full')/'selection.json';r=module.read(p);r['winning_lr_union']=[1.];write_json(p,r)
    if condition in ('busy','space'):
        assert module.run_next('full')['phase']==('waiting_for_authorized_gpus' if condition=='busy' else 'waiting_for_storage')
    else:
        with pytest.raises((RuntimeError,ValueError)):module.run_next('full')
    assert not calls


def test_preflight_does_not_create_selection_while_tuning_is_incomplete(prepared):
    job=prepared['jobs'][1];(Path(job['output_dir'])/'controlled_training_complete.json').unlink()
    result=module.preflight('full')
    assert result['phase']=='waiting_for_primary_tuning_and_selection'
    assert result['incomplete_tuning']==[job['id']]
    assert not module.destination('full').exists()


def test_union_reuses_only_identical_primary_lr_and_never_adds_a_runs():
    shared={name:dict(learning_rate_multiplier=1.) for name in ('original','second','third')}
    requests=module.confirmation_requests('full',shared,1.)
    assert len(requests)==2 and all(r['reuse_primary'] for r in requests)
    different=dict(original=dict(learning_rate_multiplier=.3),second=dict(learning_rate_multiplier=3.),
                   third=dict(learning_rate_multiplier=.3))
    requests=module.confirmation_requests('full',different,1.)
    assert len(requests)==4 and all(not r['reuse_primary'] for r in requests)
    assert all('__B__' in r['id'] for r in requests)


def test_selector_rejects_incomplete_lr_budget_and_undefined_coverage():
    with pytest.raises(ValueError,match='three distinct'):module.select_allocations([],['a','b','c'])
    history=[dict(epoch=e,step=e,macro_spearman=1.,defined_fraction=0.) for e in range(1,16)]
    candidates=[dict(job_id=str(lr),learning_rate_multiplier=lr,test_metrics_computed=False,
        selections={a:dict(validation_history=history,selected=history[0]) for a in ('a','b','c')})
        for lr in (.3,1.,3.)]
    with pytest.raises(ValueError,match='No defined'):module.select_allocations(candidates,['a','b','c'])
