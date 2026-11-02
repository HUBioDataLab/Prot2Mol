"""Native validation losses for selected B allocations, with fixed LR/macro rules.

No test-role evaluation is performed. Each validation view preserves whole units,
the frozen row order and labels. Loss pools follow the production single-device
evaluator; uniformly batched scalar Spearman still determines the frozen LR and
macro checkpoint rules. These numerical protocols are reported separately.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from . import control_plan, control_driver, control_queue, postselection
    from . import realization_selection, realization_predictions, realization_protocol
    from . import checkpoint_evaluation, retain_controlled
else:
    from run import ROOT, WORK, sha256, write_json
    import control_plan, control_driver, control_queue, postselection
    import realization_selection, realization_predictions, realization_protocol
    import checkpoint_evaluation, retain_controlled


read = postselection.read_json
immutable = postselection.immutable_json


class AllocationViews:
    """Reference existing token buffers; do not retokenize or change supervision."""
    def __init__(self, contract_path, primary_path):
        from datasets import concatenate_datasets, load_from_disk
        self.contract_path = Path(contract_path)
        self.contract,self.rows,self.indices = realization_protocol.load_contract(self.contract_path.parent)
        primary_path = Path(primary_path)
        primary = read(primary_path)
        if (self.contract['control_plan_sha256'] != sha256(primary_path) or
            Path(self.contract['control_plan_path']) != primary_path or self.contract['training_identical'] is not True):
            raise ValueError('Allocation views differ from the primary training support')
        caches=[]
        for split in ('val','test'):
            source=dict(source=primary['bases']['B']['data'][f'{split}_parquet_path'],
                source_sha256=primary['source_data_sha256']['B'][split],
                cache=str(Path(primary['bases']['B']['data']['tokenized_dataset_dir'])/f'{split}_examples'),
                cache_sha256=primary['tokenized_data_sha256']['B'][split])
            postselection.verify_data(source)
            dataset=load_from_disk(source['cache'])
            if dataset._fingerprint != self.contract['dataset_fingerprints'][split]:
                raise ValueError('Loaded allocation token-cache fingerprint changed')
            expected=self.rows[self.rows.original_split==split].reset_index(drop=True)
            locations=realization_protocol.align_rows(expected,
                dataset.select_columns(realization_protocol.IDENTITY+realization_protocol.LABELS).to_pandas())
            if not np.array_equal(locations,np.arange(len(expected))):
                raise ValueError('Allocation cache order differs from its frozen layout')
            caches.append(dataset)
        self.union=concatenate_datasets(caches)
        for allocation,counts in self.contract['realizations'].items():
            role=np.full(len(self.rows),'',dtype='<U4')
            covered=[]
            for split in ('val','test'):
                positions=self.indices[f'{allocation}_{split}']
                if (positions.dtype.kind not in 'iu' or len(np.unique(positions))!=len(positions) or
                    np.any(positions<0) or np.any(positions>=len(self.rows)) or
                    not np.array_equal(positions,np.sort(positions))):
                    raise ValueError('Allocation row indices changed')
                subset=self.rows.iloc[positions]
                actual=dict(rows=len(subset),units=int(subset.component.nunique()),groups=int(subset.group_id.nunique()))
                if actual!=counts[split]:raise ValueError('Allocation coverage differs from frozen counts')
                role[positions]=split;covered.extend(positions.tolist())
            if sorted(covered)!=list(range(len(self.rows))):
                raise ValueError('Allocation roles overlap or omit held-out rows')
            grouped=self.rows[['component','group_id']].assign(role=role)
            if any((grouped.groupby(key).role.nunique()!=1).any() for key in ('component','group_id')):
                raise ValueError('An indivisible unit or assay crosses allocation roles')

    def validation(self, allocation):
        if allocation not in self.contract['realizations']:raise ValueError('Unknown allocation')
        positions=self.indices[f'{allocation}_val']
        fingerprint=control_plan.digest_json(dict(contract_sha256=sha256(self.contract_path),allocation=allocation,role='val'))
        return self.union.select(positions,new_fingerprint=fingerprint)


def qualify_views():
    """Read production caches without loading any model or using GPU devices."""
    if (ROOT!=Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or
        os.environ.get('ACCELERATE_USE_CPU')!='true'):
        raise RuntimeError('Production view qualification requires Atum and explicit CPU-only mode')
    contract=WORK/'controlled/realizations/plan.json';primary=WORK/'controlled/plan.json'
    views=AllocationViews(contract,primary);records={}
    for name in views.contract['realizations']:
        dataset=views.validation(name)
        reference=views.rows.iloc[views.indices[name+'_val']].reset_index(drop=True)
        positions=realization_protocol.align_rows(reference,
            dataset.select_columns(realization_protocol.IDENTITY+realization_protocol.LABELS).to_pandas())
        if not np.array_equal(positions,np.arange(len(dataset))):raise ValueError('Selected view changed row order')
        records[name]=dict(rows=len(dataset),groups=int(reference.group_id.nunique()),
            units=int(reference.component.nunique()),fingerprint=dataset._fingerprint)
    return dict(contract_sha256=sha256(contract),primary_plan_sha256=sha256(primary),
        source_sha256=sha256(Path(__file__)),validation_views=records,
        token_cache_payloads_verified=True,model_inference_performed=False,training_started=False,
        interpretation='Production data-view qualification only; no losses, selection, averages or test scores')


def registered_selected_job(identity):
    directory=WORK/'controlled'
    if Path(identity).name!=identity:raise ValueError('Expected a job ID')
    metadata=read(directory/'jobs'/f'{identity}.json')
    plan=read(directory/'plan.json')
    selection=realization_selection.verify_selection(metadata['arm'])
    allocations=sorted(name for name,s in selection['selections'].items()
                       if s['learning_rate_multiplier']==metadata['learning_rate_multiplier'])
    if metadata['regime']!='B' or not allocations:
        raise ValueError('Loss comparison requires an allocation-selected B LR')
    if metadata['seed']==control_plan.TUNING_SEED:
        metadata=postselection.registered_job(directory,identity,plan)
        if metadata['phase']!='tuning':raise ValueError('Unexpected tuning seed role')
    else:
        requests=[r for r in selection['confirmation_requests'] if r['id']==identity]
        if len(requests)!=1:raise ValueError('Confirmation is outside the frozen allocation union')
        request=requests[0]
        if request['reuse_primary']:
            metadata=postselection.registered_job(directory,identity,plan)
            if metadata['phase']!='confirmation' or metadata['selection_sha256']!=selection['primary_selection_sha256']:
                raise ValueError('Primary confirmation selection changed')
        else:
            expected=realization_selection.expected_supplement(request,selection,plan)
            realization_selection.verify_registration(metadata,expected,selection['primary_plan_sha256'])
            if not control_driver.completed_job(metadata):raise RuntimeError('Supplemental training is incomplete')
    return metadata,plan,selection,allocations


def build_plan(identity, workers=4):
    from importlib.metadata import version
    from reward_model.training.config import load_reward_training_config
    if not 1<=workers<=4:raise ValueError('Use one to four authorized workers')
    metadata,primary,selection,allocations=registered_selected_job(identity)
    run=Path(metadata['output_dir'])
    config=load_reward_training_config(metadata['config_path'])
    uniform=realization_predictions.verify_validation(run)
    uniform_path=realization_predictions.destination(run)/'validation_complete.json'
    retained=read(run/'retained/retention.json')['additional_realizations']
    if (retained['sha256']!=sha256(uniform_path) or Path(retained['path'])!=uniform_path or
        uniform['config_sha256']!=metadata['config_sha256'] or uniform['contract_sha256']!=selection['contract_sha256']):
        raise ValueError('Allocation trajectory differs from completed retention evidence')
    frozen=read(realization_predictions.destination(run)/'jobs.json')
    model_config=read(run/'config.json')
    postselection.verify_model_config(model_config,config.model)
    checkpoints=[]
    for job,weight in zip(frozen['jobs'],uniform['source_weights'],strict=True):
        path=run/f"checkpoint-{job['step']}"
        if (Path(job['checkpoint'])!=path or weight['path']!=str((path/'pytorch_model.bin').relative_to(run)) or
            job['weights_sha256']!=weight['sha256'] or (job['epoch'],job['step'])!=(weight['epoch'],weight['step']) or
            read(path/'config.json')!=model_config or sha256(path/'config.json')!=job['model_config_sha256']):
            raise ValueError('Full validation checkpoint provenance changed')
        checkpoints.append(dict(id=path.name,path=str(path),**{k:job[k] for k in
            ('epoch','step','weights_sha256','model_config_sha256')}))
    if [c['epoch'] for c in checkpoints]!=list(range(1,16)):
        raise ValueError('All 15 validation checkpoints are required')
    known=None
    if config.model.contrastive_mask_policy=='training_known':
        if config.data.train_view_manifest is not None:raise ValueError('Full-data loss comparison cannot use a training subset')
        known=dict(source=config.data.train_parquet_path,source_sha256=primary['source_data_sha256']['B']['train'],
            cache=str(Path(config.data.tokenized_dataset_dir)/'train_examples'),
            cache_sha256=primary['tokenized_data_sha256']['B']['train'],
            provenance=read(run/'known_positive_index.json'),provenance_sha256=sha256(run/'known_positive_index.json'))
    modules=(control_plan,control_driver,postselection,realization_selection,realization_predictions,
             realization_protocol,checkpoint_evaluation,retain_controlled)
    paths=[Path(__file__),*(Path(m.__file__) for m in modules),*(ROOT/'RewardModel/reward_model').rglob('*.py')]
    return dict(format_version=1,job=metadata,workers=workers,allocations=allocations,
        checkpoints=checkpoints,primary_plan_path=str(WORK/'controlled/plan.json'),
        primary_plan_sha256=selection['primary_plan_sha256'],
        selection_sha256=sha256(realization_selection.destination(metadata['arm'])/'selection.json'),
        contract_path=str(WORK/'controlled/realizations/plan.json'),contract_sha256=selection['contract_sha256'],
        uniform_validation_path=str(uniform_path),uniform_validation_sha256=sha256(uniform_path),
        completion_sha256=sha256(run/'controlled_training_complete.json'),known_positive_source=known,
        software=dict(sources_sha256={str(p.resolve().relative_to(ROOT.resolve())):sha256(p) for p in paths},
            packages={p:version(p) for p in ('torch','transformers','datasets','numpy')}),
        test_roles_evaluated=False,
        interpretation='Single-device native validation loss pools; uniform batch-32 scalar Spearman still determines LR and macro rules. Dependent allocations, unchanged training support. No test selection or weight release.')


def prepare(identity, workers=4):
    plan=build_plan(identity,workers)
    if ROOT==Path('/home/atabey/Prot2Mol'):
        if __package__:from .prepare_random import archive_directory
        else:from prepare_random import archive_directory
        archive_directory(WORK/'realization_losses')
    path=WORK/'realization_losses'/identity/'plan.json'
    immutable(path,plan)
    return path


def source_for(path, checkpoint, allocation, index, cpu):
    return dict(plan_sha256=sha256(path),checkpoint=checkpoint,allocation=allocation,split='val',
                gpu_uuid=None if cpu else control_plan.GPU_UUIDS[index])


def evaluate_worker(path, index, *, cpu_qualification=False):
    import torch
    from datasets import load_from_disk
    from reward_model.model import load_reward_model
    from reward_model.training.config import load_reward_training_config
    from reward_model.training.known_positives import KnownPositiveIndex
    path=Path(path);plan=read(path)
    if not 0<=index<plan['workers']:raise ValueError('Invalid loss worker index')
    if cpu_qualification:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('ACCELERATE_USE_CPU')!='true':
            raise RuntimeError('CPU qualification requires an explicitly CPU-only process')
    elif (os.environ.get('CUDA_VISIBLE_DEVICES')!=control_plan.GPU_UUIDS[index] or
          os.environ.get('ACCELERATE_USE_CPU','false').lower() not in ('false','0','no')):
        raise RuntimeError('Loss worker must use its authorized GPU')
    if plan!=build_plan(plan['job']['id'],plan['workers']):raise ValueError('Frozen allocation-loss plan changed')
    torch.set_num_threads(4)
    views=AllocationViews(plan['contract_path'],plan['primary_plan_path'])
    datasets={name:views.validation(name) for name in plan['allocations']}
    config=load_reward_training_config(plan['job']['config_path'])
    known=None
    if plan['known_positive_source'] is not None:
        source=plan['known_positive_source'];postselection.verify_data(source)
        known=KnownPositiveIndex.from_training_dataset(load_from_disk(source['cache']),
            active_threshold=config.model.contrastive_active_threshold)
        if known.provenance!=source['provenance']:raise ValueError('Training-positive index changed')
    for checkpoint in plan['checkpoints'][index::plan['workers']]:
        model=None
        for allocation in plan['allocations']:
            output=path.parent/checkpoint['id']/allocation/'val'
            source=source_for(path,checkpoint,allocation,index,cpu_qualification)
            if (output/'complete.json').exists():
                postselection.verify_evaluation(output,source,cpu_qualification=cpu_qualification)
                if not (output/'cost.json').exists():raise RuntimeError('Completed pass is missing its measured cost receipt')
                continue
            if output.exists():raise RuntimeError('Partial loss evaluation requires inspection')
            if model is None:
                cp=Path(checkpoint['path'])
                if sha256(cp/'pytorch_model.bin')!=checkpoint['weights_sha256'] or sha256(cp/'config.json')!=checkpoint['model_config_sha256']:
                    raise ValueError('Checkpoint changed before native loss evaluation')
                model=load_reward_model(cp,device=torch.device('cpu'))
                postselection.verify_model_config(model.config.to_dict(),config.model,encoder_widths={
                    side:getattr(model,side+'_encoder').config.hidden_size for side in ('protein','molecule')})
                if json.loads(json.dumps(model.config.to_dict()))!=read(cp/'config.json'):
                    raise ValueError('Loaded checkpoint configuration changed')
                model.to('cpu' if cpu_qualification else 'cuda')
            started=time.monotonic()
            checkpoint_evaluation.evaluate_fixed_model(model,datasets[allocation],config,output,prefix='eval',
                known_positive_index=known,source=source,cpu=cpu_qualification)
            postselection.verify_evaluation(output,source,cpu_qualification=cpu_qualification)
            immutable(output/'cost.json',dict(elapsed_seconds=time.monotonic()-started,
                bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())))
        del model
        torch.cuda.empty_cache()


def collect_rules(path, *, cpu_qualification=False):
    """Freeze loss/macro rule contrasts only after every validation pass verifies."""
    path=Path(path);plan=read(path)
    if plan!=build_plan(plan['job']['id'],plan['workers']):raise ValueError('Frozen allocation-loss plan changed')
    uniform=read(plan['uniform_validation_path'])
    selections={};evidence={};elapsed=0.;size=0
    for allocation in plan['allocations']:
        records=[]
        histories=uniform['selections'][allocation]['validation_history']
        for n,(checkpoint,macro) in enumerate(zip(plan['checkpoints'],histories,strict=True)):
            output=path.parent/checkpoint['id']/allocation/'val'
            source=source_for(path,checkpoint,allocation,n%plan['workers'],cpu_qualification)
            result=postselection.verify_evaluation(output,source,cpu_qualification=cpu_qualification)
            cost=read(output/'cost.json')
            if not np.isfinite(cost['elapsed_seconds']) or cost['elapsed_seconds']<0 or cost['bytes']<0:
                raise ValueError('Invalid evaluation cost receipt')
            elapsed+=cost['elapsed_seconds'];size+=cost['bytes']
            if (macro['epoch'],macro['step'])!=(checkpoint['epoch'],checkpoint['step']):
                raise ValueError('Uniform macro and native loss epochs differ')
            records.append(dict(epoch=checkpoint['epoch'],step=checkpoint['step'],
                eval_macro_spearman=macro['macro_spearman'],eval_loss=result['metrics']['eval_loss'],
                native_eval_macro_spearman=result['metrics']['eval_macro_spearman'],
                uniform_defined_fraction=macro['defined_fraction']))
            evidence[str(output.relative_to(path.parent))]=sha256(output/'complete.json')
            evidence[str((output/'cost.json').relative_to(path.parent))]=sha256(output/'cost.json')
        rules=retain_controlled.selection_rules(records)
        if rules['best_macro']['step']!=uniform['selections'][allocation]['selected']['step']:
            raise ValueError('Native loss analysis changed the frozen macro checkpoint')
        start=int(rules['best_loss']['epoch'])
        selections[allocation]=dict(rules=rules,validation_history=records,
            averaging_epochs=list(range(start,16)))
    result=dict(plan_sha256=sha256(path),selections=selections,evidence_sha256=evidence,
        cpu_qualification=cpu_qualification,test_roles_evaluated=False,
        measured_cost=dict(sum_worker_pass_seconds=elapsed,evaluation_bytes=size,
            interpretation='Sum of timed evaluation passes, not parallel wall time; excludes input loading and checkpoint loading'),
        remaining='Create and verify validation-selected averages, evaluate selected test roles, summarize allocation/seed spread; no weight release')
    immutable(path.parent/'validation_complete.json',result)
    return result


def create_averages(path, *, cpu_qualification=False):
    """Average verified validation-selected windows; identical windows share bytes."""
    path=Path(path);plan=read(path)
    validation=collect_rules(path,cpu_qualification=cpu_qualification)
    targets={};allocations={}
    for allocation,selection in validation['selections'].items():
        epochs=selection['averaging_epochs'];name=f'epochs{epochs[0]:02d}_15'
        allocations[allocation]=name
        if name in targets:continue
        checkpoints=[c for c in plan['checkpoints'] if c['epoch'] in epochs]
        if [c['epoch'] for c in checkpoints]!=epochs:raise ValueError('Averaging window omits checkpoints')
        config_path=Path(checkpoints[0]['path'])/'config.json'
        for checkpoint in checkpoints:
            cp=Path(checkpoint['path'])
            if (sha256(cp/'pytorch_model.bin')!=checkpoint['weights_sha256'] or
                sha256(cp/'config.json')!=checkpoint['model_config_sha256'] or read(cp/'config.json')!=read(config_path)):
                raise ValueError('Averaging source checkpoint changed')
        target=path.parent/'averages'/name
        expected=dict(plan_sha256=sha256(path),validation_sha256=sha256(path.parent/'validation_complete.json'),
            source_checkpoints=checkpoints,epochs=epochs,model_config_sha256=sha256(config_path),
            cpu_qualification=cpu_qualification,singleton=len(epochs)==1,
            rule='uniform native parameters from allocation minimum validation loss through epoch15, inclusive')
        if (target/'complete.json').exists():
            result=read(target/'complete.json')
            if (any(result.get(k)!=v for k,v in expected.items()) or
                sha256(target/'pytorch_model.bin')!=result['weights_sha256'] or
                sha256(target/'config.json')!=result['saved_model_config_sha256'] or
                read(target/'config.json')!=read(config_path)):
                raise ValueError('Verified allocation average changed')
        else:
            if target.exists():raise RuntimeError('Partial allocation average requires inspection')
            started=time.monotonic()
            retain_controlled.average_weights([Path(c['path'])/'pytorch_model.bin' for c in checkpoints],target/'pytorch_model.bin')
            immutable(target/'config.json',read(config_path))
            result=dict(**expected,weights_sha256=sha256(target/'pytorch_model.bin'),
                saved_model_config_sha256=sha256(target/'config.json'),
                elapsed_seconds=time.monotonic()-started,weight_bytes=(target/'pytorch_model.bin').stat().st_size)
            immutable(target/'complete.json',result)
        targets[name]=dict(path=str(target),receipt_sha256=sha256(target/'complete.json'),**result)
    result=dict(plan_sha256=sha256(path),validation_sha256=sha256(path.parent/'validation_complete.json'),
        allocations=allocations,targets=targets,cpu_qualification=cpu_qualification,
        test_roles_evaluated=False,weights_released=False,
        remaining='Selected test evaluation and statistical allocation/seed comparisons; no checkpoint release')
    immutable(path.parent/'averages_complete.json',result)
    return result


def run(identity, workers=4):
    import fcntl
    control_queue.runtime_guard()
    with (WORK/'controlled/driver.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:return dict(phase='waiting_for_global_driver_lock')
        busy,storage=control_queue.gpu_processes(),control_queue.storage_gate()
        if busy:return dict(phase='waiting_for_authorized_gpus',live_processes=busy)
        if not storage['ready']:return dict(phase='waiting_for_storage',storage=storage)
        path=prepare(identity,workers);children=[];handles=[]
        try:
            for index in range(workers):
                env=control_driver.launch_environment(identity)
                env.update(CUDA_VISIBLE_DEVICES=control_plan.GPU_UUIDS[index],ACCELERATE_USE_CPU='false',WANDB_MODE='disabled')
                handle=(path.parent/f'worker{index}.log').open('a');handles.append(handle)
                children.append(subprocess.Popen([str(ROOT/'.venv/bin/python'),str(Path(__file__)),
                    'worker','--plan',str(path),'--index',str(index)],env=env,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT))
            write_json(path.parent/'status.json',dict(phase='validation_losses',pid=os.getpid(),
                worker_pids=[c.pid for c in children],started_unix=time.time()))
            codes=[c.wait() for c in children]
            if any(codes):raise RuntimeError(f'Loss workers failed; inspect before retry: {codes}')
        finally:
            for child in children:
                if child.poll() is None:child.terminate();child.wait()
            for handle in handles:handle.close()
        result=create_averages(path)
        write_json(path.parent/'status.json',dict(phase='validation_losses_and_averages_complete_other_work_pending',study_complete=False))
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','worker','collect','average','qualify-views'))
    parser.add_argument('--job-id');parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--plan',type=Path);parser.add_argument('--index',type=int)
    args=parser.parse_args()
    if args.action=='qualify-views':print(json.dumps(qualify_views(),indent=2))
    elif args.action=='worker':evaluate_worker(args.plan,args.index)
    elif args.action=='collect':print(json.dumps(collect_rules(args.plan),indent=2))
    elif args.action=='average':print(json.dumps(create_averages(args.plan),indent=2))
    else:print(prepare(args.job_id,args.workers) if args.action=='prepare' else json.dumps(run(args.job_id,args.workers),indent=2))
