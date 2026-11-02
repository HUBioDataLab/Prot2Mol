"""Evaluate allocation-selected checkpoints and averages after validation freezes.

Reuse verified native validation passes for epoch weights. Evaluate their test
roles, and both roles for each selected average. No test metric influences an
LR, checkpoint, averaging window, or training budget. Statistical reporting and
weight release remain separate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

if __package__:
    from .run import ROOT,WORK,sha256,write_json
    from . import realization_losses,realization_protocol,control_plan,control_queue,control_driver
    from . import postselection,checkpoint_evaluation,native_analysis
else:
    from run import ROOT,WORK,sha256,write_json
    import realization_losses,realization_protocol,control_plan,control_queue,control_driver
    import postselection,checkpoint_evaluation,native_analysis

read=postselection.read_json
immutable=postselection.immutable_json


def verify_loss_stage(identity, *, cpu_qualification=False):
    if Path(identity).name!=identity:raise ValueError('Expected a job ID')
    path=WORK/'realization_losses'/identity/'plan.json';plan=read(path)
    if plan!=realization_losses.build_plan(identity,plan['workers']):
        raise ValueError('Upstream allocation-loss plan changed')
    # Require both existing stages; this evaluator must not create missing choices.
    read(path.parent/'validation_complete.json')
    averages=read(path.parent/'averages_complete.json')
    validation=realization_losses.collect_rules(path,cpu_qualification=cpu_qualification)
    if (averages['plan_sha256']!=sha256(path) or
        averages['validation_sha256']!=sha256(path.parent/'validation_complete.json') or
        averages['cpu_qualification']!=cpu_qualification or averages['test_roles_evaluated'] is not False or
        averages['weights_released'] is not False):
        raise ValueError('Allocation averaging stage has different provenance')
    expected_names={name:f"epochs{item['averaging_epochs'][0]:02d}_15"
                    for name,item in validation['selections'].items()}
    if averages['allocations']!=expected_names or set(averages['targets'])!=set(expected_names.values()):
        raise ValueError('Allocation average coverage differs from validation choices')
    for name,target in averages['targets'].items():
        directory=path.parent/'averages'/name
        record=read(directory/'complete.json')
        if target!=dict(path=str(directory),receipt_sha256=sha256(directory/'complete.json'),**record):
            raise ValueError('Allocation average receipt changed')
        epochs=next(v['averaging_epochs'] for a,v in validation['selections'].items() if expected_names[a]==name)
        checkpoints=[c for c in plan['checkpoints'] if c['epoch'] in epochs]
        expected=dict(plan_sha256=sha256(path),validation_sha256=sha256(path.parent/'validation_complete.json'),
            source_checkpoints=checkpoints,epochs=epochs,
            model_config_sha256=sha256(Path(checkpoints[0]['path'])/'config.json'),
            cpu_qualification=cpu_qualification,singleton=len(epochs)==1,
            rule='uniform native parameters from allocation minimum validation loss through epoch15, inclusive')
        if (any(record.get(k)!=v for k,v in expected.items()) or
            sha256(directory/'pytorch_model.bin')!=record['weights_sha256'] or
            sha256(directory/'config.json')!=record['saved_model_config_sha256'] or
            read(directory/'config.json')!=read(Path(checkpoints[0]['path'])/'config.json')):
            raise ValueError('Validation-selected average changed')
    return path,plan,validation,averages


def build_plan(identity, workers=4, *, cpu_qualification=False):
    from importlib.metadata import version
    if not 1<=workers<=4:raise ValueError('Use one to four authorized workers')
    path,loss_plan,validation,averages=verify_loss_stage(identity,cpu_qualification=cpu_qualification)
    checkpoints={};rules={};references={};tasks=[]
    def register(checkpoint):
        directory=Path(checkpoint['path'])
        if (sha256(directory/'pytorch_model.bin')!=checkpoint['weights_sha256'] or
            sha256(directory/'config.json')!=checkpoint['model_config_sha256']):
            raise ValueError('Selected checkpoint changed')
        previous=checkpoints.setdefault(checkpoint['id'],checkpoint)
        if previous!=checkpoint:raise ValueError('Ambiguous selected checkpoint ID')
        return checkpoint['id']
    for allocation,selection in validation['selections'].items():
        rules[allocation]={};selected={}
        for name,rule in selection['rules'].items():
            candidates=[c for c in loss_plan['checkpoints'] if c['step']==rule['step']]
            if len(candidates)!=1:raise ValueError('Selected epoch is absent from validation trajectory')
            checkpoint=dict(**candidates[0],kind='epoch')
            key=register(checkpoint);selected[key]=checkpoint
            rules[allocation][name]=dict(**rule,checkpoint=key)
        average_name=averages['allocations'][allocation];average=averages['targets'][average_name]
        checkpoint=dict(id='average_'+average_name,path=average['path'],kind='average',
            weights_sha256=average['weights_sha256'],model_config_sha256=average['saved_model_config_sha256'])
        key=register(checkpoint);selected[key]=checkpoint
        rules[allocation]['average_from_loss_minimum']=dict(checkpoint=key,epochs=average['epochs'],singleton=average['singleton'])
        for key,checkpoint in sorted(selected.items()):
            if checkpoint['kind']=='epoch':
                n=next(i for i,c in enumerate(loss_plan['checkpoints']) if c['id']==key)
                original=loss_plan['checkpoints'][n]
                output=path.parent/key/allocation/'val'
                source=realization_losses.source_for(path,original,allocation,n%loss_plan['workers'],cpu_qualification)
                postselection.verify_evaluation(output,source,cpu_qualification=cpu_qualification)
                references[f'{key}/{allocation}/val']=dict(output=str(output),source=source,
                    completion_sha256=sha256(output/'complete.json'))
            else:tasks.append(dict(checkpoint=key,allocation=allocation,split='val'))
            tasks.append(dict(checkpoint=key,allocation=allocation,split='test'))
    modules=(realization_losses,realization_protocol,control_plan,control_queue,control_driver,
             postselection,checkpoint_evaluation,native_analysis)
    files=[Path(__file__),*(Path(m.__file__) for m in modules),*(ROOT/'RewardModel/reward_model').rglob('*.py')]
    return dict(format_version=1,job=loss_plan['job'],workers=workers,cpu_qualification=cpu_qualification,
        loss_plan_path=str(path),loss_plan_sha256=sha256(path),
        validation_sha256=sha256(path.parent/'validation_complete.json'),averages_sha256=sha256(path.parent/'averages_complete.json'),
        contract_path=loss_plan['contract_path'],primary_plan_path=loss_plan['primary_plan_path'],
        known_positive_source=loss_plan['known_positive_source'],rules=rules,
        checkpoints=checkpoints,validation_references=references,tasks=tasks,
        software=dict(sources_sha256={str(p.resolve().relative_to(ROOT.resolve())):sha256(p) for p in files},
            native_prediction_verifier=native_analysis.software(),
            packages={name:version(name) for name in ('torch','transformers','datasets','numpy')}),
        interpretation='Native single-device validation/test evaluation of frozen allocation rules; LR/macro selection retains the separate uniform-inference protocol. Original epoch validation outputs reused. No test-driven selection or weight release.')


def prepare(identity, workers=4):
    plan=build_plan(identity,workers)
    if ROOT==Path('/home/atabey/Prot2Mol'):
        if __package__:from .prepare_random import archive_directory
        else:from prepare_random import archive_directory
        archive_directory(WORK/'realization_postselection')
    path=WORK/'realization_postselection'/identity/'plan.json';immutable(path,plan)
    return path


def dataset_view(views, allocation, split):
    if allocation not in views.contract['realizations'] or split not in ('val','test'):
        raise ValueError('Unknown allocation or role')
    if split=='val':return views.validation(allocation)
    fingerprint=control_plan.digest_json(dict(contract_sha256=sha256(views.contract_path),allocation=allocation,role='test'))
    return views.union.select(views.indices[f'{allocation}_test'],new_fingerprint=fingerprint)


def source_for(path, plan, task, index):
    return dict(plan_sha256=sha256(path),checkpoint=plan['checkpoints'][task['checkpoint']],
        allocation=task['allocation'],split=task['split'],
        gpu_uuid=None if plan['cpu_qualification'] else control_plan.GPU_UUIDS[index])


def verify_output(output, source, views, dataset, *, cpu_qualification=False):
    result=postselection.verify_evaluation(output,source,cpu_qualification=cpu_qualification)
    allocation,split=source['allocation'],source['split']
    expected=views.rows.iloc[views.indices[f'{allocation}_{split}']].reset_index(drop=True)
    native_analysis.native_frame(result['prediction_manifest'],expected,dataset._fingerprint,
        'eval' if split=='val' else 'test')
    return result


def evaluate_worker(path,index, *, cpu_qualification=False):
    import torch
    from datasets import load_from_disk
    from reward_model.model import load_reward_model
    from reward_model.training.config import load_reward_training_config
    from reward_model.training.known_positives import KnownPositiveIndex
    path=Path(path);plan=read(path)
    if not 0<=index<plan['workers'] or plan['cpu_qualification']!=cpu_qualification:
        raise ValueError('Invalid allocation evaluator assignment or qualification mode')
    if cpu_qualification:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('ACCELERATE_USE_CPU')!='true':
            raise RuntimeError('CPU qualification requires an explicitly CPU-only process')
    elif (os.environ.get('CUDA_VISIBLE_DEVICES')!=control_plan.GPU_UUIDS[index] or
          os.environ.get('ACCELERATE_USE_CPU','false').lower() not in ('false','0','no')):
        raise RuntimeError('Allocation evaluator must use its authorized GPU')
    if plan!=build_plan(plan['job']['id'],plan['workers'],cpu_qualification=cpu_qualification):
        raise ValueError('Frozen allocation evaluation plan changed')
    views=realization_losses.AllocationViews(plan['contract_path'],plan['primary_plan_path'])
    datasets={(a,s):dataset_view(views,a,s) for a in plan['rules'] for s in ('val','test')}
    config=load_reward_training_config(plan['job']['config_path']);known=None
    if plan['known_positive_source'] is not None:
        source=plan['known_positive_source'];postselection.verify_data(source)
        known=KnownPositiveIndex.from_training_dataset(load_from_disk(source['cache']),
            active_threshold=config.model.contrastive_active_threshold)
        if known.provenance!=source['provenance']:raise ValueError('Training-positive index changed')
    torch.set_num_threads(4);model=None;loaded=None
    for task in plan['tasks'][index::plan['workers']]:
        key,allocation,split=(task[k] for k in ('checkpoint','allocation','split'))
        checkpoint=plan['checkpoints'][key];output=path.parent/key/allocation/split
        source=source_for(path,plan,task,index);dataset=datasets[allocation,split]
        if (output/'complete.json').exists():
            verify_output(output,source,views,dataset,cpu_qualification=cpu_qualification)
            if not (output/'cost.json').exists():raise RuntimeError('Completed evaluation lacks its cost receipt')
            continue
        if output.exists():raise RuntimeError('Partial allocation evaluation requires inspection')
        if loaded!=key:
            del model;torch.cuda.empty_cache();cp=Path(checkpoint['path'])
            if (sha256(cp/'pytorch_model.bin')!=checkpoint['weights_sha256'] or
                sha256(cp/'config.json')!=checkpoint['model_config_sha256']):
                raise ValueError('Selected checkpoint changed before evaluation')
            model=load_reward_model(cp,device=torch.device('cpu'))
            postselection.verify_model_config(model.config.to_dict(),config.model,encoder_widths={
                side:getattr(model,side+'_encoder').config.hidden_size for side in ('protein','molecule')})
            if json.loads(json.dumps(model.config.to_dict()))!=read(cp/'config.json'):
                raise ValueError('Loaded selected-model configuration changed')
            model.to('cpu' if cpu_qualification else 'cuda');loaded=key
        started=time.monotonic()
        checkpoint_evaluation.evaluate_fixed_model(model,dataset,config,output,prefix='eval' if split=='val' else 'test',
            known_positive_index=known,source=source,cpu=cpu_qualification)
        verify_output(output,source,views,dataset,cpu_qualification=cpu_qualification)
        immutable(output/'cost.json',dict(elapsed_seconds=time.monotonic()-started,
            bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())))
    del model


def verify_complete(path, *, cpu_qualification=False):
    path=Path(path);plan=read(path)
    if plan!=build_plan(plan['job']['id'],plan['workers'],cpu_qualification=cpu_qualification):
        raise ValueError('Frozen allocation evaluation plan changed')
    views=realization_losses.AllocationViews(plan['contract_path'],plan['primary_plan_path'])
    datasets={(a,s):dataset_view(views,a,s) for a in plan['rules'] for s in ('val','test')}
    evidence={};costs={}
    for key,reference in plan['validation_references'].items():
        output=Path(reference['output']);source=reference['source']
        if sha256(output/'complete.json')!=reference['completion_sha256']:
            raise ValueError('Reused validation completion changed')
        verify_output(output,source,views,datasets[source['allocation'],'val'],cpu_qualification=cpu_qualification)
        evidence[key]=dict(output=str(output),completion_sha256=sha256(output/'complete.json'),reused_validation=True)
    for number,task in enumerate(plan['tasks']):
        key=f"{task['checkpoint']}/{task['allocation']}/{task['split']}";output=path.parent/key
        source=source_for(path,plan,task,number%plan['workers'])
        verify_output(output,source,views,datasets[task['allocation'],task['split']],cpu_qualification=cpu_qualification)
        cost=read(output/'cost.json')
        if not math.isfinite(cost['elapsed_seconds']) or cost['elapsed_seconds']<0 or cost['bytes']<0:
            raise ValueError('Invalid evaluation cost')
        costs[key]=dict(**cost,receipt_sha256=sha256(output/'cost.json'))
        evidence[key]=dict(output=str(output),completion_sha256=sha256(output/'complete.json'),reused_validation=False)
    result=dict(plan_sha256=sha256(path),evidence=evidence,costs=costs,cpu_qualification=cpu_qualification,
        test_used_for_selection=False,weights_released=False,study_complete=False,
        remaining='Statistical per-allocation and seed comparisons, interpretation and final report; weight release requires its own verified protocol')
    immutable(path.parent/'complete.json',result)
    return result


def run(identity,workers=4):
    import fcntl
    control_queue.runtime_guard()
    with (WORK/'controlled/driver.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
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
                children.append(subprocess.Popen([str(ROOT/'.venv/bin/python'),str(Path(__file__)),'worker',
                    '--plan',str(path),'--index',str(index)],env=env,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT))
            write_json(path.parent/'status.json',dict(phase='selected_allocation_evaluation',pid=os.getpid(),
                worker_pids=[p.pid for p in children],started_unix=time.time()))
            codes=[p.wait() for p in children]
            if any(codes):raise RuntimeError(f'Allocation evaluation failed; inspect before retry: {codes}')
        finally:
            for child in children:
                if child.poll() is None:child.terminate();child.wait()
            for handle in handles:handle.close()
        result=verify_complete(path)
        write_json(path.parent/'status.json',dict(phase='allocation_evaluation_complete_analysis_pending',study_complete=False))
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','worker','verify'))
    parser.add_argument('--job-id');parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--plan',type=Path);parser.add_argument('--index',type=int)
    args=parser.parse_args()
    if args.action=='worker':evaluate_worker(args.plan,args.index)
    elif args.action=='verify':print(json.dumps(verify_complete(args.plan),indent=2))
    else:print(prepare(args.job_id,args.workers) if args.action=='prepare' else json.dumps(run(args.job_id,args.workers),indent=2))
