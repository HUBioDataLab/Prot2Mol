"""Measured reference-run cadence and storage scenarios; no work is scheduled.

Checkpoint save times are wall-clock proxies including epoch evaluation/export,
not isolated GPU kernel timings. Extrapolations are explicitly reference-paced.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time

if __package__:
    from .run import ROOT,WORK,sha256
    from . import control_plan,control_driver,control_queue
else:
    from run import ROOT,WORK,sha256
    import control_plan,control_driver,control_queue

read=control_queue.read


def cadence(samples,horizon=15):
    if len(samples)<2 or [s['epoch'] for s in samples]!=list(range(1,len(samples)+1)) or len(samples)>horizon:
        raise ValueError('Need a contiguous multi-epoch prefix within the registered horizon')
    differences=[b['mtime']-a['mtime'] for a,b in zip(samples,samples[1:])]
    if any(not (0<d<float('inf')) for d in differences):raise ValueError('Checkpoint save times must increase')
    median=statistics.median(differences)
    return dict(completed_epochs=len(samples),intervals=len(differences),
        seconds_between_saves=dict(minimum=min(differences),median=median,maximum=max(differences)),
        horizon_seconds_at_median_cadence=horizon*median,
        observed_first_to_last_save_seconds=samples[-1]['mtime']-samples[0]['mtime'],
        interpretation='Filesystem save-time differences, excluding startup/first epoch and final post-training work; includes work between epoch saves. This is a measured wall-clock proxy, not an isolated training benchmark.')


def footprint(roots):
    """Allocated bytes de-duplicate hard links across the supplied scoped roots."""
    global_seen=set();result=[]
    for root in roots:
        root=Path(root);local_seen=set();logical=allocated=additional=files=0;categories={}
        if not root.is_dir():raise ValueError('Missing scoped storage root')
        for path in root.rglob('*'):
            if path.is_symlink() or not path.is_file():continue
            stat=path.stat();key=(stat.st_dev,stat.st_ino);files+=1;logical+=stat.st_size
            if key not in local_seen:allocated+=stat.st_blocks*512;local_seen.add(key)
            if key not in global_seen:additional+=stat.st_blocks*512;global_seen.add(key)
            category='weights' if path.name=='pytorch_model.bin' else 'optimizer' if path.name=='optimizer.pt' else 'other'
            categories[category]=categories.get(category,0)+stat.st_size
        result.append(dict(path=str(root),files=files,logical_bytes=logical,unique_allocated_bytes=allocated,
            additional_allocated_bytes_across_roots=additional,categories_logical_bytes=categories))
    return result


def scenario(primary,timings,weight_bytes,retained_a_weights):
    jobs=primary['tuning_jobs'];settings=len(primary['arms'])
    counts={r:sum(j['regime']==r for j in jobs)+len(primary['confirmation_seeds'])*settings for r in ('A','B')}
    if (sum(counts.values())!=primary['accounting']['total_runs'] or set(counts)!={'A','B'} or
        primary['accounting']['epochs_per_run']!=15):raise ValueError('Primary fit accounting changed')
    if set(timings)!={'A','B'} or weight_bytes<=0 or retained_a_weights<1:raise ValueError('Missing reference costs')
    durations={r:timings[r]['horizon_seconds_at_median_cadence'] for r in ('A','B')}
    seconds=sum(counts[r]*durations[r] for r in counts)
    # Current B policy retains every epoch, plus selected root and one average.
    copies_b=15+2;copies=counts['A']*retained_a_weights+counts['B']*copies_b
    return dict(primary_fit_counts=counts,reference_seconds_per_fit=durations,
        full_model_pace_primary_wall_days=seconds/86400,
        full_model_pace_primary_gpu_hours=seconds*4/3600,
        retained_full_weight_bytes=weight_bytes*copies,
        retained_weight_copies_per_fit=dict(A=retained_a_weights,B=copies_b),
        additional_allocation_confirmation_bound=dict(minimum_fits=0,maximum_fits_per_setting=4,
            maximum_fits_all_settings=4*settings,
            maximum_wall_days_at_reference_B_pace=4*settings*durations['B']/86400),
        interpretation='A reference-paced scenario for the entire primary grid, not a completion ETA or lower/upper runtime bound. Frozen/partial/LoRA/loss arms may differ. Excludes queue delays, final exports/retention, selected inference/CPU reporting, learning curves, leak designs, adaptations and conditional methods. Storage is a full-model weight-only scenario using observed A retention and current B policy; it excludes data, predictions, optimizer peaks and any future verified releases. Extra allocation confirmations depend on the union of validation-selected LRs; the bound is not a request to run them.')


def audit():
    started=time.time()
    primary_path=WORK/'controlled/plan.json';primary=read(primary_path);sources={str(primary_path):sha256(primary_path)}
    timings={};runs={};roots=[];a_count=None;sample_bytes=None
    for regime in ('A','B'):
        identity=control_plan.job_id('full',regime,42,1.);metadata_path=WORK/'controlled/jobs'/f'{identity}.json'
        metadata=read(metadata_path);run=Path(metadata['output_dir']);launch_path=run/'launch_provenance.json';launch=read(launch_path)
        if (run!=WORK/'controlled/runs'/identity or metadata['id']!=identity or
            metadata['plan_sha256']!=sha256(primary_path) or sha256(Path(metadata['config_path']))!=metadata['config_sha256'] or
            launch['config_sha256']!=metadata['config_sha256'] or launch['world_size']!=4 or
            launch['visible_devices'].split(',')!=list(control_plan.GPU_UUIDS) or launch['config']['training']['num_train_epochs']!=15):
            raise ValueError('Reference run provenance changed')
        for path in (metadata_path,Path(metadata['config_path']),launch_path):sources[str(path)]=sha256(path)
        samples=[]
        for path in sorted(run.glob('checkpoint-*/trainer_state.json'),key=lambda p:int(p.parent.name.split('-')[-1])):
            stat=path.stat();record=read(path);digest=sha256(path)
            if path.stat().st_mtime_ns!=stat.st_mtime_ns:raise RuntimeError('Checkpoint state changed while sampling; inspect same live run')
            samples.append(dict(epoch=record['epoch'],step=record['global_step'],mtime=stat.st_mtime,path=str(path),sha256=digest))
            sources[str(path)]=digest
        timings[regime]=cadence(samples)
        complete=control_driver.completed_job(metadata)
        details=dict(job_id=identity,samples=samples,verified_training_complete=complete)
        if complete:
            path=run/'controlled_training_complete.json';record=read(path);sources[str(path)]=sha256(path)
            details['launch_file_to_completion_seconds']=record['finished_unix']-launch_path.stat().st_mtime
            details['completion_timing_limit']='Launch provenance file time is a proxy; excludes preparation before it was written.'
        runs[regime]=details;roots.append(run)
        if regime=='A':
            if not complete:raise ValueError('Need the completed A reference for observed retention')
            path=run/'retained/retention.json';retention=read(path);sources[str(path)]=sha256(path)
            a_count=len(retention['preserved_weights']);sample_bytes=(run/'pytorch_model.bin').stat().st_size
    protected=WORK/'controlled/protected_trajectories/full__A__seed42__lr1p0'
    roots.insert(1,protected)
    footprint_records=footprint(roots);projection=scenario(primary,timings,sample_bytes,a_count)
    storage=control_queue.storage_gate()
    for path,digest in sources.items():
        if sha256(Path(path))!=digest:raise RuntimeError('Sampled provenance changed; inspect same live run')
    result=dict(sampling_started_unix=started,checked_unix=time.time(),source_sha256=sources,module_sha256=sha256(Path(__file__)),
        measured_runs=runs,checkpoint_cadence=timings,scoped_footprints=footprint_records,storage=storage,
        reference_scenario=projection,new_fits_created=0,files_deleted=0,settings_or_scope_changed=False,
        footprint_limit='A live run can grow during the scoped file scan; this is not an atomic filesystem snapshot. Directory metadata and symlink targets outside the supplied roots are not counted.',
        affordability_decision='Pending measured allocation costs and user time/compute constraints; this inventory alone does not authorize or reject optional work.')
    path=WORK/'resource_audits/full_reference_cost'/(control_plan.digest_json(result)[:16]+'.json')
    control_queue.immutable(path,result);return path


if __name__=='__main__':
    if ROOT!=Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Use CPU-only Atum resource inventory')
    print(audit())
