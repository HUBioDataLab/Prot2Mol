"""Run one E41 whole-unit curve job with the primary grid's execution contract.

The 100% endpoint reuses completed primary full-arm fits. Smaller fractions
have their own equal LR budgets and B-validation-selected confirmation seeds.
The primary and curve drivers share one lock; neither can overlap GPU training.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess

import yaml

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from . import control_driver, control_plan
    from .learning_curve_plan import FRACTIONS, fraction_name
else:
    from run import ROOT, WORK, sha256, write_json
    import control_driver, control_plan
    from learning_curve_plan import FRACTIONS, fraction_name


def frozen_metadata(job, directory, digest):
    metadata = json.loads((directory/'jobs'/f"{job['id']}.json").read_text())
    expected = {key:value for key,value in job.items() if key!='config'}
    if metadata['plan_sha256'] != digest or any(metadata.get(k)!=v for k,v in expected.items()):
        raise ValueError('Curve job belongs to different prepared inputs')
    path = Path(metadata['config_path'])
    if (sha256(path) != metadata['config_sha256'] or
        control_plan.digest_json(yaml.safe_load(path.read_text())) != control_plan.digest_json(job['config'])):
        raise ValueError('Curve configuration differs from the frozen plan')
    if Path(metadata['output_dir']) != Path(job['config']['training']['output_dir']):
        raise ValueError('Curve output directory differs from its configuration')
    return metadata


def confirm_fraction(fraction, directory=None):
    if fraction not in FRACTIONS[:-1]:
        raise ValueError('Only the registered smaller fractions need new fits')
    directory = Path(directory or WORK/'learning_curves')
    plan_path = directory/'plan.json'
    plan = json.loads(plan_path.read_text())
    digest = sha256(plan_path)
    jobs = [j for j in plan['tuning_jobs'] if j['fraction']==fraction]
    expected = {(regime,multiplier) for regime in ('A','B') for multiplier in control_plan.LEARNING_RATE_MULTIPLIERS}
    if len(jobs)!=6 or {(j['regime'],j['learning_rate_multiplier']) for j in jobs} != expected:
        raise ValueError('A curve fraction needs all six distinct A/B tuning jobs')
    candidates,evidence = [],[]
    for job in jobs:
        metadata = frozen_metadata(job,directory,digest)
        if not control_driver.completed_job(metadata):
            raise ValueError('Curve tuning job is not fully complete')
        candidate,source = control_plan.validation_candidate(metadata)
        evidence.append(source)
        if job['regime']=='B':
            candidates.append(candidate)
    name = fraction_name(fraction)
    selection = dict(fraction=fraction,plan_sha256=digest,
        **control_plan.select_learning_rate(candidates),evidence=evidence)
    path = directory/'selections'/f'{name}.json'
    control_plan._write_immutable(path,json.dumps(selection,sort_keys=True,indent=2)+'\n')
    confirmations = []
    for seed in control_plan.CONFIRMATION_SEEDS:
        for regime in ('A','B'):
            winner = next(j for j in jobs if j['regime']==regime and
                          j['learning_rate_multiplier']==selection['learning_rate_multiplier'])
            job = copy.deepcopy(winner)
            job['id'] = control_plan.job_id(job['arm'],regime,seed,job['learning_rate_multiplier'])
            job.update(seed=seed,phase='confirmation',selection_sha256=sha256(path))
            job['config']['training'].update(seed=seed,output_dir=str(directory/'runs'/job['id']))
            confirmations.append(control_plan.materialize_job(job,directory,digest))
    return dict(selection=selection,confirmation_jobs=confirmations)


def next_job(directory, plan):
    directory = Path(directory)
    digest = sha256(directory/'plan.json')
    for fraction in FRACTIONS[:-1]:
        for job in (j for j in plan['tuning_jobs'] if j['fraction']==fraction):
            metadata = frozen_metadata(job,directory,digest)
            if not control_driver.completed_job(metadata):
                return metadata
        for metadata in confirm_fraction(fraction,directory)['confirmation_jobs']:
            if not control_driver.completed_job(metadata):
                return metadata
    return None


def full_endpoint_gate(primary):
    directory = WORK/'controlled'
    digest = sha256(directory/'plan.json')
    evidence,missing = [],[]
    for job in (j for j in primary['tuning_jobs'] if j['arm']=='full'):
        metadata = frozen_metadata(job,directory,digest)
        if not control_driver.completed_job(metadata):
            missing.append(job['id'])
        else:
            path = Path(metadata['output_dir'])/'controlled_training_complete.json'
            evidence.append(dict(job_id=job['id'],path=str(path),sha256=sha256(path)))
    if missing:
        return dict(ready=False,missing=missing,evidence=evidence)
    for metadata in control_plan.confirm('full')['confirmation_jobs']:
        if not control_driver.completed_job(metadata):
            missing.append(metadata['id'])
        else:
            path = Path(metadata['output_dir'])/'controlled_training_complete.json'
            evidence.append(dict(job_id=metadata['id'],path=str(path),sha256=sha256(path)))
    if len(evidence)+len(missing)!=10:
        raise ValueError('The shared full-data endpoint requires six tuning and four confirmation fits')
    return dict(ready=not missing,missing=missing,evidence=evidence)


def verify_prepared():
    directory = WORK/'learning_curves'
    path = directory/'plan.json'
    plan = json.loads(path.read_text())
    primary_path = WORK/'controlled/plan.json'
    primary = json.loads(primary_path.read_text())
    ready = json.loads((directory/'preparation_complete.json').read_text())
    qualified = json.loads((directory/'qualified_data_views.json').read_text())
    if ready['plan_sha256']!=sha256(path) or qualified['plan_sha256']!=sha256(path):
        raise ValueError('Curve preparation/data qualification belongs to another plan')
    if plan['primary_plan_sha256']!=sha256(primary_path) or plan['spec_sha256']!=sha256(ROOT/'Paired input ranking generalization.md'):
        raise ValueError('Curve parent plan or MD changed')
    if primary['gpu_uuids']!=list(control_plan.GPU_UUIDS):
        raise ValueError('Authorized GPU set changed')
    if plan['fractions']!=list(FRACTIONS) or len(plan['tuning_jobs'])!=18:
        raise ValueError('Curve execution must cover the prepared grid')
    if {r['regime'] for r in qualified['views']}!={'A','B'}:
        raise ValueError('Both production cache views must have passed CPU qualification')
    for views in plan['views'].values():
        for item in views.values():
            path = Path(item['path'])
            if sha256(path)!=item['sha256']:
                raise ValueError('Frozen curve view changed')
            record = json.loads(path.read_text())
            for key in ('indices','labels','source_metadata'):
                if sha256(Path(record[key]['path']))!=record[key]['sha256']:
                    raise ValueError('Frozen curve view payload changed')
    return plan,primary


def run_next():
    import fcntl
    if ROOT!=Path('/home/atabey/Prot2Mol'):
        raise RuntimeError('Scientific curve training is restricted to Atum Prot2Mol')
    lock = (WORK/'controlled/driver.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan,primary = verify_prepared()
    endpoint = full_endpoint_gate(primary)
    if not endpoint['ready']:
        return dict(phase='waiting_for_shared_full_data_endpoint',**endpoint)
    gate = control_driver.baseline_gate(WORK,primary)
    if not gate['ready']:
        return dict(phase='waiting_for_cheap_baselines',missing=gate['missing'])
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    busy = control_driver.busy_authorized_devices(processes)
    if busy:
        return dict(phase='waiting_for_authorized_gpus',live_processes=busy)
    directory = WORK/'learning_curves'
    free = shutil.disk_usage(directory).free
    if free<30*1024**3:
        return dict(phase='waiting_for_storage',bytes_free=free,minimum_bytes=30*1024**3)
    job = next_job(directory,plan)
    if job is None:
        return dict(phase='curve_training_complete_test_comparisons_pending',shared_endpoint=endpoint)
    result = control_driver.execute_job(job,primary,gate)
    result['shared_full_endpoint'] = endpoint
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-fraction',type=float,choices=FRACTIONS[:-1])
    args = parser.parse_args()
    result = confirm_fraction(args.confirm_fraction) if args.confirm_fraction is not None else run_next()
    if args.confirm_fraction is None:
        write_json(WORK/'learning_curves/driver_status.json',result)
    print(json.dumps(result,indent=2))
