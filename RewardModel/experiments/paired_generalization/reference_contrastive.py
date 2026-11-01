"""Finish the matched A reference's protected-checkpoint contrastive audits.

Validation and the original fixed training probe only; no test data, fitting,
checkpoint selection or hyperparameter selection. The fifteen-epoch training
and managed retention must finish before this separate GPU inference job.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .control_plan import GPU_UUIDS, digest_json
    from .control_driver import busy_authorized_devices, launch_environment, verify_training_data
    from .postselection import registered_job, immutable_json
    from .protect_trajectory import REFERENCE, verify_complete
    from .checkpoint_evaluation import parameter_digest, read_native_predictions
    from .export_predictions import score_dataset
    from .contrastive_audit import candidate_pool, directional_losses
    from .label_precision import trainer_label_frame
    from .realization_protocol import IDENTITY, LABELS
    from .scorer_agreement import group_agreement, summarize
else:
    from run import ROOT, WORK, sha256, write_json
    from control_plan import GPU_UUIDS, digest_json
    from control_driver import busy_authorized_devices, launch_environment, verify_training_data
    from postselection import registered_job, immutable_json
    from protect_trajectory import REFERENCE, verify_complete
    from checkpoint_evaluation import parameter_digest, read_native_predictions
    from export_predictions import score_dataset
    from contrastive_audit import candidate_pool, directional_losses
    from label_precision import trainer_label_frame
    from realization_protocol import IDENTITY, LABELS
    from scorer_agreement import group_agreement, summarize


def read(path):
    return json.loads(Path(path).read_text())


def check_policy(config):
    """The existing direction evaluator implements precisely this reference."""
    model,data,training = config.model,config.data,config.training
    if (model.contrastive_active_threshold != 6. or not model.contrastive_strict_active_only or
        model.contrastive_mask_policy != 'reference' or model.cosine_scale_init != 13. or
        model.pair_scoring_mode != 'mlp_contrastive' or model.ranking_readout != 'ranking_head' or
        data.ranking_max_ligands != 16 or data.evaluation_ranking_partitions != 3 or
        data.ranking_min_pchembl_span != .5 or training.per_device_train_batch_size != 12 or
        training.seed != 42):
        raise ValueError('Contrastive audit requires the fixed matched A reference policy')


def verify_loaded_config(model, launch_config):
    """Allow encoder-resolved widths, while rejecting every semantic change.

    RewardModel fills missing widths from its loaded encoder bundles before
    checkpoint serialization. This does not change the launched architecture or
    objectives; explicit widths and every other configuration field stay exact.
    """
    actual = model.config.to_dict()
    expected = launch_config.to_dict()
    for side in ('protein', 'molecule'):
        key = side+'_hidden_size'
        observed = getattr(model, side+'_encoder').config.hidden_size
        if actual[key] != observed:
            raise ValueError(f'Saved {key} differs from its loaded encoder')
        if expected[key] is None:
            expected[key] = observed
    differing = sorted(key for key in actual.keys() | expected.keys() if actual.get(key) != expected.get(key))
    if differing:
        raise ValueError(f'Protected model configuration differs from its launch: {differing}')


def build_plan(directory):
    from importlib.metadata import version
    from reward_model.training.config import load_reward_training_config
    directory = Path(directory)
    primary = read(directory/'plan.json')
    job = registered_job(directory,REFERENCE,primary)
    if job['regime'] != 'A' or job['arm'] != 'full' or job['seed'] != 42 or job['learning_rate_multiplier'] != 1.:
        raise ValueError('Only the original matched A reference is accepted')
    config = load_reward_training_config(job['config_path'])
    check_policy(config)
    if config.training.num_train_epochs != 15 or config.training.fixed_train_eval_assays != 2048:
        raise ValueError('Expected the full horizon and original training probe')
    if not config.training.bf16 or config.training.fp16 or not config.training.dynamic_padding:
        raise ValueError('Exporter precision/padding must match the reference training')
    protected = directory/'protected_trajectories'/REFERENCE
    protection = verify_complete(protected.resolve())
    run = Path(job['output_dir'])
    if protection['config_sha256'] != job['config_sha256'] or len(protection['epochs']) != 15:
        raise ValueError('Protected trajectory belongs to a different configuration')
    checkpoints = []
    for record in protection['epochs']:
        sources = {}
        for prefix in ('eval','train_eval'):
            path = run/'prediction_exports/trajectory'/f"{prefix}_step{record['step']:08d}.json"
            native = read(path)
            if (native['phase'] != 'trajectory' or native['epoch'] != record['epoch'] or
                native['step'] != record['step'] or native['metric_prefix'] != prefix or
                native['ranking_readout'] != config.model.ranking_readout):
                raise ValueError('Native snapshot does not match its protected checkpoint')
            sources[prefix] = dict(path=str(path),sha256=sha256(path),fingerprint=native['dataset_fingerprint'])
        checkpoints.append(dict(**record,native=sources))
    sources = sorted((ROOT/'RewardModel/reward_model').rglob('*.py'))
    sources += [Path(__file__), *(Path(__file__).with_name(name) for name in
        ('run.py','control_plan.py','control_driver.py','postselection.py','retain_controlled.py',
         'protect_trajectory.py','checkpoint_evaluation.py','export_predictions.py','contrastive_audit.py',
         'label_precision.py','realization_protocol.py','scorer_agreement.py','analyze_predictions.py','metrics.py'))]
    software = dict(source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sources},
        packages={name:version(name) for name in ('torch','transformers','datasets','numpy','pandas','scipy')})
    return dict(job=job,primary_plan_sha256=sha256(directory/'plan.json'),
        training_complete_sha256=sha256(run/'controlled_training_complete.json'),
        protection_sha256=sha256(protected/'complete.json'),checkpoints=checkpoints,software=software,
        test_data_used=False,interpretation='All fifteen reference A checkpoints, validation and original fixed training probe; learned versus initial scale on identical candidate pools. No test access or selection.')


def analyze_export(output, native_path, config, fingerprint):
    """Recompute directional losses and quantify single-device/native drift."""
    check_policy(config)
    record = read(native_path)
    if (record['phase'] != 'trajectory' or record['metric_prefix'] not in ('eval','train_eval') or
        record['dataset_fingerprint'] != fingerprint):
        raise ValueError('Embedding audit cannot change partition or dataset')
    frame = trainer_label_frame(pd.read_parquet(output/'predictions.parquet'))
    native,record = read_native_predictions(native_path)
    native = trainer_label_frame(native)
    if record['metric_prefix'] not in ('eval','train_eval') or record['dataset_fingerprint'] != fingerprint:
        raise ValueError('Embedding audit cannot change partition or dataset')
    pd.testing.assert_frame_equal(frame[IDENTITY+LABELS],native[IDENTITY+LABELS],check_dtype=False,check_exact=True)
    if not np.array_equal(frame.row_index,np.arange(len(frame))):
        raise ValueError('Embedding export lost observation order')
    protein = torch.from_numpy(np.load(output/'protein.npy',allow_pickle=False))
    molecule = torch.from_numpy(np.load(output/'molecule.npy',allow_pickle=False))
    if (protein.shape != molecule.shape or protein.ndim != 2 or protein.shape[0] != len(frame) or
        not torch.isfinite(protein).all() or not torch.isfinite(molecule).all()):
        raise ValueError('Embedding export is incomplete or nonfinite')
    scale = read(output/'export.json')['score_scale']
    if not np.isfinite(scale) or scale <= 0 or not np.isclose(scale,record['contrastive_score_scale'],rtol=1e-6,atol=1e-6):
        raise ValueError('Exported logit scale differs from the native snapshot')
    pools,pool_hash = candidate_pool(frame,batch_size=config.training.per_device_train_batch_size,
        seed=config.training.seed,partitions=config.data.evaluation_ranking_partitions,max_ligands=config.data.ranking_max_ligands)
    losses = directional_losses(protein,molecule,pools,dict(learned_scale=scale,fixed_initial_scale13=13.))
    reference = record['metrics'][record['metric_prefix']+'_contrastive_loss']
    difference = losses['learned_scale']['total']-reference
    # Same material-drift criterion used by the original B checkpoint audit.
    if abs(difference) > .01:
        raise ValueError(f'Exported contrastive loss differs materially from native evaluation: {difference}')
    assay_ids = sorted({x.split('__',1)[1] for x in frame.group_id})
    agreement = summarize(group_agreement(frame,dict(zip(assay_ids,range(len(assay_ids))))))
    macro = agreement['original_ranking_eligible']['ranking_score__pchembl_value']['macro_spearman']
    native_macro = record['metrics'][record['metric_prefix']+'_macro_spearman']
    if macro is None or native_macro is None or abs(macro-native_macro) > .001:
        raise ValueError('Exported ranking differs materially from the native snapshot')
    scorer_drift = {}
    for name in ('ranking_score','activity_logits','cosine_similarity'):
        delta = frame[name].to_numpy()-native[name].to_numpy()
        if not np.isfinite(delta).all():
            raise ValueError('Nonfinite exported scorer')
        scorer_drift[name] = dict(max_abs=float(np.abs(delta).max()),mean_abs=float(np.abs(delta).mean()))
    active = frame.pchembl_value > 6.
    return dict(rows=len(frame),epoch=record['epoch'],step=record['step'],prefix=record['metric_prefix'],
        candidate_pool_sha256=pool_hash,losses=losses,score_scale=scale,
        native_contrastive_loss=reference,export_minus_native_contrastive_loss=difference,
        export_macro_spearman=macro,native_macro_spearman=native_macro,scorer_drift=scorer_drift,
        positive_alignment=dict(rows=int(active.sum()),mean_cosine=float(frame.loc[active,'cosine_similarity'].mean()) if active.any() else None),
        agreement=agreement,
        interpretation='Three fixed complete-coverage candidate partitions, capped at16 molecules/list and12 lists/batch. Scale13 rescoring isolates the numerical temperature contribution, not fixed-temperature retraining. Training-probe statistics are descriptive. Single-device bf16 inference may differ slightly from native four-rank batching; drift is explicitly bounded and reported.')


def verify_export(output, source, *, cpu=False):
    result = read(output/'complete.json')
    if result['source'] != source or result['cpu_qualification'] != cpu or not result['parameters_unchanged']:
        raise ValueError('Embedding audit source or execution mode changed')
    for name,digest in result['artifacts_sha256'].items():
        if sha256(output/name) != digest:
            raise ValueError('Embedding audit artifact changed')
    if not cpu:
        native = source['checkpoint']['native'][source['prefix']]
        if sha256(Path(native['path'])) != native['sha256']:
            raise ValueError('Native reference manifest changed')
        read_native_predictions(native['path'])  # Verify the referenced payload as well.
    return result


def evaluate_checkpoint(model, dataset, config, native_path, output, source, *, cpu=False):
    check_policy(config)
    verify_loaded_config(model, config.model)
    native = read(native_path)
    if native['phase'] != 'trajectory' or native['metric_prefix'] not in ('eval','train_eval'):
        raise ValueError('Reference inference accepts only native validation/training trajectories')
    if native['dataset_fingerprint'] != dataset._fingerprint:
        raise ValueError('Reference inference dataset differs from the native snapshot')
    if output.exists():
        raise RuntimeError('Existing embedding output requires verification or inspection')
    before = parameter_digest(model)
    result = score_dataset(model,dataset,output,batch_size=config.training.per_device_eval_batch_size,
        device='cpu' if cpu else 'cuda',save_embeddings=True)
    result['dtype'] = 'float32 CPU qualification' if cpu else 'bf16 autocast; exported float32'
    write_json(output/'export.json',result)
    analysis = analyze_export(output,native_path,config,dataset._fingerprint)
    if parameter_digest(model) != before:
        raise RuntimeError('Embedding inference changed model weights')
    result = dict(source=source,cpu_qualification=cpu,parameters_unchanged=True,parameter_sha256=before,
        analysis=analysis,artifacts_sha256={p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    immutable_json(output/'complete.json',result)
    return verify_export(output,source,cpu=cpu)


def worker(path, index, workers):
    from datasets import load_from_disk
    from reward_model.model import load_reward_model
    from reward_model.training.config import load_reward_training_config
    from reward_model.training.data import select_fixed_ranking_evaluation_subset
    if ROOT != Path('/home/atabey/Prot2Mol') or not 1<=workers<=4 or not 0<=index<workers:
        raise RuntimeError('Invalid reference audit worker')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUIDS[index]:
        raise RuntimeError('Reference audit worker must use its assigned authorized GPU')
    torch.set_num_threads(4)
    path = Path(path)
    plan = read(path)
    if plan != build_plan(WORK/'controlled'):
        raise ValueError('Reference audit plan changed')
    config = load_reward_training_config(plan['job']['config_path'])
    primary = read(WORK/'controlled/plan.json')
    datasets = {}
    for split in ('val','train'):
        cache = Path(config.data.tokenized_dataset_dir)/f'{split}_examples'
        for name in ('state.json','dataset_info.json'):
            if sha256(cache/name) != primary['tokenized_data_sha256']['A'][split][name]:
                raise ValueError('Token cache registration changed after launch verification')
        datasets[split] = load_from_disk(str(cache))
    datasets['train'] = select_fixed_ranking_evaluation_subset(datasets['train'],num_assays=config.training.fixed_train_eval_assays,
        seed=config.training.seed,min_pchembl_span=config.data.ranking_min_pchembl_span)
    for checkpoint in plan['checkpoints'][index::workers]:
        model = None
        for prefix,split in [('eval','val'),('train_eval','train')]:
            native = checkpoint['native'][prefix]
            if sha256(Path(native['path'])) != native['sha256'] or datasets[split]._fingerprint != native['fingerprint']:
                raise ValueError('Native snapshot or original fixed probe changed')
            output = path.parent/f"epoch{checkpoint['epoch']:02d}"/prefix
            source = dict(plan_sha256=sha256(path),checkpoint=checkpoint,prefix=prefix,gpu_uuid=GPU_UUIDS[index])
            if (output/'complete.json').exists():
                verify_export(output,source)
                continue
            if output.exists():
                raise RuntimeError('Partial reference inference requires inspection')
            if model is None:
                model = load_reward_model(checkpoint['path'],device=torch.device('cpu'))
                verify_loaded_config(model, config.model)
                model.to('cuda')
            evaluate_checkpoint(model,datasets[split],config,Path(native['path']),output,source)
        del model
        torch.cuda.empty_cache()


def run(workers=4):
    import fcntl
    if ROOT != Path('/home/atabey/Prot2Mol') or not 1<=workers<=4:
        raise RuntimeError('Use Atum and at most the four authorized GPUs')
    with (WORK/'controlled/driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        busy = busy_authorized_devices(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True))
        if busy:
            return dict(phase='waiting_for_authorized_gpus',live_processes=busy)
        plan = build_plan(WORK/'controlled')
        # Hash all source data and token payloads once per invocation, before
        # launching readers. Workers check the frozen registration and identities.
        verify_training_data(read(WORK/'controlled/plan.json'),'A')
        output = WORK/'tier1/reference_contrastive'/REFERENCE/digest_json(plan['software'])[:16]
        path = output/'plan.json'
        immutable_json(path,plan)
        children,handles = [],[]
        try:
            for index in range(workers):
                env = launch_environment(REFERENCE)
                env.update(CUDA_VISIBLE_DEVICES=GPU_UUIDS[index],WANDB_MODE='disabled')
                handle = (output/f'worker{index}.log').open('a');handles.append(handle)
                command = [str(ROOT/'.venv/bin/python'),str(Path(__file__)),'worker','--plan',str(path),'--index',str(index),'--workers',str(workers)]
                children.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT))
            write_json(output/'status.json',dict(phase='reference_contrastive_inference',driver_pid=os.getpid(),worker_pids=[p.pid for p in children],started_unix=time.time()))
            codes = [p.wait() for p in children]
            if any(codes):
                raise RuntimeError(f'Reference inference failed; inspect logs before retry: {codes}')
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate();child.wait()
            for handle in handles:
                handle.close()
        results,pools = [],{}
        for index in range(workers):
            for checkpoint in plan['checkpoints'][index::workers]:
                for prefix in ('eval','train_eval'):
                    directory = output/f"epoch{checkpoint['epoch']:02d}"/prefix
                    source = dict(plan_sha256=sha256(path),checkpoint=checkpoint,prefix=prefix,gpu_uuid=GPU_UUIDS[index])
                    result = verify_export(directory,source)
                    pool = result['analysis']['candidate_pool_sha256']
                    if pools.setdefault(prefix,pool) != pool:
                        raise ValueError('Candidate pools differ between epochs')
                    results.append(dict(path=str(directory),sha256=sha256(directory/'complete.json'),analysis=result['analysis']))
        summary = dict(plan_sha256=sha256(path),exports=len(results),trajectory=sorted(results,key=lambda r:(r['analysis']['epoch'],r['analysis']['prefix'])),candidate_pools=pools,test_data_used=False)
        immutable_json(output/'complete.json',summary)
        write_json(output/'status.json',dict(phase='reference_contrastive_complete',exports=len(results)))
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action',required=True)
    launch = commands.add_parser('run');launch.add_argument('--workers',type=int,default=4)
    child = commands.add_parser('worker')
    child.add_argument('--plan',type=Path,required=True);child.add_argument('--index',type=int,required=True);child.add_argument('--workers',type=int,required=True)
    args = parser.parse_args()
    if args.action == 'run':
        result = run(args.workers)
        print(json.dumps({k:result[k] for k in ('phase','exports','test_data_used') if k in result}))
    else:
        worker(args.plan,args.index,args.workers)
