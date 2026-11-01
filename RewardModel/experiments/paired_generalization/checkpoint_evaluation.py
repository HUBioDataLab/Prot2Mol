"""Evaluate fixed model weights through the production Trainer, without training.

Selection and GPU scheduling are the caller's responsibility. This module
preserves the training configuration's evaluation pools, labels and readout.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .run import sha256, write_json
else:
    from run import sha256, write_json

from reward_model.training.data import RewardAssayListCollator, RewardEvaluationDataset
from reward_model.training.trainer import RewardModelTrainer, create_training_arguments


def parameter_digest(model):
    digest = hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype,tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def read_native_predictions(manifest):
    manifest = Path(manifest)
    record = json.loads(manifest.read_text())
    payload = manifest.parent/record['predictions_file']
    if payload.parent.resolve()!=manifest.parent.resolve() or sha256(payload)!=record['predictions_sha256']:
        raise ValueError('Native evaluation payload differs from its manifest')
    with np.load(payload,allow_pickle=False) as arrays:
        if not np.array_equal(arrays['row_index'],np.arange(record['rows'])):
            raise ValueError('Native evaluation does not cover each row in order')
        import pandas as pd
        frame = pd.DataFrame({key:arrays[key].copy() for key in arrays.files})
    return frame,record


def evaluate_fixed_model(model, examples, config, output, *, prefix='test',
                         known_positive_index=None, source=None, cpu=False):
    """One immutable evaluation pass; never construct an optimizer or call train."""
    if prefix not in ('eval','test'):
        raise ValueError('Expected a validation or test evaluation')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise ValueError('Use one independent process/GPU per fixed-weight evaluation')
    if model.config.contrastive_mask_policy=='training_known' and known_positive_index is None:
        raise ValueError('Training-known evaluation must receive the original training-positive index')
    if model.config.ranking_min_pchembl_span!=config.data.ranking_min_pchembl_span:
        raise ValueError('Evaluation eligibility differs from the model configuration')
    output = Path(output)
    if output.exists():
        raise RuntimeError('Existing evaluation requires verification/inspection before reuse')
    before = parameter_digest(model)
    training = replace(config.training,output_dir=str(output),training_mode='single_gpu',
        report_to=[],fixed_train_eval_assays=0,export_evaluation_predictions=True,
        objective_gradient_diagnostics_steps=0)
    if cpu:
        # Explicit qualification path; scientific GPU evaluation keeps precision.
        training = replace(training,bf16=False,fp16=False,dataloader_num_workers=0)
    args = create_training_arguments(training)
    if cpu and args.device.type!='cpu':
        raise ValueError('CPU qualification requires a CPU-only process')
    dataset = RewardEvaluationDataset(examples,
        ranking_min_pchembl_span=config.data.ranking_min_pchembl_span,
        ranking_max_ligands=config.data.ranking_max_ligands,
        ranking_num_partitions=config.data.evaluation_ranking_partitions,
        ranking_partition_seed=config.training.seed,
        protein_shuffle_sensitivity=config.training.protein_shuffle_sensitivity)
    collator = RewardAssayListCollator(known_positive_index=known_positive_index,
        dynamic_padding=config.training.dynamic_padding,
        protein_pad_token_id=getattr(model.protein_tokenizer,'pad_token_id',0),
        molecule_pad_token_id=getattr(model.molecule_tokenizer,'pad_token_id',0))
    trainer = RewardModelTrainer(model=model,args=args,eval_dataset=dataset,data_collator=collator)
    trainer.prediction_export_phase = 'fixed_checkpoint'
    metrics = trainer.evaluate(eval_dataset=dataset,metric_key_prefix=prefix)
    if trainer.optimizer is not None or trainer.state.global_step!=0:
        raise RuntimeError('Fixed-weight evaluation unexpectedly initialized training')
    after = parameter_digest(model)
    if after!=before:
        raise RuntimeError('Evaluation changed model weights/buffers')
    manifest = output/'prediction_exports/fixed_checkpoint'/f'{prefix}_step00000000.json'
    frame,record = read_native_predictions(manifest)
    if len(frame)!=len(examples) or record['ranking_readout']!=model.config.ranking_readout:
        raise ValueError('Native evaluation changed row count or ranking readout')
    result = dict(format_version=1,source=source,rows=len(examples),prefix=prefix,
        metrics=record['metrics'],parameter_sha256=before,parameters_unchanged=True,
        optimizer_created=False,world_size=1,device=str(args.device),cpu_qualification=cpu,
        ranking_partition_seed=config.training.seed,
        ranking_num_partitions=config.data.evaluation_ranking_partitions,
        contrastive_assay_batch_size=training.per_device_train_batch_size,
        bf16=training.bf16,fp16=training.fp16,
        known_positive_index=known_positive_index.provenance if known_positive_index is not None else None,
        prediction_manifest=str(manifest),prediction_manifest_sha256=sha256(manifest),
        prediction_payload_sha256=record['predictions_sha256'],
        interpretation='Native fixed-weight evaluation. Single-device inference can have different floating-point batching from four-rank training; saved predictions define these reported metrics. No checkpoint or hyperparameter selection occurs here.')
    write_json(output/'complete.json',result)
    del trainer
    return result
