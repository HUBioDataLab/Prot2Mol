"""Support-only fitting worker for the MD's staged head-retraining probes.

This is not a scientific job launcher or a selection protocol. Callers must
verify Tier 3, the selected base model, support views and GPU scheduling first.
No query examples enter this interface. Validation selection and query scoring
use saved parameter subsets in separate calls, with an unchanged base model.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from transformers import TrainerCallback

if __package__:
    from .run import ROOT, sha256, write_json
    from .control_plan import GPU_UUIDS
    from .checkpoint_evaluation import parameter_digest
    from .objective_probes import PRIVATE_ROOTS, parameter_scope
else:
    from run import ROOT, sha256, write_json
    from control_plan import GPU_UUIDS
    from checkpoint_evaluation import parameter_digest
    from objective_probes import PRIVATE_ROOTS, parameter_scope

from reward_model.training.data import RewardAssayListDataset, RewardAssayListCollator
from reward_model.training.trainer import RewardModelTrainer, create_training_arguments

SCOPES = ('heads', 'heads_cross_attention')
CROSS_ATTENTION_ROOTS = {'fusion', 'extra_fusion_layers'}


def scope_parameters(model, scope):
    if scope not in SCOPES or model.config.pair_scoring_mode != 'mlp_contrastive':
        raise ValueError('Expected a registered head scope and mlp_contrastive model')
    allowed = PRIVATE_ROOTS | (CROSS_ATTENTION_ROOTS if scope == 'heads_cross_attention' else set())
    names = []
    for name, parameter in model.named_parameters():
        parameter_scope(name)  # Reject architecture additions lacking a scope.
        if name.split('.')[0] in allowed and parameter.requires_grad:
            names.append(name)
    if not names:
        raise ValueError('Adaptation has no active trainable head parameters')
    return sorted(names)


def state_digest(model, excluded=()):
    digest = hashlib.sha256()
    excluded = set(excluded)
    for name, tensor in sorted(model.state_dict().items()):
        if name in excluded:
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def adaptation_scope(model, scope):
    """Keep frozen feature modules in eval mode even when Trainer calls train."""
    names = scope_parameters(model, scope)
    parameters = dict(model.named_parameters())
    original_flags = {name: p.requires_grad for name, p in parameters.items()}
    original_modes = {module: module.training for module in model.modules()}
    original_kept = model._encoder_modules_kept_in_eval
    try:
        for name, parameter in parameters.items():
            parameter.requires_grad_(name in names)
            parameter.grad = None
        # Shared input projections, normalizations and projection_dropout stay
        # fixed in both arms. Only the second arm also trains the fusion blocks.
        kept = [module for _, module in model.named_children()
                if not any(p.requires_grad for p in module.parameters())]
        model._encoder_modules_kept_in_eval = tuple(kept)
        model.train(model.training)
        yield names
    finally:
        for name, parameter in parameters.items():
            parameter.requires_grad_(original_flags[name])
            parameter.grad = None
        model._encoder_modules_kept_in_eval = original_kept
        for module, mode in original_modes.items():
            module.training = mode


def restore_adapter(model, path, expected_sha256):
    """Restore only a verified parameter subset into its exact unchanged base."""
    path = Path(path)
    if sha256(path) != expected_sha256:
        raise ValueError('Adaptation payload checksum differs')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['format_version'] != 1 or parameter_digest(model) != payload['base_parameter_sha256']:
        raise ValueError('Adaptation payload requires its exact unchanged base model')
    names = scope_parameters(model, payload['scope'])
    if sorted(payload['parameters']) != names:
        raise ValueError('Adaptation payload contains a different trainable scope')
    frozen = state_digest(model, names)
    if frozen != payload['frozen_parameter_sha256']:
        raise ValueError('Frozen base parameters differ')
    parameters = dict(model.named_parameters())
    # Validate every tensor before making the first mutation.
    for name, tensor in payload['parameters'].items():
        if (tensor.shape != parameters[name].shape or tensor.dtype != parameters[name].dtype
                or not torch.isfinite(tensor).all()):
            raise ValueError('Invalid adaptation tensor: '+name)
    with torch.no_grad():
        for name, tensor in payload['parameters'].items():
            parameters[name].copy_(tensor)
    if state_digest(model, names) != frozen:
        raise RuntimeError('Adapter restoration changed frozen state')
    return payload


class AdapterTrajectory(TrainerCallback):
    def __init__(self, model, names, scope, base_digest, frozen_digest, output, epochs):
        self.model, self.names, self.scope = model, names, scope
        self.base_digest, self.frozen_digest = base_digest, frozen_digest
        self.output, self.epochs = Path(output), set(epochs)
        self.records, self.optimizer_groups = [], []

    def on_train_begin(self, args, state, control, optimizer=None, **kwargs):
        parameters = dict(self.model.named_parameters())
        expected = {id(parameters[name]) for name in self.names}
        actual = [id(p) for group in optimizer.param_groups for p in group['params']]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise RuntimeError('Optimizer does not contain exactly the adaptation parameters')
        for group in optimizer.param_groups:
            group_ids = {id(p) for p in group['params']}
            self.optimizer_groups.append(dict(
                names=[name for name in self.names if id(parameters[name]) in group_ids],
                learning_rate=float(group['lr']), weight_decay=float(group['weight_decay'])))

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = round(state.epoch)
        if not math.isclose(state.epoch, epoch, abs_tol=1e-6):
            raise ValueError('Adaptation must finish whole support epochs')
        if epoch not in self.epochs:
            return
        if any(record['epoch'] == epoch for record in self.records):
            raise ValueError('An adaptation epoch was saved twice')
        parameters = dict(self.model.named_parameters())
        weights = {name: parameters[name].detach().cpu().clone() for name in self.names}
        if any(not torch.isfinite(tensor).all() for tensor in weights.values()):
            raise ValueError('Adaptation produced nonfinite parameters')
        payload = dict(format_version=1, scope=self.scope, epoch=epoch, step=state.global_step,
                       base_parameter_sha256=self.base_digest,
                       frozen_parameter_sha256=self.frozen_digest, parameters=weights)
        path = self.output/f'epoch{epoch:02d}.pt'
        with path.open('xb') as handle:
            torch.save(payload, handle)
        restored = torch.load(path, map_location='cpu', weights_only=True)
        if any(not torch.equal(restored['parameters'][name], weights[name]) for name in weights):
            raise RuntimeError('Adapter payload failed tensor round-trip')
        self.records.append(dict(epoch=epoch, step=state.global_step, file=path.name,
                                 sha256=sha256(path), bytes=path.stat().st_size))


def fit_support(model, examples, config, output, *, scope, base_parameter_sha256,
                source=None, known_positive_index=None, save_epochs=None, cpu=False):
    """Fit only support data for a fixed horizon; no evaluation or selection."""
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Each adaptation task uses one independent process/device')
    if cpu:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('CPU qualification requires hidden CUDA devices')
    elif ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') not in GPU_UUIDS:
        raise ValueError('Scientific adaptation requires one authorized Atum GPU UUID')
    if len(examples) == 0 or config.model != model.config:
        raise ValueError('Expected nonempty support and the exact base model configuration')
    if model.config.ranking_min_pchembl_span != config.data.ranking_min_pchembl_span:
        raise ValueError('Model and support ranking eligibility differ')
    if model.config.contrastive_mask_policy == 'training_known' and known_positive_index is None:
        raise ValueError('Training-known masking requires a caller-verified support-safe positive index')
    before = parameter_digest(model)
    if before != base_parameter_sha256:
        raise ValueError('Every task must start from the exact unchanged base model')
    horizon = int(config.training.num_train_epochs)
    if horizon < 1 or horizon != config.training.num_train_epochs or config.training.max_steps is not None:
        raise ValueError('Adaptation requires a fixed whole-epoch horizon, without max_steps')
    epochs = list(range(1, horizon+1)) if save_epochs is None else list(save_epochs)
    if (not epochs or epochs != sorted(set(epochs)) or horizon not in epochs
            or any(not isinstance(e, int) or not 1 <= e <= horizon for e in epochs)):
        raise ValueError('Save epochs must be distinct increasing whole epochs including the horizon')
    output = Path(output)
    if output.exists():
        raise RuntimeError('Existing adaptation output requires inspection; never resume implicitly')
    training = replace(config.training, output_dir=str(output), training_mode='single_gpu',
                       fixed_train_eval_assays=0, export_evaluation_predictions=False,
                       objective_gradient_diagnostics_steps=0, evaluate_test=False)
    if cpu:
        training = replace(training, bf16=False, fp16=False, dataloader_num_workers=0, report_to=[])
    args = create_training_arguments(training)
    args.load_best_model_at_end = False
    args.eval_strategy = 'no'
    args.save_strategy = 'no'
    args.do_eval = False
    if cpu and args.device.type != 'cpu':
        raise ValueError('Qualification must execute on CPU')
    if not cpu and (args.device.type != 'cuda' or torch.cuda.device_count() != 1):
        raise ValueError('Scientific adaptation requires exactly one visible CUDA device')
    dataset = RewardAssayListDataset(examples, seed=training.seed,
        ranking_max_ligands=config.data.ranking_max_ligands,
        ranking_opportunity_divisor=config.data.ranking_opportunity_divisor,
        ranking_min_pchembl_span=config.data.ranking_min_pchembl_span,
        max_classification_only_per_item=config.data.max_classification_only_per_item,
        include_all_assays_for_contrastive=True, item_count_multiple=1,
        balance_ranking_assays=config.model.ranking_balance_assays)
    collator = RewardAssayListCollator(known_positive_index=known_positive_index,
        dynamic_padding=training.dynamic_padding,
        protein_pad_token_id=getattr(model.protein_tokenizer, 'pad_token_id', 0),
        molecule_pad_token_id=getattr(model.molecule_tokenizer, 'pad_token_id', 0))
    with adaptation_scope(model, scope) as names:
        frozen = state_digest(model, names)
        output.mkdir(parents=True)
        write_json(output/'fit_plan.json', dict(scope=scope, source=source,
            base_parameter_sha256=before, frozen_parameter_sha256=frozen,
            trainable_parameters=names, model_config=model.config.to_dict(),
            training=asdict(training), data=asdict(config.data), save_epochs=epochs,
            worker_source_sha256=sha256(Path(__file__)),
            support_fingerprint=examples._fingerprint,
            dropout_policy='Frozen top-level feature modules, including shared projection_dropout, remain in eval mode; trainable heads and optional fusion blocks follow Trainer mode.',
            query_examples_received=False, selection_performed=False,
            support_rows=len(examples), support_assays=len(set(examples['group_id'])),
            cpu_qualification=cpu, known_positive_index=(known_positive_index.provenance
                if known_positive_index is not None else None)))
        trajectory = AdapterTrajectory(model, names, scope, before, frozen, output, epochs)
        trainer = RewardModelTrainer(model=model, args=args, train_dataset=dataset,
                                     data_collator=collator, callbacks=[trajectory])
        result = trainer.train()
        if (not math.isclose(trainer.state.epoch, horizon, abs_tol=1e-6)
                or [r['epoch'] for r in trajectory.records] != epochs):
            raise RuntimeError('Adaptation did not finish the declared trajectory')
        if any(any(k.startswith('eval_') or k.startswith('test_') for k in record)
               for record in trainer.state.log_history):
            raise RuntimeError('Support fitting unexpectedly performed query evaluation')
        if state_digest(model, names) != frozen:
            raise RuntimeError('Adaptation changed frozen parameters or buffers')
        record = dict(format_version=1, scope=scope, source=source,
            fit_plan_sha256=sha256(output/'fit_plan.json'), base_parameter_sha256=before,
            final_parameter_sha256=parameter_digest(model), frozen_parameters_unchanged=True,
            trainable_parameters=names, trainable_elements=sum(
                p.numel() for name, p in model.named_parameters() if name in names),
            optimizer_groups=trajectory.optimizer_groups, epochs=horizon,
            global_step=trainer.state.global_step, snapshots=trajectory.records,
            support_rows=len(examples), support_assays=len(set(examples['group_id'])),
            sampler_stats=asdict(dataset.stats), training_metrics=result.metrics,
            training_history=trainer.state.log_history, cpu_qualification=cpu,
            query_examples_received=False, selection_performed=False,
            interpretation='Fixed-horizon support-only fit. Compact weights require the identical base checkpoint. No adaptation ranking result, selected epoch or complete scientific experiment follows from fitting alone.')
        write_json(output/'complete.json', record)
        del trainer
    return record


def qualify_cpu(output):
    """Tiny synthetic execution only, without loading any scientific checkpoint."""
    from datasets import Dataset
    if __package__:
        from .qualify_controls import make_model, make_batch
    else:
        from qualify_controls import make_model, make_batch
    from reward_model.training.config import RewardTrainerConfig, RewardTrainingConfigBundle, RewardTrainingDataConfig
    output = Path(output)
    if output.exists():
        raise RuntimeError('Qualification output already exists')
    records = []
    for scope in SCOPES:
        model = make_model(None, {})
        base = parameter_digest(model)
        batch = make_batch(model, torch.device('cpu'), False)
        rows = [dict(group_id=f'p{i//3}__a{i//3}', assay_id=f'a{i//3}',
            target_chembl_id=f'p{i//3}', compound_id=f'm{i}', activity_type='IC50',
            pchembl_value=float(batch['pchembl_values'][i]), binary_label=int(batch['activity_labels'][i]),
            **{key:batch[key][i].tolist() for key in ('protein_input_ids', 'protein_attention_mask',
                'molecule_input_ids', 'molecule_attention_mask')}) for i in range(9)]
        config = RewardTrainingConfigBundle(model=model.config,
            data=RewardTrainingDataConfig(train_parquet_path='unused_support', val_parquet_path='unused_query',
                test_parquet_path='unused_test_query',
                tokenized_dataset_dir='unused_cache', enforce_binary_label_threshold=False, sample_all_assays=True),
            training=RewardTrainerConfig(output_dir='unused_output', num_train_epochs=2,
                per_device_train_batch_size=2, gradient_accumulation_steps=2,
                learning_rate=.001, logging_steps=1, evaluate_test=False))
        record = fit_support(model, Dataset.from_list(rows), config, output/scope, scope=scope,
                             base_parameter_sha256=base, cpu=True, source={'synthetic':True})
        fresh = make_model(None, {})
        snapshot = record['snapshots'][-1]
        restore_adapter(fresh, output/scope/snapshot['file'], snapshot['sha256'])
        if parameter_digest(fresh) != record['final_parameter_sha256']:
            raise RuntimeError('Compact restoration did not reproduce fitted parameters')
        records.append(dict(scope=scope, complete_sha256=sha256(output/scope/'complete.json'),
                            steps=record['global_step'], restored_parameters_match=True))
    result = dict(cpu_qualification=True, scientific_results=False, scopes=records)
    write_json(output/'qualification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify-cpu', action='store_true', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(qualify_cpu(args.output), indent=2))
