"""Numerical worker for MD E56/E57; not a scientific batch/job launcher.

Per-block gradients use actual trainable parameters, not separate pathway
outputs. TAG-style steps are stateless weighted-SGD counterfactuals evaluated
on a separate, fixed validation batch. No optimizer or trained weights change.
Scientific callers must verify Tier3 completion, checkpoint/data provenance,
fixed training/B-validation batches, step sizes and the shared GPU lock.
"""
from __future__ import annotations

from contextlib import nullcontext
import itertools
import math
import os
from pathlib import Path
import re

import torch

if __package__:
    from .run import ROOT
    from .control_plan import GPU_UUIDS
    from .checkpoint_evaluation import parameter_digest
else:
    from run import ROOT
    from control_plan import GPU_UUIDS
    from checkpoint_evaluation import parameter_digest

OBJECTIVES = dict(R=('ranking_loss', 'ranking_loss_weight'),
                  B=('classification_loss', 'classification_loss_weight'),
                  C=('contrastive_loss', 'contrastive_loss_weight'))
INTERACTION_ROOTS = {'protein_projection', 'molecule_projection', 'protein_norm',
                     'molecule_norm', 'fusion', 'extra_fusion_layers'}
PRIVATE_ROOTS = {'ranking_head', 'classification_head', 'contrastive_protein_projection',
                 'contrastive_molecule_projection', 'logit_scale'}


def parameter_block(name):
    """Disjoint transformer blocks plus explicit projection/fusion/head blocks."""
    root = name.split('.')[0]
    if root in ('protein_encoder', 'molecule_encoder'):
        match = re.match(r'((?:protein|molecule)_encoder\.encoder\.layer\.\d+)\.', name)
        if match:
            return match.group(1)
        if name.startswith(root+'.embeddings.'):
            return root+'.embeddings'
        return root+'.other'
    if root == 'extra_fusion_layers':
        return '.'.join(name.split('.')[:2])
    if root in INTERACTION_ROOTS | PRIVATE_ROOTS:
        return root
    raise ValueError('Unregistered parameter block: '+name)


def parameter_scope(name):
    root = name.split('.')[0]
    if root in ('protein_encoder', 'molecule_encoder'):
        return root
    if root in INTERACTION_ROOTS:
        return 'interaction'
    if root in PRIVATE_ROOTS:
        return 'private'
    raise ValueError('Unregistered parameter scope: '+name)


def objective_values(output, weights):
    result = {}
    for objective, (field, _) in OBJECTIVES.items():
        value = getattr(output, field, None)
        if weights[objective] == 0:
            result[objective] = None
        elif value is None or value.numel() != 1 or not torch.isfinite(value).all():
            raise ValueError('Active objective must have a finite scalar loss: '+objective)
        elif float(value.detach()) < 0:
            raise ValueError('TAG-style relative transfer requires nonnegative losses')
        else:
            result[objective] = value
    return result


def gradient_statistics(parameters, gradients, weights):
    groups = {}
    for name in parameters:
        groups.setdefault(parameter_block(name), []).append(name)
    result = {}
    for block, names in groups.items():
        trainable = [n for n in names if parameters[n].requires_grad]
        record = dict(parameter_tensors=len(names), parameter_elements=sum(parameters[n].numel() for n in names),
                      trainable_elements=sum(parameters[n].numel() for n in trainable), objectives={}, cosines={})
        for objective in OBJECTIVES:
            grads = gradients[objective]
            connected = [grads[n] for n in trainable if grads is not None and grads.get(n) is not None]
            square = sum(float(torch.sum(g.double()**2)) for g in connected)
            norm = math.sqrt(square)
            state = ('disabled' if grads is None else 'frozen' if not trainable else
                     'disconnected' if not connected else 'zero_gradient' if norm == 0 else 'nonzero')
            record['objectives'][objective] = dict(state=state, connected_tensors=len(connected),
                weighted_norm=norm if grads is not None else None,
                raw_norm=norm/weights[objective] if weights[objective] else None)
        for left, right in itertools.combinations(OBJECTIVES, 2):
            ln = record['objectives'][left]['weighted_norm']
            rn = record['objectives'][right]['weighted_norm']
            cosine = None
            if ln and rn:
                dot = sum(float(torch.sum(gradients[left][n].double()*gradients[right][n].double()))
                    for n in trainable if gradients[left].get(n) is not None and gradients[right].get(n) is not None)
                cosine = max(-1., min(1., dot/(ln*rn)))
            record['cosines'][left+'__'+right] = cosine
        result[block] = record
    return result


def measure(model, training_batch, validation_batch, learning_rates, *,
            precision='float32', scopes=('encoders', 'interaction', 'shared')):
    """Return batch-level cosines and actual forward-evaluated transfer ratios.

    Loss weights scale the source gradients. Target loss ratios use raw losses;
    positive target weights cancel in the ratio. Private heads/projections/scale
    and frozen parameters are held fixed. Steps are not AdamW updates or fits.
    """
    if precision not in ('float32', 'bf16'):
        raise ValueError('Expected float32 or bf16 probe precision')
    if set(learning_rates) != {'protein_encoder', 'molecule_encoder', 'interaction'} or any(
        not math.isfinite(float(v)) or float(v) <= 0 for v in learning_rates.values()):
        raise ValueError('Freeze positive finite learning rates for both encoders and interaction')
    if not scopes or len(set(scopes)) != len(scopes) or not set(scopes) <= {'encoders','interaction','shared'}:
        raise ValueError('Unsupported or repeated lookahead scopes')
    if getattr(model.config, 'pair_scoring_mode', None) != 'mlp_contrastive':
        raise ValueError('Probe block map requires the registered mlp_contrastive architecture')
    original = dict(model.named_parameters())
    if not original:
        raise ValueError('Model has no parameters')
    devices = {p.device for p in original.values()}
    if len(devices) != 1:
        raise ValueError('Use one independent device per diagnostic model')
    device = next(iter(devices))
    if device.type not in ('cpu', 'cuda') or (device.type == 'cpu' and precision != 'float32'):
        raise ValueError('CPU qualification uses float32 mode; scientific workers use authorized CUDA')
    if device.type == 'cuda' and (ROOT != Path('/home/atabey/Prot2Mol') or
                                os.environ.get('CUDA_VISIBLE_DEVICES') not in GPU_UUIDS):
        raise RuntimeError('Scientific gradient probes require one authorized Atum GPU')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Per-batch probes require an independent, non-DDP model')
    blocks = {name:parameter_block(name) for name in original}
    weights = {key:float(getattr(model.config, weight)) for key,(_,weight) in OBJECTIVES.items()}
    if any(not math.isfinite(w) or w < 0 for w in weights.values()) or not any(weights.values()):
        raise ValueError('Expected finite nonnegative weights with an active objective')
    before = parameter_digest(model)
    modes = [(module, module.training) for module in model.modules()]
    # All parameter/buffer storage used by the probe is independent of the
    # loaded model, including the initial gradient and validation forwards.
    parameters = {name:p.detach().clone().requires_grad_(p.requires_grad) for name,p in original.items()}
    buffers = {name:b.detach().clone() for name,b in model.named_buffers()}
    trainable = {name:p for name,p in parameters.items() if p.requires_grad}
    gradients, result = {}, None

    def forward(values, batch):
        autocast = torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()
        with autocast:
            output = torch.func.functional_call(model,
                (values, {name:b.clone() for name,b in buffers.items()}), (), dict(batch, return_dict=True), strict=True)
        return objective_values(output, weights)

    try:
        model.eval()  # Disable dropout so before/after differences use the same function.
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
            losses = forward(parameters, training_batch)
            training_losses = {k:float(v.detach()) if v is not None else None for k,v in losses.items()}
            for objective, loss in losses.items():
                if loss is None:
                    gradients[objective] = None
                    continue
                if trainable and loss.requires_grad:
                    values = torch.autograd.grad(loss*weights[objective], tuple(trainable.values()),
                                                 allow_unused=True, retain_graph=True)
                else:
                    values = (None,)*len(trainable)
                if any(g is not None and (g.is_sparse or not torch.isfinite(g).all()) for g in values):
                    raise ValueError('Nonfinite or unsupported sparse diagnostic gradient')
                gradients[objective] = {name:g.detach().cpu() if g is not None else None
                                       for name,g in zip(trainable, values)}
            del losses, loss, values
            statistics = gradient_statistics(parameters, gradients, weights)
            with torch.no_grad():
                values = forward(parameters, validation_batch)
                baseline = {k:float(v) if v is not None else None for k,v in values.items()}
                lookahead = {}
                for source, grads in gradients.items():
                    if grads is None:
                        lookahead[source] = dict(state='disabled', scopes={})
                        continue
                    records = {}
                    for scope in scopes:
                        future = {name:p.detach() for name,p in parameters.items()}
                        changed, delta_square, planned_square = [], 0., 0.
                        for name, gradient in grads.items():
                            category = parameter_scope(name)
                            included = category != 'private' and (scope == 'shared' or
                                (scope == 'encoders' and category != 'interaction') or
                                (scope == 'interaction' and category == 'interaction'))
                            if not included or gradient is None:
                                continue
                            update = gradient.to(device=device, dtype=parameters[name].dtype)*float(learning_rates[category])
                            future[name] = parameters[name].detach()-update
                            delta = future[name]-parameters[name].detach()
                            planned_square += float(torch.sum(update.double()**2))
                            delta_square += float(torch.sum(delta.double()**2))
                            if torch.count_nonzero(delta):
                                changed.append(name)
                        after_values = forward(future, validation_batch) if changed else values
                        after = {k:float(v) if v is not None else None for k,v in after_values.items()}
                        transfer = {}
                        for target, value in baseline.items():
                            transfer[target] = dict(before=value, after=after[target],
                                absolute_loss_change=after[target]-value if value is not None else None,
                                relative_transfer=1.-after[target]/value if changed and value is not None and value > 0 else None)
                        records[scope] = dict(state='evaluated' if changed else 'no_effective_parameter_step',
                            changed_parameters=changed, planned_step_norm=math.sqrt(planned_square),
                            actual_step_norm=math.sqrt(delta_square), transfer=transfer)
                    lookahead[source] = dict(state='active', scopes=records)
            result = dict(training_losses=training_losses, validation_losses=baseline, weights=weights,
                parameter_blocks=blocks, gradient_statistics=statistics, lookahead=lookahead,
                learning_rates={k:float(v) for k,v in learning_rates.items()}, precision=precision,
                cpu_qualification=device.type == 'cpu', parameter_sha256=before,
                parameters_unchanged=True, optimizer_created=False,
                method='Actual functional weighted-SGD step on trainable shared parameters, target loss on separate validation batch; positive 1-L_after/L_before denotes lower validation loss. Not optimizer-state/AdamW replay.',
                interpretation='Fixed eval-mode batch diagnostic; dropout disabled. Frozen/disconnected/zero-gradient cosines and zero-baseline-loss ratios are undefined, not evidence of no conflict. No trained-model improvement, confidence interval or generalization conclusion is established.')
    finally:
        for module, mode in modes:
            module.training = mode
        if parameter_digest(model) != before:
            raise RuntimeError('Objective probe changed the loaded model state')
    return result
