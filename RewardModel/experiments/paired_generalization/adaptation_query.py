"""Query-only scoring for E49-E52, using the frozen adaptation views.

Library only: the caller verifies Tier 3, the selected base checkpoint, source
hashes and GPU lock. This does not fit, select epochs or enqueue any work.
Scientific GPU qualification remains separate from the CPU fixture path.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from . import adaptation_fit
from .adaptation_protocol import HORIZON, SEEDS, SCOPES
from .adaptation_views import AdaptationViews, checked_artifact
from .checkpoint_evaluation import evaluate_fixed_model, parameter_digest, read_native_predictions
from .control_plan import GPU_UUIDS, _write_immutable
from .realization_protocol import IDENTITY, LABELS, example_frame
from .run import ROOT, WORK, sha256
from reward_model.training.evaluation import _compute_groupwise_spearman_with_records

SCORERS = ('ranking_score', 'activity_logits', 'cosine_similarity')


def task_source(views, task_id, seed):
    """The provenance required when calling the support-only fitting worker."""
    if not isinstance(views, AdaptationViews) or seed not in SEEDS:
        raise ValueError('Expected verified adaptation views and a registered seed')
    task = views.record['tasks'][task_id]
    support = views.load(task['support_view']) if task['support_view'] else None
    return dict(view_manifest_sha256=views.digest, regime=views.record['regime'],
        seed=seed, task_id=task_id, task=task,
        support_fingerprint=support._fingerprint if support is not None else None)


def query_view(views, task_id, role):
    if role not in ('validation', 'test'):
        raise ValueError('Expected a validation or test query role')
    task = views.record['tasks'][task_id]
    if task['kind'] == 'across':
        name = task['validation_view'] if role == 'validation' else task['query_view']
    elif task['kind'] == 'within' and role == task['probe_role']:
        name = task['query_view']
    else:
        raise ValueError('Within-unit query role differs from the frozen task')
    view = views.record['views'][name]
    groups = views.groups.set_index('group_id').loc[view['group_ids']].reset_index()
    if not groups.probe_role.eq(role).all():
        raise ValueError('Query contains a unit assigned to a different role')
    if task['support_view']:
        support_ids = views.record['views'][task['support_view']]['group_ids']
        support = views.groups.set_index('group_id').loc[support_ids]
        if set(support_ids) & set(groups.group_id):
            raise ValueError('Support and query assays overlap')
        if task['kind'] == 'across' and set(support.unit) & set(groups.unit):
            raise ValueError('Across-unit support and query units overlap')
        if task['kind'] == 'within' and (set(support.unit) | set(groups.unit)) != {task['unit']}:
            raise ValueError('Within-unit task includes another unit')
    return name, views.load(name), groups


def checked_snapshot(directory, complete_sha256, source, model, config, scope, epoch, cpu):
    directory = Path(directory)
    complete = json.loads(checked_artifact(dict(path=str(directory/'complete.json'),
                                               sha256=complete_sha256)).read_text())
    plan = json.loads(checked_artifact(dict(path=str(directory/'fit_plan.json'),
        sha256=complete['fit_plan_sha256'])).read_text())
    base = parameter_digest(model)
    if (complete['source'] != source or plan['source'] != source
            or complete['scope'] != scope or plan['scope'] != scope
            or complete['cpu_qualification'] != cpu or plan['cpu_qualification'] != cpu
            or complete['base_parameter_sha256'] != base or plan['base_parameter_sha256'] != base
            or not complete['frozen_parameters_unchanged']
            or complete['query_examples_received'] or complete['selection_performed']
            or plan['query_examples_received'] or plan['selection_performed']
            or plan['support_fingerprint'] != source['support_fingerprint']
            or plan['model_config'] != json.loads(json.dumps(model.config.to_dict()))
            or plan['training']['seed'] != config.training.seed
            or plan['worker_source_sha256'] != sha256(Path(adaptation_fit.__file__))):
        raise ValueError('Support fit differs from the registered task, base or execution mode')
    if (not cpu and complete['epochs'] != HORIZON) or complete['epochs'] != plan['training']['num_train_epochs']:
        raise ValueError('Support fit did not finish the prescribed horizon')
    snapshots = complete['snapshots']
    if ([x['epoch'] for x in snapshots] != plan['save_epochs']
            or complete['epochs'] not in plan['save_epochs']):
        raise ValueError('Support snapshot inventory is incomplete')
    matches = [x for x in snapshots if x['epoch'] == epoch]
    if len(matches) != 1:
        raise ValueError('Requested adaptation epoch has no unique saved payload')
    snapshot = matches[0]
    path = directory/snapshot['file']
    if path.resolve().parent != directory.resolve():
        raise ValueError('Adapter payload escapes its fit directory')
    checked_artifact(dict(path=str(path), sha256=snapshot['sha256']))
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload['epoch'] != epoch or payload['step'] != snapshot['step']
            or payload['scope'] != scope or payload['base_parameter_sha256'] != base):
        raise ValueError('Adapter payload disagrees with its fit receipt')
    return path, snapshot['sha256']


def pathway_records(frame, groups, min_span):
    """Retain every assay/unit, including ineligible and undefined correlations."""
    if set(frame.group_id) != set(groups.group_id) or groups.group_id.duplicated().any():
        raise ValueError('Query assay inventory changed')
    counts = frame.groupby('group_id').size()
    if any(counts[row.group_id] != row.rows for row in groups.itertuples()):
        raise ValueError('Query no longer contains complete frozen assays')
    if not np.isfinite(frame[list(SCORERS)+LABELS].to_numpy()).all():
        raise ValueError('Query scores or labels are nonfinite')
    summaries, rows = {}, []
    expected = set(groups.loc[groups.ranking_eligible, 'group_id'])
    for scorer in SCORERS:
        metrics, records = _compute_groupwise_spearman_with_records(
            group_ids=frame.group_id.tolist(), ranking_scores=frame[scorer].tolist(),
            pchembl_values=frame.pchembl_value.tolist(), min_group_size=3, min_pchembl_span=min_span)
        indexed = {r['group_id']:r for r in records}
        if set(indexed) != expected:
            raise ValueError('Native float32 eligibility differs from the frozen query inventory')
        summaries[scorer] = {k.removeprefix('eval_'):(float(v) if math.isfinite(v) else None)
                             for k,v in metrics.items()}
        for group in groups.sort_values('group_id').itertuples():
            record = indexed.get(group.group_id)
            rows.append(dict(scorer=scorer, group_id=group.group_id, unit=group.unit,
                rows=int(group.rows), eligible=record is not None,
                defined=bool(record and record['spearman_defined']),
                rho=float(record['spearman']) if record and record['spearman_defined'] else None))
    return summaries, rows


def evaluate_query(model, views, config, output, *, task_id, role, epoch,
                   base_parameter_sha256, scope=None, fit_directory=None,
                   fit_complete_sha256=None, known_positive_index=None, cpu=False):
    """Score one saved adapter or k=0 base; restore the base before returning.

An adaptation selection coordinator must separately require every B/42
validation query at all 15 epochs before calling adaptation_protocol.select_epoch.
"""
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Query scoring requires one independent process')
    if cpu:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(p.device.type != 'cpu' for p in model.parameters()):
            raise ValueError('CPU qualification requires hidden GPUs and a CPU model')
    elif (ROOT != Path('/home/atabey/Prot2Mol')
          or os.environ.get('CUDA_VISIBLE_DEVICES') not in GPU_UUIDS
          or not Path(output).resolve().is_relative_to(WORK.resolve())):
        raise ValueError('Scientific query scoring requires an authorized Atum GPU and project output')
    if config.model != model.config or type(epoch) is not int or epoch < 0:
        raise ValueError('Expected the unchanged model configuration and a whole epoch')
    if parameter_digest(model) != base_parameter_sha256:
        raise ValueError('Each query call must start from the exact unchanged base')
    source = task_source(views, task_id, config.training.seed)
    task = source['task']
    name, examples, groups = query_view(views, task_id, role)
    source = dict(source, query_view=name, query_role=role, query_fingerprint=examples._fingerprint,
                  epoch=epoch, scope=scope)
    baseline = task['k'] == 0
    adapter = None
    if baseline:
        if epoch != 0 or any(x is not None for x in (scope, fit_directory, fit_complete_sha256)):
            raise ValueError('k=0 uses the unchanged base without an adapter or selection epoch')
    else:
        if epoch < 1 or scope not in SCOPES or fit_directory is None or fit_complete_sha256 is None:
            raise ValueError('Adapted queries require a saved scoped fit and positive epoch')
        adapter = checked_snapshot(fit_directory, fit_complete_sha256,
            task_source(views, task_id, config.training.seed), model, config, scope, epoch, cpu)
    output = Path(output)
    if output.exists():
        raise RuntimeError('Existing query output requires inspection; no implicit retry')
    parameters = dict(model.named_parameters())
    names = adaptation_fit.scope_parameters(model, scope) if adapter else []
    original = {name:parameters[name].detach().cpu().clone() for name in names}
    modes = {module:module.training for module in model.modules()}
    flags = {name:p.requires_grad for name,p in parameters.items()}
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    device = next(model.parameters()).device
    cuda_devices = [] if cpu else [device.index or 0]
    prefix = 'eval' if role == 'validation' else 'test'
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            if adapter:
                adaptation_fit.restore_adapter(model, *adapter)
            native = evaluate_fixed_model(model, examples, config, output/'native', prefix=prefix,
                known_positive_index=known_positive_index, source=source, cpu=cpu)
            frame, manifest = read_native_predictions(native['prediction_manifest'])
            if manifest['dataset_fingerprint'] != examples._fingerprint or manifest['metric_prefix'] != prefix:
                raise ValueError('Native export changed query provenance')
            expected = example_frame(examples.select_columns(IDENTITY+LABELS).to_pandas())
            actual = example_frame(frame)
            pd.testing.assert_frame_equal(actual[IDENTITY+LABELS], expected[IDENTITY+LABELS],
                                          check_dtype=False, check_exact=True)
            summaries, records = pathway_records(actual, groups, config.data.ranking_min_pchembl_span)
            reported = native['metrics'][prefix+'_macro_spearman']
            reconstructed = summaries['ranking_score']['macro_spearman']
            if not np.isclose(float('nan') if reconstructed is None else reconstructed,
                              float('nan') if reported is None else reported, atol=1e-14, rtol=0, equal_nan=True):
                raise ValueError('Query macro differs from native evaluation')
    finally:
        with torch.no_grad():
            for name, value in original.items():
                parameters[name].copy_(value)
        for module, mode in modes.items():
            module.training = mode
        for name, flag in flags.items():
            parameters[name].requires_grad_(flag)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        if parameter_digest(model) != base_parameter_sha256:
            raise RuntimeError('Query scoring failed to preserve the base model')
    selection_records = [dict(regime=source['regime'], seed=source['seed'], role=role,
        kind=task['kind'], scope=scope, k=task['k'], epoch=epoch,
        group_id=r['group_id'], unit=r['unit'], rho=r['rho']) for r in records
        if r['scorer'] == 'ranking_score' and r['eligible']] if not baseline else []
    result = dict(format_version=1, source=source, worker_source_sha256=sha256(Path(__file__)),
        base_parameter_sha256=base_parameter_sha256,
        fit_complete_sha256=fit_complete_sha256, adapter_sha256=adapter[1] if adapter else None,
        native_complete_sha256=sha256(output/'native/complete.json'),
        prediction_manifest=native['prediction_manifest'],
        prediction_manifest_sha256=native['prediction_manifest_sha256'],
        rows=len(examples), assays=len(groups), units=int(groups.unit.nunique()),
        pathways=summaries, assay_records=records, selection_records=selection_records,
        base_restored=True, modes_and_rng_restored=True, optimizer_created=False,
        selection_performed=False, cpu_qualification=cpu,
        interpretation='Fixed query cohort and saved parameters only. Undefined assays remain explicit. '
        'No epoch selection, adaptation performance comparison, interval or Tier-4 completion follows from one query pass.')
    _write_immutable(output/'complete.json', json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return result
