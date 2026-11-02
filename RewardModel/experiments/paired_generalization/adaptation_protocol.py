"""Freeze E49-E52 fitting/selection rules without fitting or reading predictions."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .control_plan import GPU_UUIDS, digest_json, _write_immutable
    from .adaptation_design import K_VALUES
    from .adaptation_views import checked_artifact, view_requests
else:
    from run import ROOT, WORK, sha256, write_json
    from control_plan import GPU_UUIDS, digest_json, _write_immutable
    from adaptation_design import K_VALUES
    from adaptation_views import checked_artifact, view_requests

SEEDS = (42, 43, 44)
SCOPES = ('heads', 'heads_cross_attention')
HORIZON = 15


def validation_groups(record, groups, kind, k):
    """Label-eligible B validation assays, including undefined future predictions."""
    if kind == 'across':
        task = record['tasks'].get(f'across_k{k}')
        view_ids = [task['validation_view']] if task else []
    elif kind == 'within':
        view_ids = [task['query_view'] for task in record['tasks'].values()
                    if task['kind'] == 'within' and task['k'] == k
                    and task['probe_role'] == 'validation']
    else:
        raise ValueError('Unregistered adaptation kind')
    ids = [group for view in view_ids for group in record['views'][view]['group_ids']]
    if len(ids) != len(set(ids)):
        raise ValueError('Validation query assays overlap between tasks')
    selected = groups.set_index('group_id').loc[ids]
    if not selected.probe_role.eq('validation').all():
        raise ValueError('Epoch selection includes a non-validation unit')
    return dict(sorted(selected.loc[selected.ranking_eligible, 'unit'].items()))


def build_protocol(primary, records, groups):
    if ([primary['tuning_seed'], *primary['confirmation_seeds']] != list(SEEDS)
            or primary['gpu_uuids'] != list(GPU_UUIDS)):
        raise ValueError('Expected the registered three seeds and four authorized GPUs')
    for regime in ('A', 'B'):
        if records[regime]['k_values'] != list(K_VALUES):
            raise ValueError('Adaptation group budgets changed')
        training = primary['bases'][regime]['training']
        if (training['per_device_train_batch_size'] != 12
                or training['gradient_accumulation_steps'] != 4
                or training['num_train_epochs'] != HORIZON):
            raise ValueError('Expected the registered 12x4x4, 15-epoch base settings')
    cells, unavailable = [], []
    for kind in ('across', 'within'):
        for k in K_VALUES:
            expected = validation_groups(records['B'], groups['B'], kind, k)
            if not expected:
                unavailable.append(dict(kind=kind, k=k,
                    reason='No ranking-eligible B validation query assay; no epoch can be selected'))
                continue
            bindings = {}
            for regime in ('A', 'B'):
                tasks = records[regime]['tasks']
                if kind == 'across':
                    ids = [f'across_k{k}'] if f'across_k{k}' in tasks else []
                    bindings[regime] = dict(across=ids)
                else:
                    bindings[regime] = {role: sorted(identity for identity, task in tasks.items()
                        if task['kind'] == 'within' and task['k'] == k and task['probe_role'] == role)
                        for role in ('validation', 'test')}
                if not any(bindings[regime].values()):
                    unavailable.append(dict(regime=regime, kind=kind, k=k,
                        reason='No disjoint support/query task at this whole-unit budget'))
            fits_per_seed = {regime: sum(len(v) for v in roles.values())
                             for regime, roles in bindings.items()}
            for scope in SCOPES:
                cells.append(dict(id=f'{kind}__{scope}__k{k}', kind=kind, scope=scope, k=k,
                    selection_regime='B', selection_seed=42, selection_role='validation',
                    validation_groups=expected, validation_groups_sha256=digest_json(expected),
                    task_bindings=bindings, fits_per_seed=fits_per_seed,
                    total_fits=len(SEEDS)*sum(fits_per_seed.values())))
    return dict(format_version=1, seeds=list(SEEDS), scopes=list(SCOPES), k_values=list(K_VALUES),
        epochs=HORIZON, cells=cells, unavailable=unavailable,
        base_policy='For each A/B seed, use the full R+B+C primary-grid model at the common B-validation-selected learning rate and that run\'s own best validation-macro checkpoint. No winning alternative arm or test score chooses the base.',
        optimizer_policy='Inherit the selected full-run optimizer, learning rates, weight decay, warmup, precision and 15-epoch schedule. No additional adaptation LR or weight search.',
        batching=dict(devices_per_fit=1, per_device_train_batch_size=12,
            gradient_accumulation_steps=16, nominal_effective_batch=192,
            interpretation='Preserves the original per-device contrastive pool and nominal 12x4x4 optimizer batch. Small supports can yield shorter final accumulation windows; report actual exposures and optimizer steps.'),
        selection_policy='For each kind/scope/k, maximize native float32-label macro Spearman over epochs1..15 on B seed42 validation queries; earliest exact tie. Compute the mean over defined eligible assays and report coverage, as in the primary metric. Apply that epoch to A and B, seeds42/43/44. No test query or confirmation seed selects an epoch.',
        fitting_order='Fit B seed42 validation tasks first and freeze the epoch. Then fit remaining A/B support tasks and confirmation seeds, retaining the selected epoch and final epoch under the original 15-epoch schedule. Every within-unit support is a separate model reset to the exact same selected base for its regime/seed.',
        baseline_policy='Unchanged k=0 base on each exact query cohort, shared between scopes. k=0 is a reported comparison, not a checkpoint candidate. No zero-support optimization.',
        reporting_policy='Report native per-assay macro rho, eligibility/defined coverage, all pathway scores, support/objective exposures, and available plus identical common-unit cohorts across k. Use paired whole-unit bootstrap intervals within each seed and report seed spread separately; do not treat repeated seeds as independent units.',
        logging_policy='Per-unit fits write compact local receipts/parameters. Aggregate probe progress and results in W&B at kind/scope/k/regime/seed level, without starting a separate W&B run for every small unit.',
        accounting=dict(support_fits=sum(cell['total_fits'] for cell in cells),
                        selection_cells=len(cells), additional_learning_rates=0,
                        k0_baselines_require_fitting=False),
        fitting_gates=['Tier 3 experiments and their selected-model evaluations complete',
            'Resolve and verify every base checkpoint, selection/configuration receipt and data-view hash',
            'Qualify production GPU fitting/restoration and query scoring under this protocol',
            'Acquire the shared GPU lock and expose only assigned authorized UUIDs'],
        limitation='Base-model selection already used original validation units. These are conditional mechanistic probes on the existing benchmark. Large-k within-unit cohorts may be tiny or infeasible; no missing cohort is replaced or pooled with another unit.',
        model_fitting=False, model_inference=False, production_runtime_qualified=False)


def adaptation_config(base, output):
    """Preserve optimizer/objective settings; serialize independent model fits."""
    t = base.training
    if (t.num_train_epochs != HORIZON or t.max_steps is not None
            or t.per_device_train_batch_size != 12 or t.gradient_accumulation_steps != 4
            or t.seed not in SEEDS):
        raise ValueError('Expected a selected registered full-run training configuration')
    return replace(base, training=replace(t, output_dir=str(output), training_mode='single_gpu',
        gradient_accumulation_steps=16, report_to=[], evaluate_test=False,
        fixed_train_eval_assays=0, export_evaluation_predictions=False,
        objective_gradient_diagnostics_steps=0))


def select_epoch(records, cell):
    """Validate complete B/42 validation coverage before choosing a common epoch.

Each row is one native float32-label eligible assay correlation. Scientific
callers must reconstruct these rows from verified native prediction exports.
Undefined correlations remain explicit rows, never silently missing assays.
"""
    expected = cell['validation_groups']
    if not expected or digest_json(expected) != cell['validation_groups_sha256']:
        raise ValueError('Selection requires the frozen nonempty eligible query inventory')
    rows, by_epoch = [], {epoch: {} for epoch in range(1, HORIZON+1)}
    for record in records:
        if any(record.get(key) != value for key, value in dict(
            regime='B', seed=42, role='validation', kind=cell['kind'], scope=cell['scope'], k=cell['k']).items()):
            raise ValueError('Only the registered B seed42 validation cell may select epochs')
        epoch, group = record['epoch'], record['group_id']
        if (type(epoch) is not int or epoch not in by_epoch or group not in expected
                or record['unit'] != expected[group] or group in by_epoch[epoch]):
            raise ValueError('Wrong epoch, query unit, assay identity or duplicate selection row')
        rho = record['rho']
        if rho is not None:
            rho = float(rho)
            if math.isnan(rho):
                rho = None
            elif not math.isfinite(rho) or not -1.000000000001 <= rho <= 1.000000000001:
                raise ValueError('Invalid assay correlation')
        by_epoch[epoch][group] = rho
        rows.append(dict(epoch=epoch, group_id=group, unit=expected[group], rho=rho))
    if any(set(values) != set(expected) for values in by_epoch.values()):
        raise ValueError('Every epoch must include every frozen eligible validation assay')
    candidates = []
    for epoch, values in by_epoch.items():
        defined = [v for _, v in sorted(values.items()) if v is not None]
        candidates.append(dict(epoch=epoch, macro_spearman=float(np.mean(defined)) if defined else None,
            eligible_assays=len(expected), defined_assays=len(defined),
            defined_fraction=len(defined)/len(expected)))
    valid = [r for r in candidates if r['macro_spearman'] is not None]
    winner = max(valid, key=lambda r: (r['macro_spearman'], -r['epoch'])) if valid else None
    return dict(cell_id=cell['id'], validation_groups_sha256=cell['validation_groups_sha256'],
        validation_records_sha256=digest_json(sorted(rows, key=lambda r:(r['epoch'], r['group_id']))),
        status='selected' if winner else 'not_selectable', selected=winner,
        candidates=candidates, criterion='Maximum B seed42 validation native macro Spearman; earliest tie; undefined-prediction coverage reported',
        test_queries_used=False, confirmation_seeds_used=False)


def prepare():
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use CPU-only preparation inside Atum Prot2Mol')
    primary_path = WORK/'controlled/plan.json'
    complete_path = WORK/'tier4/adaptation_views/complete.json'
    primary = json.loads(primary_path.read_text())
    complete = json.loads(complete_path.read_text())
    if primary['spec_sha256'] != sha256(ROOT/'Paired input ranking generalization.md'):
        raise ValueError('Governing MD differs from the registered primary plan')
    records, groups, evidence = {}, {}, {}
    for regime in ('A', 'B'):
        path = checked_artifact(complete['manifests'][regime])
        record = json.loads(path.read_text())
        if (record['regime'] != regime or record['primary_plan']['sha256'] != sha256(primary_path)
                or record['spec_sha256'] != primary['spec_sha256']):
            raise ValueError('Adaptation views belong to different primary inputs')
        group_path = checked_artifact(record['groups'])
        frame = pd.read_parquet(group_path)
        requests = view_requests(frame, k_values=K_VALUES)
        if (any(record[key] != value for key, value in requests.items() if key != 'views')
                or set(record['views']) != set(requests['views'])):
            raise ValueError('Adaptation memberships no longer match their frozen design')
        for name, request in requests['views'].items():
            if any(record['views'][name][key] != value for key, value in request.items()):
                raise ValueError('Packed view changed query/support group membership')
        for name in ('indices', 'labels', 'design'):
            checked_artifact(record[name])
        records[regime], groups[regime] = record, frame
        evidence[regime] = complete['manifests'][regime]
    plan = build_protocol(primary, records, groups)
    plan.update(primary_plan_sha256=sha256(primary_path), spec_sha256=primary['spec_sha256'],
        view_manifests=evidence, view_completion_sha256=sha256(complete_path),
        protocol_document_sha256=sha256(Path(__file__).with_name('ADAPTATION_PROTOCOL.md')),
        software_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (
            Path(__file__), Path(__file__).with_name('adaptation_fit.py'),
            Path(__file__).with_name('adaptation_views.py'))})
    directory = WORK/'tier4/adaptation_protocol'
    path = directory/'plan.json'
    _write_immutable(path, json.dumps(plan, indent=2, sort_keys=True)+'\n')
    result = dict(plan_sha256=sha256(path), accounting=plan['accounting'],
                  source_checks='Frozen view manifests, memberships and packed vectors; scientific callers must still verify actual token payloads and selected base checkpoints',
                  model_fitting=False, model_inference=False, production_runtime_qualified=False)
    _write_immutable(directory/'complete.json', json.dumps(result, indent=2, sort_keys=True)+'\n')
    return result


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(prepare(), indent=2))
