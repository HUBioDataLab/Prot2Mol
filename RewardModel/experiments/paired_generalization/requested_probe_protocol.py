"""Apply the user's fixed seed42 scope to the existing Tier4 probe designs.

Preserve every frozen support/query task, geometry cohort and gradient batch.
Resolve only the two existing full-reference checkpoints; never fit or infer.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256
    from . import requested_evaluation as evaluation, requested_results as results
    from . import adaptation_protocol as adaptation, objective_probe_protocol as objective, adaptation_fit
    from . import representation_geometry as geometry
else:
    from run import ROOT, WORK, sha256
    import requested_evaluation as evaluation, requested_results as results
    import adaptation_protocol as adaptation, objective_probe_protocol as objective, adaptation_fit
    import representation_geometry as geometry

read = evaluation.read
immutable = evaluation.immutable


def destination():
    return WORK / 'tier4/requested_single_seed_protocol'


def checked(path, digest, inputs):
    path = Path(path)
    if not path.absolute().is_relative_to(ROOT) or sha256(path) != digest:
        raise ValueError('Frozen probe input changed or is outside the project: ' + str(path))
    if str(path) in inputs and inputs[str(path)] != digest:
        raise ValueError('Conflicting frozen probe input hashes')
    inputs[str(path)] = digest
    return path


def originals():
    """Verify the old designs without mutating their three-seed registrations."""
    inputs = {}; primary_path = WORK / 'controlled/plan.json'; primary = read(primary_path)
    checked(primary_path, sha256(primary_path), inputs)
    checked(ROOT / 'Paired input ranking generalization.md', primary['spec_sha256'], inputs)
    plans = {}
    for name, folder in (('adaptation', 'adaptation_protocol'), ('objective', 'objective_probe_protocol')):
        directory = WORK / 'tier4' / folder
        complete = read(directory / 'complete.json')
        checked(directory / 'plan.json', complete['plan_sha256'], inputs)
        checked(directory / 'complete.json', sha256(directory / 'complete.json'), inputs)
        plan = read(directory / 'plan.json')
        if plan['primary_plan_sha256'] != sha256(primary_path) or plan['spec_sha256'] != primary['spec_sha256']:
            raise ValueError('Probe protocol belongs to different primary inputs')
        plans[name] = plan
    old = plans['adaptation']; records, groups = {}, {}
    checked(WORK / 'tier4/adaptation_views/complete.json', old['view_completion_sha256'], inputs)
    checked(Path(adaptation.__file__).with_name('ADAPTATION_PROTOCOL.md'), old['protocol_document_sha256'], inputs)
    for name, digest in old['software_sha256'].items(): checked(ROOT / name, digest, inputs)
    for regime in ('A', 'B'):
        artifact = old['view_manifests'][regime]
        record = read(checked(artifact['path'], artifact['sha256'], inputs))
        if (record['regime'] != regime or record['primary_plan']['sha256'] != sha256(primary_path) or
            record['spec_sha256'] != primary['spec_sha256']):
            raise ValueError('Adaptation view identity changed')
        for name in ('groups', 'indices', 'labels', 'design'):
            item = record[name]; checked(item['path'], item['sha256'], inputs)
        groups[regime] = pd.read_parquet(record['groups']['path'])
        expected = adaptation.view_requests(groups[regime], k_values=adaptation.K_VALUES)
        if (any(record[k] != v for k, v in expected.items() if k != 'views') or
            set(record['views']) != set(expected['views'])):
            raise ValueError('Frozen adaptation tasks changed')
        for name, view in expected['views'].items():
            if any(record['views'][name][k] != v for k, v in view.items()):
                raise ValueError('Frozen support/query memberships changed')
        records[regime] = record
    rebuilt = adaptation.build_protocol(primary, records, groups)
    if any(old[k] != v for k, v in rebuilt.items()):
        raise ValueError('Original adaptation protocol is not reproducible from frozen views')
    old = plans['objective']
    directory, frozen = objective.reconstruction.inputs()
    rebuilt = objective.build_protocol(primary, frozen)
    if any(old[k] != v for k, v in rebuilt.items()):
        raise ValueError('Original objective protocol differs from frozen batch design')
    for name, digest in old['input_sha256'].items(): checked(name, digest, inputs)
    for name, digest in old['source_sha256'].items(): checked(ROOT / name, digest, inputs)
    from importlib.metadata import version
    if any(version(n) != v for n, v in old['packages'].items()):
        raise ValueError('Frozen objective-probe packages changed')
    verified = read(directory / 'reconstruction_verification.json')
    if (verified['complete_sha256'] != sha256(directory / 'complete.json') or
        verified['verifier_sha256'] != sha256(Path(objective.reconstruction.__file__)) or
        verified['encoder_source_sha256'] != sha256(ROOT / 'RewardModel/reward_model/model/encoders.py') or
        set(verified['contexts']) != {'A_train', 'A_cold_validation', 'B_train', 'B_cold_validation'} or
        verified['batches_verified'] != 128 or
        any(verified[k] for k in ('model_weights_loaded', 'model_inference', 'new_fits_created', 'scientific_probe_results', 'test_data_read')) or
        not all(c['exact_native_features'] and c['exact_native_collator_tensors'] and c['source_row_table_verified'] for c in verified['contexts'].values())):
        raise ValueError('Native objective-batch reconstruction is not verified')
    directory = WORK / 'tier4/geometry_design'; old = read(directory / 'complete.json')
    checked(directory / 'complete.json', sha256(directory / 'complete.json'), inputs)
    if (old['primary_plan_sha256'] != sha256(primary_path) or old['spec_sha256'] != primary['spec_sha256'] or
        old['sample_cap'] != geometry.SAMPLE_CAP or old['pair_budget'] != geometry.PAIR_BUDGET or
        old['seed'] != 42 or old['stages'] != ['encoder_pooled', 'contrastive_projected'] or
        old['experiment_ids'] != ['E54', 'E55'] or
        set(old['artifacts_sha256']) != {f'{r}_{s}.parquet' for r in ('A', 'B') for s in ('train', 'val', 'test')}):
        raise ValueError('Frozen geometry cohort scope changed')
    checked(Path(geometry.__file__), old['code_sha256'], inputs)
    for name, digest in old['source_sha256'].items(): checked(name, digest, inputs)
    for name, digest in old['artifacts_sha256'].items(): checked(directory / name, digest, inputs)
    plans['geometry'] = old
    return plans, inputs


def amend(plans, bases):
    """Only seed/LR/reference-policy and accounting changes are permitted."""
    if set(bases) != {'A', 'B'}:
        raise ValueError('Use both completed full-reference bases')
    for regime, base in bases.items():
        job = base['job']
        if ((job['arm'], job['regime'], job['seed'], job['learning_rate_multiplier']) != ('full', regime, 42, 1.) or
            job['id'] != f'full__{regime}__seed42__lr1p0' or set(base['rules']) != {'best_macro'} or
            base['rules']['best_macro']['checkpoint'] != base['checkpoint']['id']):
            raise ValueError('Probe base must be the fixed full seed42/base-LR best-macro model')
    result = copy.deepcopy(plans)
    for name in ('adaptation', 'objective'):
        if result[name]['seeds'] != [42, 43, 44]:
            raise ValueError('Expected the immutable original three-seed protocol')
        result[name]['seeds'] = [42]
        result[name]['base_policy'] = 'Reuse full A/B seed42 at fixed base LR, each own best validation-macro checkpoint. No alternative-arm, test-score, LR or additional seed selection.'
        result[name]['resolved_bases'] = {r: copy.deepcopy(dict(job=b['job'], checkpoint=b['checkpoint'], rules=b['rules'])) for r, b in bases.items()}
    a = result['adaptation']
    for cell in a['cells']:
        count = sum(cell['fits_per_seed'].values())
        if cell['total_fits'] != 3 * count or cell['selection_seed'] != 42 or cell['selection_regime'] != 'B':
            raise ValueError('Original adaptation fit or selection accounting changed')
        cell['total_fits'] = count
    total = sum(c['total_fits'] for c in a['cells'])
    if plans['adaptation']['accounting']['support_fits'] != 3 * total:
        raise ValueError('Original support-fit total changed')
    a['accounting']['support_fits'] = total
    a['selection_policy'] = 'For each frozen kind/scope/k, maximize native float32-label macro Spearman over epochs1..15 on B seed42 validation queries, earliest exact tie. Report undefined coverage; apply that epoch to A/B seed42 only. No test query selects epochs.'
    a['fitting_order'] = 'Fit B42 validation tasks first, freeze the common epoch, then fit remaining A/B42 support tasks under the same15-epoch schedule. Each within-unit task resets to its identical base. No confirmation seeds.'
    a['reporting_policy'] = 'Report all native pathway scores, coverage, exposures and actual steps. Compare k on available and common-unit cohorts. Use paired whole-unit bootstrap intervals conditional on seed42; no seed-spread estimate.'
    a['fitting_gates'][0] = 'Requested Tier3 fixed fits and their selected-model evaluations/analyses complete'
    o = result['objective']; slots = []
    for regime, base in bases.items():
        matching = [s for s in o['checkpoint_slots'] if s['regime'] == regime and s['seed'] == 42]
        if len(matching) != 1 or matching[0]['batch_pairs'] != 32:
            raise ValueError('Expected one existing32-pair slot for each A/B42 reference')
        slot = matching[0]
        if base['job']['id'] not in slot['candidate_job_ids']:
            raise ValueError('Fixed reference is outside the original probe registration')
        slot.update(candidate_job_ids=[base['job']['id']], selected_job_id=base['job']['id'], selected_checkpoint=base['checkpoint'])
        slots.append(slot)
    candidates = {b['job']['id']: o['candidates'][b['job']['id']] for b in bases.values()}
    if any(c['learning_rate_multiplier'] != 1. for c in candidates.values()):
        raise ValueError('No alternate step-rate candidate is authorized')
    o.update(checkpoint_slots=slots, candidates=candidates, selection_resolved=True,
        selection_requirement='Verify the fixed requested reference evaluation plans and their native training/retention evidence. Do not call legacy learning-rate selection.',
        reporting_policy='Report each of the32 frozen pairs per A/B42 base, undefined states, per-block norms/cosines and per-source/scope/target changes. Batch distributions are descriptive; shared units preclude independent-batch intervals. No seed-spread estimate.')
    o['accounting'].update(existing_model_slots=2, diagnostic_pairs=64)
    o['scientific_gates'][0] = a['fitting_gates'][0]
    result['geometry'] = dict(design=result['geometry'], resolved_bases=a['resolved_bases'],
        experiments=['E54', 'E55'], seed=42, jobs=[dict(regime=r, partition=s) for r in ('A', 'B') for s in ('train', 'val', 'test')],
        scientific_gate=a['fitting_gates'][0],
        interpretation='Same frozen observation and unique-entity views on each own-regime partition. Before/after refers to projection stages, not extra training checkpoints. No score-dependent resampling or parameter search.')
    return result


def snapshot_budget(cells, bytes_by_scope):
    """Raw tensor upper bounds for the existing snapshot retention rule."""
    rows = []
    for cell in cells:
        bindings = cell['task_bindings']['B']
        validation = len(bindings['across']) if cell['kind'] == 'across' else len(bindings['validation'])
        total = sum(cell['fits_per_seed'].values())
        if not 0 < validation <= total or cell['total_fits'] != total:
            raise ValueError('Single-seed adaptation task accounting differs')
        # All15 epochs permit B validation selection; other tasks keep the
        # selected epoch and final epoch (one payload if both epochs coincide).
        snapshots = validation * 15 + (total - validation) * 2
        rows.append(dict(cell_id=cell['id'], validation_fits=validation,
            remaining_fits=total-validation, maximum_snapshots=snapshots,
            raw_tensor_bytes=snapshots * bytes_by_scope[cell['scope']]))
    return dict(cells=rows, maximum_snapshots=sum(r['maximum_snapshots'] for r in rows),
        raw_tensor_bytes=sum(r['raw_tensor_bytes'] for r in rows),
        bytes_per_snapshot_by_scope=bytes_by_scope,
        exclusions='Serialization metadata, query predictions, logs, base checkpoints and other experiments are additional. Check actual quota before each execution stage.',
        policy='Retain all15 B validation adapter snapshots and at most selected+final snapshots for remaining tasks. No existing checkpoint is removed or changed.')


def checkpoint_snapshot_sizes(bases, inputs):
    """Read mmap tensor metadata on CPU; never construct a model or optimizer."""
    import torch
    by_regime = {}
    for regime, base in bases.items():
        path = Path(base['checkpoint']['path']) / 'pytorch_model.bin'
        checked(path, base['checkpoint']['weights_sha256'], inputs)
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        sizes = {}
        for scope in adaptation.SCOPES:
            roots = objective.probes.PRIVATE_ROOTS | (adaptation_fit.CROSS_ATTENTION_ROOTS if scope == 'heads_cross_attention' else set())
            sizes[scope] = sum(t.numel()*t.element_size() for n, t in state.items() if n.split('.')[0] in roots)
            if sizes[scope] <= 0: raise ValueError('Adaptation scope has no saved tensor elements')
        by_regime[regime] = sizes
        del state
    if by_regime['A'] != by_regime['B']:
        raise ValueError('Full A/B adaptation parameter dimensions differ')
    return by_regime['A']


def build():
    scope_path, scope_plan = evaluation.scope()
    plans, files = originals(); bases = {}
    for job in scope_plan['references']['jobs']:
        path = evaluation.destination(job['id']) / 'plan.json'
        base = read(path)
        if base != evaluation.build_plan(job['id']):
            raise ValueError('Prepared full-reference checkpoint changed')
        checked(path, sha256(path), files); bases[job['regime']] = base
    checked(scope_path, sha256(scope_path), files)
    for module in (evaluation, results, adaptation, adaptation_fit, objective, geometry):
        path = Path(module.__file__); checked(path, sha256(path), files)
    checked(Path(__file__), sha256(Path(__file__)), files)
    probes = amend(plans, bases)
    storage = snapshot_budget(probes['adaptation']['cells'], checkpoint_snapshot_sizes(bases, files))
    return dict(format_version=1, scope_plan_sha256=sha256(scope_path), seeds=[42],
        learning_rate_multiplier=1., gpu_uuids=list(evaluation.GPU_UUIDS), probes=probes,
        inputs_sha256=files, boot_id=evaluation.control_queue.boot_id(), adaptation_snapshot_budget=storage,
        accounting=dict(adaptation_support_fits=probes['adaptation']['accounting']['support_fits'],
            adaptation_selection_cells=len(probes['adaptation']['cells']), geometry_exports=6,
            objective_model_slots=2, objective_batch_pairs=64, additional_seed_fits=0,
            additional_learning_rates=0),
        model_fitting=False, model_inference=False, scientific_probe_results=False,
        production_runtime_qualified=False, executable_gpu_queue=False,
        interpretation='CPU protocol amendment only. Preserve all existing task memberships/budgets/metric definitions and unavailable cohorts. Base checkpoints are resolved; requested Tier3 completion, token checks, GPU execution and scientific reporting remain required.')


def prepare():
    evaluation.control_queue.runtime_guard()
    plan = build(); path = destination() / 'plan.json'; immutable(path, plan)
    immutable(destination() / 'complete.json', dict(plan_sha256=sha256(path), accounting=plan['accounting'],
        source_and_membership_hashes_verified=True, model_fitting=False, model_inference=False,
        scientific_probe_results=False, production_runtime_qualified=False))
    return path


def verify():
    path = destination() / 'plan.json'; plan = read(path)
    if (read(destination() / 'complete.json')['plan_sha256'] != sha256(path) or plan != build()):
        raise ValueError('Requested probe protocol changed')
    return plan


def require_tier3(analysis_paths):
    """Selected comparisons are a prerequisite, not a new model-selection step."""
    plan = verify()
    scope_path, scope_plan = evaluation.scope()
    done = read(scope_path.parent / 'complete.json')
    if (done['phase'] != 'requested_fixed_config_training_complete_evaluations_and_probes_pending' or
        done['plan_sha256'] != plan['scope_plan_sha256'] or done['training_fits_created'] != 24 or
        done['parameter_search'] or done['additional_seed_fits'] != 0 or
        done['evidence'] != evaluation.control_queue.require_completed(scope_plan['jobs'])):
        raise ValueError('Requested training comparisons are not complete')
    expected = {j['id'] for j in [*scope_plan['references']['jobs'], *scope_plan['jobs']]}
    paths = [Path(p) for p in analysis_paths]
    if len(paths) != 26 or len(set(paths)) != 26:
        raise ValueError('Need every requested selected-model analysis')
    records = [results.verify_analysis(path) for path in paths]
    if {r['job']['id'] for r in records} != expected:
        raise ValueError('Tier3 analyses omit or add an unrequested model')
    return dict(training_complete_sha256=sha256(scope_path.parent / 'complete.json'),
        analysis_sha256={str(p): sha256(p) for p in paths}, requested_fits=26,
        scientific_probes_complete=False, gpu_execution_still_required=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify'))
    args = parser.parse_args()
    print(prepare() if args.action == 'prepare' else json.dumps(verify()['accounting'], indent=2))
