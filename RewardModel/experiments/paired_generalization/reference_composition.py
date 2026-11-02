"""E08/E09: descriptive A/B validation comparison for the two base-LR references.

Use completed, independently verified native trajectories and the existing
score-independent composition cells. No fitting, test predictions or new
model-selection procedure. Same-epoch results and native validation peaks are
reported separately; neither is an unbiased selected-test comparison.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256
    from . import native_analysis, verify_native_analysis, control_driver, control_plan, similarity_composition as composition
else:
    from run import ROOT, WORK, sha256
    import native_analysis, verify_native_analysis, control_driver, control_plan, similarity_composition as composition

read = composition.read
immutable = composition.immutable


def software():
    result = native_analysis.software()
    for path in (Path(__file__), Path(verify_native_analysis.__file__), Path(control_driver.__file__)):
        result['source_sha256'][str(path.relative_to(ROOT))] = sha256(path)
    return result


def bind(path, evidence, expected=None):
    path = Path(path); digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(f'Composition source changed: {path}')
    if str(path) in evidence and evidence[str(path)] != digest:
        raise ValueError('Composition source changed during reading')
    evidence[str(path)] = digest
    return digest


def load_reference(regime, evidence):
    if regime not in ('A', 'B'):
        raise ValueError('Only the existing A/B references are accepted')
    identity = control_plan.job_id('full', regime, 42, 1.)
    job, primary, registration = native_analysis.registration(identity)
    if not control_driver.completed_job(job):
        raise RuntimeError('Reference training/export/retention is incomplete')
    directory = WORK/'tier1/native_trajectories'/identity/control_plan.digest_json(native_analysis.software())[:16]
    trajectory_path = directory/'trajectory_through_epoch15.json'
    verification_path = directory/'verification_through_epoch15.json'
    bind(verification_path, evidence); verified = read(verification_path)
    bind(trajectory_path, evidence, verified['trajectory_sha256']); trajectory = read(trajectory_path)
    bind(directory/'analysis_plan.json', evidence, trajectory['analysis_plan_sha256'])
    plan = read(directory/'analysis_plan.json')
    if (trajectory['job'] != job or not trajectory['complete_horizon_observed'] or trajectory['horizon'] != 15 or
        trajectory['epochs_analyzed'] != list(range(1, 16)) or trajectory['test_predictions_read'] or
        verified['test_predictions_read'] or not verified['source_and_artifact_hashes_verified'] or
        verified['analysis_plan_sha256'] != trajectory['analysis_plan_sha256'] or
        verified['verifier_sha256'] != sha256(Path(verify_native_analysis.__file__)) or
        plan['software'] != native_analysis.software() or
        any(plan[k] != v for k, v in registration.items())):
        raise ValueError('Reference needs the exact completed independent validation/probe audit')
    checks = {(x['epoch'], x['prefix']): x for x in verified['checks']}
    if len(verified['checks']) != 30 or set(checks) != {(e, p) for e in range(1, 16) for p in ('eval', 'train_eval')}:
        raise ValueError('Independent verification omits or duplicates snapshots')
    bind(WORK/'controlled/plan.json', evidence, job['plan_sha256'])
    bind(job['config_path'], evidence, job['config_sha256'])
    bind(WORK/'controlled/jobs'/f'{identity}.json', evidence, registration['job_metadata_sha256'])
    run = Path(job['output_dir'])
    bind(run/'launch_provenance.json', evidence, registration['launch_sha256'])
    bind(run/'controlled_training_complete.json', evidence)
    bind(run/'runtime_qualification.json', evidence); qualification = read(run/'runtime_qualification.json')
    if (qualification['world_size'] != 4 or qualification['signature'] != control_plan.digest_json(
        dict(sources=qualification['sources'], plan_sha256=job['plan_sha256']))):
        raise ValueError('Training qualification differs from the primary plan')
    for name, digest in {**plan['software']['source_sha256'], **qualification['sources']}.items():
        if not (ROOT/name).resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Source code is outside the authorized project')
        bind(ROOT/name, evidence, digest)
    bind(directory/'references/complete.json', evidence, plan['reference_sha256'])
    bind(WORK/'tier1/data/assay_units.parquet', evidence, plan['units_sha256'])
    bind(WORK/'tier1/similarity_composition/design/complete.json', evidence, plan['similarity_design_sha256'])
    if [row['epoch'] for row in trajectory['trajectory']] != list(range(1, 16)):
        raise ValueError('Reference trajectory is incomplete or out of order')
    for row in trajectory['trajectory']:
        for prefix in ('eval', 'train_eval'):
            snapshot = row[prefix]; source = snapshot['source']; manifest_path = Path(source['manifest'])
            if (snapshot['epoch'] != row['epoch'] or snapshot['step'] != row['step'] or snapshot['prefix'] != prefix or
                snapshot['ranking_readout'] != 'ranking_head' or source['provenance'] != plan or
                manifest_path != run/'prediction_exports/trajectory'/f"{prefix}_step{row['step']:08d}.json"):
                raise ValueError('Reference snapshot role, checkpoint or provenance differs')
            bind(manifest_path, evidence, source['manifest_sha256']); manifest = read(manifest_path)
            if (Path(manifest['predictions_file']).name != manifest['predictions_file'] or
                manifest['predictions_sha256'] != source['payload_sha256'] or
                manifest['metrics'] != snapshot['native_metrics']):
                raise ValueError('Reference prediction metadata differs')
            bind(manifest_path.parent/manifest['predictions_file'], evidence, source['payload_sha256'])
            for name, digest in snapshot['artifacts_sha256'].items():
                if Path(name).name != name:
                    raise ValueError('Analysis artifact must belong to its snapshot')
                bind(directory/manifest_path.stem/name, evidence, digest)
            expected = checks[row['epoch'], prefix]['macros']['ranking_score']
            native_analysis.check_native_metric(snapshot['metrics']['ranking_score']['original_eligible_macro_spearman'], expected, 'verified macro')
    return dict(directory=directory, trajectory=trajectory, qualification=qualification)


def join_scores(design, scores):
    """Retain every original unit, including units with no eligible score."""
    if (design.group_id.duplicated().any() or scores.group_id.duplicated().any() or
        set(design.group_id) != set(scores.group_id)):
        raise ValueError('Composition group identity coverage differs')
    scored = scores.set_index('group_id').loc[design.group_id]
    for a, b in (('size', 'size'), ('eligible', 'original_ranking_eligible'), ('unit_id', 'component')):
        if not np.array_equal(design[a].to_numpy(), scored[b].to_numpy()):
            raise ValueError('Composition size, eligibility or indivisible unit differs')
    return design.assign(spearman=scored.spearman.to_numpy())


def comparisons(tables, cells, rows):
    results = {str(e): composition.compare_composition(tables['A'][e], tables['B'][e], cells) for e in range(1, 16)}
    peaks = {r: int(max(rows[r], key=lambda x: (x['eval']['native_metrics']['eval_macro_spearman'], -x['epoch']))['epoch']) for r in ('A', 'B')}
    return dict(same_epoch=results, own_native_macro_peaks=dict(epochs=peaks,
        composition=composition.compare_composition(tables['A'][peaks['A']], tables['B'][peaks['B']], cells),
        interpretation='In-sample validation peak comparison at the existing base LR; not the primary LR selection or an unbiased test estimate.'))


def run():
    evidence = {}; code = software()
    references = {r: load_reference(r, evidence) for r in ('A', 'B')}
    if references['A']['qualification'] != references['B']['qualification']:
        raise ValueError('Reference training qualifications differ')
    design = WORK/'tier1/similarity_composition/design'; design_record = composition.verify_design(design)
    bind(design/'complete.json', evidence)
    for name, digest in design_record['artifacts_sha256'].items():
        bind(design/name, evidence, digest)
    cells = pd.read_parquet(design/'weights_val.parquet'); tables = {}; rows = {}
    for regime, reference in references.items():
        groups = pd.read_parquet(design/f'groups_{regime}_val.parquet')
        rows[regime] = reference['trajectory']['trajectory']; tables[regime] = {}
        for row in rows[regime]:
            table = reference['directory']/f"eval_step{row['step']:08d}"/'groups_ranking_score.parquet'
            tables[regime][int(row['epoch'])] = join_scores(groups, pd.read_parquet(table))
    source = dict(input_sha256=evidence, software=code)
    target = WORK/'tier1/reference_composition'/control_plan.digest_json(source)[:16]/'complete.json'
    if target.exists():
        saved = read(target)
        if saved['source'] != source or saved['payload_sha256'] != control_plan.digest_json(saved['result']):
            raise ValueError('Completed composition report changed')
        return target
    result = dict(**comparisons(tables, cells, rows),
        similarity={r: {str(int(x['epoch'])): x['eval']['similarity'] for x in rows[r]} for r in rows},
        similarity_verification='Reuses hash-bound native analysis summaries. Whole-assay stratum means are independently checked below. Within-assay restricted-row metrics and bootstrap intervals are not independently recomputed here.',
        validation_only=True, test_predictions_read=False,
        new_fits_created=0, study_complete=False, source_jobs={r: references[r]['trajectory']['job']['id'] for r in references},
        interpretation='All15 matched epochs of two completed full-model seed42 base-LR references. Fixed cells use assay size, positive fraction and mean pChEMBL. Report common-support mass, defined coverage and residual within-bin imbalance. Whole-unit intervals condition on fitted models and fixed empirical weights; they omit seed, split, weight-estimation and checkpoint-selection uncertainty. A-minus-B gaps compare distinct populations and are descriptive, not causal. A and B partition roles can overlap, so no independent cross-regime uncertainty is asserted. Binary scaffold support is not graded chemical similarity. No learning rate, model family, test checkpoint or new experiment is selected.')
    # Independently recompute point estimates from the joined group rows; this
    # checks the new weighting/join, not just the upstream correlation values.
    lookup = cells.set_index('cell')
    for epoch, compared in result['same_epoch'].items():
        for regime in ('A', 'B'):
            table = tables[regime][int(epoch)]; eligible = table.eligible.to_numpy()
            if regime == 'A':
                w = table.cell.map(lookup.A_group_weight).fillna(0).to_numpy()*eligible
                modes = dict(raw=eligible, common=w > 0, standardized=w)
            else:
                modes = dict(raw=eligible, common=table.cell.map(lookup.common_support).fillna(False).to_numpy() & eligible)
            for name, weights in modes.items():
                finite = np.isfinite(table.spearman) & (weights > 0)
                expected = float(np.average(table.spearman[finite], weights=weights[finite])) if finite.any() else None
                native_analysis.check_native_metric(compared[regime][name]['score'], expected, 'independent weighted mean')
            for name, summary in result['similarity'][regime][epoch]['whole_groups'].items():
                mask = eligible & (table.similarity_class.to_numpy() == name) & np.isfinite(table.spearman)
                expected = float(table.spearman[mask].mean()) if mask.any() else None
                native_analysis.check_native_metric(summary['original_eligible_macro_spearman'] if summary['eligible_groups'] else None,
                    expected, 'independent whole-assay stratum mean')
    for name, digest in evidence.items():
        bind(name, {}, digest)
    if software() != code:
        raise ValueError('Reporting software changed during analysis')
    immutable(target, dict(source=source, result=result, payload_sha256=control_plan.digest_json(result)))
    return target


if __name__ == '__main__':
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use CPU-only Atum Prot2Mol')
    print(run(), flush=True)
