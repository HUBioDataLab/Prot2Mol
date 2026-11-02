"""Reference-only composition joins, evidence gates and analytic weighting."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization import reference_composition as module
from experiments.paired_generalization.run import write_json, sha256
from test_paired_generalization_similarity_composition import population


def score_table(design):
    return design[['group_id', 'size', 'spearman']].assign(
        original_ranking_eligible=design.eligible, component=design.unit_id)


@pytest.mark.parametrize('damage', ['duplicate', 'missing', 'extra', 'size', 'eligibility', 'unit'])
def test_join_rejects_changed_cohorts_and_units(damage):
    design = population(['low', 'high'], [.2, .8]); scores = score_table(design)
    if damage == 'duplicate': scores = pd.concat([scores, scores.iloc[[0]]])
    elif damage == 'missing': scores = scores.iloc[:1]
    elif damage == 'extra': scores.loc[0, 'group_id'] = 'unknown'
    elif damage == 'size': scores.loc[0, 'size'] = 4
    elif damage == 'eligibility': scores.loc[0, 'original_ranking_eligible'] = False
    else: scores.loc[0, 'component'] = 99
    with pytest.raises(ValueError): module.join_scores(design, scores)


def test_join_aligns_by_identity_and_keeps_undefined_and_zero_eligible_units():
    design = population(['low', 'high', 'high'], [.2, np.nan, .7])
    design.loc[2, 'eligible'] = False
    result = module.join_scores(design, score_table(design).iloc[::-1])
    np.testing.assert_allclose(result.spearman, [.2, np.nan, .7], equal_nan=True)
    assert len(result) == 3 and result.unit_id.nunique() == 3


@pytest.fixture
def references(tmp_path, monkeypatch):
    """Synthetic audit receipts exercise assembly, not scientific verification."""
    root = tmp_path/'project'; work = root/'outputs'; design = work/'tier1/similarity_composition/design'
    root.mkdir(); monkeypatch.setattr(module, 'ROOT', root); monkeypatch.setattr(module, 'WORK', work)
    native_code = dict(source_sha256={}, packages={})
    monkeypatch.setattr(module.native_analysis, 'software', lambda: copy.deepcopy(native_code))
    monkeypatch.setattr(module, 'software', lambda: dict(synthetic_reporting=True))
    monkeypatch.setattr(module.control_driver, 'completed_job', lambda job: True)
    primary = work/'controlled/plan.json'; write_json(primary, {})
    units = work/'tier1/data/assay_units.parquet'; units.parent.mkdir(parents=True); units.write_bytes(b'synthetic units')
    frames = {'A': population(['low', 'low', 'high'], [.6, .6, 0.]),
              'B': population(['low', 'high', 'high', 'high'], [.1, .1, .1, .1])}
    for regime, frame in frames.items():
        frame['similarity_class'] = 'all_shared' if regime == 'A' else 'all_unseen'
        excluded = frame.iloc[[0]].copy(); excluded['group_id'] = 'excluded'; excluded['unit_id'] = len(frame)
        excluded['size'] = 1; excluded['eligible'] = False; excluded['spearman'] = np.nan
        frames[regime] = pd.concat([frame, excluded], ignore_index=True)
    design.mkdir(parents=True)
    for regime, frame in frames.items(): frame.to_parquet(design/f'groups_{regime}_val.parquet', index=False)
    module.composition.composition_weights(frames['A'], frames['B']).to_parquet(design/'weights_val.parquet', index=False)
    design_record = dict(artifacts_sha256={p.name: sha256(p) for p in design.glob('*.parquet')})
    write_json(design/'complete.json', design_record)
    monkeypatch.setattr(module.composition, 'verify_design', lambda path: design_record)
    jobs = {}; registrations = {}; directories = {}
    for regime in ('A', 'B'):
        identity = module.control_plan.job_id('full', regime, 42, 1.)
        run = work/'controlled/runs'/identity; config = work/'controlled/configs'/f'{identity}.yaml'
        config.parent.mkdir(parents=True, exist_ok=True); config.write_text('synthetic: true\n')
        job = dict(id=identity, regime=regime, arm='full', seed=42, learning_rate_multiplier=1.,
            output_dir=str(run), config_path=str(config), config_sha256=sha256(config), plan_sha256=sha256(primary))
        jobs[identity] = job; job_path = work/'controlled/jobs'/f'{identity}.json'; write_json(job_path, job)
        write_json(run/'launch_provenance.json', dict(synthetic=True)); write_json(run/'controlled_training_complete.json', dict(synthetic=True))
        runtime = dict(world_size=4, sources={}, signature=module.control_plan.digest_json(dict(sources={}, plan_sha256=sha256(primary))))
        write_json(run/'runtime_qualification.json', runtime)
        registration = dict(job_metadata_sha256=sha256(job_path), launch_sha256=sha256(run/'launch_provenance.json'))
        registrations[identity] = registration
        directory = work/'tier1/native_trajectories'/identity/module.control_plan.digest_json(native_code)[:16]
        directories[regime] = directory; write_json(directory/'references/complete.json', dict(synthetic=True))
        plan = dict(**registration, software=native_code, reference_sha256=sha256(directory/'references/complete.json'),
            units_sha256=sha256(units), similarity_design_sha256=sha256(design/'complete.json'))
        write_json(directory/'analysis_plan.json', plan); rows = []; checks = []
        for epoch in range(1, 16):
            offset = .01*(1-abs(epoch-({'A': 4, 'B': 6}[regime]))/15)
            frame = frames[regime].copy(); frame['spearman'] += offset
            macro = float(frame.loc[frame.eligible, 'spearman'].mean())
            row = dict(epoch=epoch, step=epoch*10)
            for prefix in ('eval', 'train_eval'):
                manifest = run/'prediction_exports/trajectory'/f'{prefix}_step{epoch*10:08d}.json'
                manifest.parent.mkdir(parents=True, exist_ok=True); payload = manifest.with_suffix('.npz'); payload.write_bytes(b'synthetic predictions')
                metrics = {prefix+'_macro_spearman': macro}
                write_json(manifest, dict(predictions_file=payload.name, predictions_sha256=sha256(payload), metrics=metrics))
                table = directory/manifest.stem/'groups_ranking_score.parquet'; table.parent.mkdir(parents=True)
                score_table(frame).to_parquet(table, index=False)
                similarity = dict(whole_groups={name: dict(eligible_groups=int((frame.eligible & (frame.similarity_class == name)).sum()),
                    original_eligible_macro_spearman=macro if (frame.similarity_class == name).any() else None)
                    for name in ('all_shared', 'mixed', 'all_unseen')}, within_stratum=dict(synthetic_reused=True))
                row[prefix] = dict(epoch=epoch, step=epoch*10, prefix=prefix, ranking_readout='ranking_head',
                    source=dict(manifest=str(manifest), manifest_sha256=sha256(manifest), payload_sha256=sha256(payload), provenance=plan),
                    native_metrics=metrics, metrics=dict(ranking_score=dict(original_eligible_macro_spearman=macro)),
                    similarity=similarity, artifacts_sha256={table.name: sha256(table)})
                checks.append(dict(epoch=epoch, prefix=prefix, macros=dict(ranking_score=macro)))
            rows.append(row)
        trajectory = dict(job=job, horizon=15, complete_horizon_observed=True, epochs_analyzed=list(range(1, 16)),
            test_predictions_read=False, analysis_plan_sha256=sha256(directory/'analysis_plan.json'), trajectory=rows)
        path = directory/'trajectory_through_epoch15.json'; write_json(path, trajectory)
        write_json(directory/'verification_through_epoch15.json', dict(trajectory_sha256=sha256(path), checks=checks,
            test_predictions_read=False, source_and_artifact_hashes_verified=True,
            analysis_plan_sha256=trajectory['analysis_plan_sha256'], verifier_sha256=sha256(Path(module.verify_native_analysis.__file__))))
    monkeypatch.setattr(module.native_analysis, 'registration', lambda identity: (jobs[identity], {}, registrations[identity]))
    return dict(work=work, directories=directories, frames=frames)


def test_full_report_analytic_weights_peaks_and_no_test_reads(references, monkeypatch):
    original = pd.read_parquet; read_paths = []
    def track(path, *args, **kwargs):
        assert 'test' not in Path(path).name
        read_paths.append(path); return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, 'read_parquet', track)
    path = module.run(); result = module.read(path)['result']
    assert len(result['same_epoch']) == 15
    assert result['own_native_macro_peaks']['epochs'] == {'A': 4, 'B': 6}
    for epoch, record in result['same_epoch'].items():
        offset = .01*(1-abs(int(epoch)-4)/15)
        assert record['A']['raw']['score'] == pytest.approx(.4+offset)
        assert record['A']['standardized']['score'] == pytest.approx(.15+offset)
        assert record['A']['standardized']['units'] == 4
        assert record['B']['common']['units'] == 5
        assert record['A']['standardization_change']['score'] == pytest.approx(-.25)
    assert result['validation_only'] and not result['test_predictions_read'] and result['new_fits_created'] == 0
    assert not result['study_complete'] and len(read_paths) > 30
    assert module.run() == path
    damaged = module.read(path); damaged['result']['same_epoch'] = {}; write_json(path, damaged)
    with pytest.raises(ValueError, match='report changed'): module.run()


@pytest.mark.parametrize('damage', ['missing_checks', 'verification_flag', 'source_hash', 'wrong_role', 'table_payload', 'training_runtime'])
def test_incomplete_or_changed_upstream_evidence_is_rejected(references, damage):
    d = references['directories']['B']; v = d/'verification_through_epoch15.json'; t = d/'trajectory_through_epoch15.json'
    verified = module.read(v)
    if damage == 'missing_checks': verified['checks'].pop()
    elif damage == 'verification_flag': verified['source_and_artifact_hashes_verified'] = False
    elif damage == 'source_hash': verified['trajectory_sha256'] = 'changed'
    elif damage == 'wrong_role':
        record = module.read(t); record['trajectory'][0]['eval']['prefix'] = 'test'
        write_json(t, record); verified['trajectory_sha256'] = sha256(t)
    elif damage == 'table_payload': (d/'eval_step00000010/groups_ranking_score.parquet').write_bytes(b'changed')
    else:
        run = references['work']/'controlled/runs/full__B__seed42__lr1p0'
        record = module.read(run/'runtime_qualification.json'); record['world_size'] = 2
        write_json(run/'runtime_qualification.json', record)
    write_json(v, verified)
    with pytest.raises(ValueError): module.run()
    assert not list((references['work']/'tier1/reference_composition').glob('*/complete.json'))


def test_incomplete_training_blocks_report(references, monkeypatch):
    monkeypatch.setattr(module.control_driver, 'completed_job', lambda job: False)
    with pytest.raises(RuntimeError, match='incomplete'): module.run()


def test_mutation_during_comparison_is_rejected(references, monkeypatch):
    original = module.composition.compare_composition; changed = []
    def mutate(*args):
        result = original(*args)
        if not changed:
            path = references['directories']['B']/'eval_step00000010/groups_ranking_score.parquet'
            path.write_bytes(b'changed'); changed.append(path)
        return result
    monkeypatch.setattr(module.composition, 'compare_composition', mutate)
    with pytest.raises(ValueError, match='source changed'): module.run()
    assert not list((references['work']/'tier1/reference_composition').glob('*/complete.json'))
