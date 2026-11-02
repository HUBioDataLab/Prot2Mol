"""Matched full-horizon LR reports preserve scope, coverage and input evidence."""
import copy
from pathlib import Path

import pandas as pd
import pytest

from experiments.paired_generalization import native_lr_comparison as m
from experiments.paired_generalization.run import sha256, write_json


def config(scale=1.):
    return dict(training=dict(output_dir=str(scale), seed=42, num_train_epochs=15,
        **{name: rate*scale for name, rate in zip(m.RATES, (1e-4, 1e-4, 1e-5, 1e-5))}),
        data={'split': 'A'}, model={'ranking_loss_weight': .25})


def trajectory(offset=0.):
    rows = []
    for epoch in range(1, 16):
        row = dict(epoch=epoch, step=epoch*802)
        for prefix in ('eval', 'train_eval'):
            value = epoch/100+offset
            row[prefix] = dict(prefix=prefix, rows=100,
                metrics={'ranking_score': dict(eligible_groups=10, eligible_defined_fraction=1.,
                    original_eligible_macro_spearman=value, matched_eligible_ndcg={'1':value+.4, '3':value+.5, '5':value+.6})},
                native_metrics={prefix+'_'+name: 3.-value for name in ('loss', 'classification_loss', 'ranking_loss', 'contrastive_loss')})
        rows.append(row)
    return rows


def test_only_the_four_rates_and_output_directory_may_differ():
    first, second = config(), config(.3)
    saved = copy.deepcopy((first, second))
    result = m.check_configurations(first, second)
    assert tuple(result) == m.RATES and result['learning_rate'] == [1e-4, 3e-5]
    assert (first, second) == saved


@pytest.mark.parametrize('damage', ['seed', 'horizon', 'data', 'loss', 'one_rate'])
def test_other_configuration_changes_are_rejected(damage):
    second = config(.3)
    if damage == 'seed': second['training']['seed'] = 43
    elif damage == 'horizon': second['training']['num_train_epochs'] = 20
    elif damage == 'data': second['data']['split'] = 'B'
    elif damage == 'loss': second['model']['ranking_loss_weight'] = .5
    else: second['training']['protein_encoder_learning_rate'] = 1e-5
    with pytest.raises(ValueError): m.check_configurations(config(), second)


def test_all_fifteen_matched_differences_are_reported_without_selection():
    result = m.compare(trajectory(), trajectory(-.01))
    assert len(result['rows']) == 15
    assert result['descriptive_validation_peak_epochs'] == {'reference':15, 'lower_lr':15}
    assert result['lower_lr_validation_below_reference_at_all_epochs']
    for row in result['rows']:
        assert row['difference']['eval']['macro_spearman'] == pytest.approx(-.01)
        assert row['difference']['train_eval']['native_losses']['loss'] == pytest.approx(.01)


@pytest.mark.parametrize('damage', ['missing_epoch', 'duplicate_epoch', 'step', 'rows', 'groups', 'coverage'])
def test_comparison_rejects_unmatched_horizon_or_headline_cohort(damage):
    second = trajectory(-.01)
    if damage == 'missing_epoch': second.pop()
    elif damage == 'duplicate_epoch': second[-1]['epoch'] = 14
    elif damage == 'step': second[3]['step'] += 1
    elif damage == 'rows': second[3]['eval']['rows'] -= 1
    elif damage == 'groups': second[3]['eval']['metrics']['ranking_score']['eligible_groups'] -= 1
    else: second[3]['train_eval']['metrics']['ranking_score']['eligible_defined_fraction'] = .9
    with pytest.raises(ValueError): m.compare(trajectory(), second)


def test_tied_descriptive_peak_keeps_the_earlier_epoch():
    first = trajectory(); first[-1]['eval'] = copy.deepcopy(first[-2]['eval'])
    assert m.compare(first, first)['descriptive_validation_peak_epochs']['reference'] == 14


def test_changed_trajectory_is_rejected_before_reading_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'WORK', tmp_path)
    monkeypatch.setattr(m.native, 'registration', lambda identity: ({}, {}, {}))
    monkeypatch.setattr(m.driver, 'completed_job', lambda job: True)
    monkeypatch.setattr(m.native, 'software', lambda: {})
    path = tmp_path/'tier1/native_trajectories'/m.JOBS[0]/m.native.digest_json({})[:16]/'trajectory_through_epoch15.json'
    write_json(path, {})
    with pytest.raises(ValueError, match='source changed'): m.load(m.JOBS[0], {})


@pytest.fixture
def report(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'WORK', tmp_path)
    monkeypatch.setattr(m, 'runtime_guard', lambda: None)
    source = tmp_path/'input.json'; write_json(source, dict(fixed=True))
    inputs = []
    for scale, offset in ((1., 0.), (.3, -.01)):
        inputs.append(dict(trajectory={'trajectory':trajectory(offset)}, qualification={'same':True},
            config=config(scale), references={'sources':{'source':'fixed'}, 'probe':{'seed':42}},
            frames={p:pd.DataFrame({'id':[1,2], 'label':[5.,7.]}) for p in ('eval', 'train_eval')}))
    def load(identity, evidence):
        evidence[str(source)] = sha256(source)
        return inputs[m.JOBS.index(identity)]
    monkeypatch.setattr(m, 'load', load)
    return dict(inputs=inputs, source=source, work=tmp_path)


def test_report_preserves_sources_and_does_not_select_or_retrain(report):
    path = m.run(); result = m.read(path)
    assert result['input_sha256'][str(report['source'])] == sha256(report['source'])
    assert not any(result[k] for k in ('new_fits_created', 'model_inference', 'test_predictions_read', 'learning_rate_selected', 'independent_seed_variance_estimated'))
    with pytest.raises(RuntimeError, match='already exists'): m.run()


@pytest.mark.parametrize('damage', ['qualification', 'identity', 'label', 'source_during_analysis'])
def test_report_rejects_different_data_or_evidence_changed_during_comparison(report, monkeypatch, damage):
    if damage == 'qualification': report['inputs'][1]['qualification'] = {'same':False}
    elif damage in ('identity', 'label'):
        report['inputs'][1]['frames']['eval'].loc[0, 'id' if damage == 'identity' else 'label'] += 1
    else:
        compare = m.compare
        def changed(*args):
            result = compare(*args); report['source'].write_text('changed'); return result
        monkeypatch.setattr(m, 'compare', changed)
    with pytest.raises((ValueError, AssertionError)): m.run()
    assert not (report['work']/'tier1/native_lr_comparison/full_A_seed42_lr0p3_vs_lr1p0/complete.json').exists()


def test_runtime_requires_authorized_project_and_no_gpu_visibility(monkeypatch):
    monkeypatch.setattr(m, 'ROOT', Path('/home/atabey/Prot2Mol'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(RuntimeError): m.runtime_guard()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', ''); m.runtime_guard()
