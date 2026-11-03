"""Native tiny predictions and all-variant comparison coverage, without fits."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization import requested_results as module
from experiments.paired_generalization.run import sha256, write_json
from test_paired_generalization_selected_analysis import native_pair


@pytest.fixture
def result_fixture(tmp_path, native_pair, monkeypatch):
    evaluations, refs = native_pair
    identity = 'full__A__seed42__lr1p0'; output = tmp_path / 'analysis'
    checkpoint = dict(id='checkpoint-50', path=str(tmp_path / 'checkpoint-50'))
    plan = dict(job=dict(id=identity, arm='full', regime='A', seed=42, learning_rate_multiplier=1.,
        config_path='fixture'), checkpoint=checkpoint,
        rules={'best_macro':dict(checkpoint=checkpoint['id'], epoch=5, step=50)})
    source = dict(scope_plan_sha256='fixed scope', design_sha256='fixture', noise_summary_sha256='fixture', software={})
    design = tmp_path / 'design'
    monkeypatch.setattr(module, 'provenance', lambda _: (plan, evaluations, design, .7, source, output))
    monkeypatch.setattr(module.diagnostics, 'references', lambda *args: refs)
    monkeypatch.setattr(module.diagnostics.similarity_composition, 'verify_design', lambda *args: None)
    from reward_model.training import config
    monkeypatch.setattr(config, 'load_reward_training_config', lambda *_: SimpleNamespace(model=SimpleNamespace(ranking_readout='ranking_head')))
    trajectory_source = tmp_path / 'trajectory_source.json'; write_json(trajectory_source, dict(synthetic_history=True))
    trajectory = dict(evidence_sha256={str(trajectory_source):sha256(trajectory_source)},
        cohorts=dict(val='native fixture val', train_probe='synthetic probe'))
    monkeypatch.setattr(module.encoder_analysis, 'trajectory', lambda *_: trajectory)
    path = module.analyze(identity)
    return path, module.read(path), trajectory_source


def test_selected_report_uses_native_metrics_and_has_no_seed_interval(result_fixture):
    path, result, _ = result_fixture
    verified = module.verify_analysis(path)
    assert verified == result and set(result['rules']) == {'best_macro'}
    assert not result['seed_uncertainty_available'] and not result['test_used_for_selection']
    stats = result['checkpoints']['checkpoint-50']['partitions']
    for split, prefix in (('val', 'eval'), ('test', 'test')):
        assert stats[split]['metrics']['ranking_score']['original_eligible_macro_spearman'] == pytest.approx(stats[split]['native_metrics'][prefix+'_macro_spearman'])
    before = sha256(path); assert module.analyze(result['job']['id']) == path and sha256(path) == before


@pytest.mark.parametrize('damage', ['payload', 'table', 'native_input'])
def test_report_rejects_changed_metrics_artifacts_or_trajectory_inputs(result_fixture, damage):
    path, result, native_input = result_fixture
    if damage == 'payload':
        result['checkpoints']['checkpoint-50']['regime'] = 'B'; write_json(path, result)
    elif damage == 'table':
        (path.parent / 'checkpoint-50/test/groups_ranking_score.parquet').write_bytes(b'changed')
    else: native_input.write_text('changed')
    with pytest.raises(ValueError): module.verify_analysis(path)


@pytest.fixture
def all_variants(tmp_path, result_fixture, monkeypatch):
    path, original, _ = result_fixture
    # Use actual native metric/table schemas; controlled synthetic shifts make
    # the expected paired changes independent of this comparison implementation.
    records = {}
    from experiments.paired_generalization.analyze_predictions import unit_bootstrap
    monkeypatch.setattr(module.controlled_summary, 'unit_bootstrap', lambda frame: unit_bootstrap(frame, repetitions=40))
    for i, arm in enumerate(module.ARMS):
        for regime in ('A', 'B'):
            record = copy.deepcopy(original)
            record['job'].update(id=f'{arm}__{regime}__seed42__lr1p0', arm=arm, regime=regime)
            directory = tmp_path / f'{arm}_{regime}'; directory.mkdir()
            for split in ('val', 'test'):
                for scorer in module.diagnostics.SCORERS:
                    relative = Path('checkpoint-50') / split / f'groups_{scorer}.parquet'
                    table = pd.read_parquet(path.parent / relative)
                    table['spearman'] = np.array([-.2, .1, .4]) + i*.01 + (0.05 if regime == 'A' else 0.)
                    if arm == 'lora4': table.loc[0, 'spearman'] = np.nan
                    target = directory / relative; target.parent.mkdir(parents=True, exist_ok=True)
                    table.to_parquet(target, index=False)
                    metric = record['checkpoints']['checkpoint-50']['partitions'][split]['metrics'][scorer]
                    metric['original_eligible_macro_spearman'] = float(table.spearman.mean())
            records[arm, regime] = (directory / 'complete.json', record)
    return records


def test_all_variants_matched_controls_and_common_defined_sensitivity(all_variants):
    result = module.comparisons(all_variants)
    assert list(result['metrics']) == list(module.ARMS)
    assert len(result['matched_changes']) == 14
    contrast = result['matched_changes']['pairwise_uniform_minus_pairwise_top']['B']['test']['ranking_score']
    assert contrast['mean_change'] == pytest.approx(.01)
    soft = result['matched_changes']['pairwise_noise_soft_minus_pairwise_hard_zero_margin']['A']['test']['ranking_score']
    assert soft['mean_change'] == pytest.approx(.01)
    assert result['common_cohort']['test']['A']['common_defined_eligible_groups'] == 2
    own = result['variant_relations']['test']['own_defined']; common = result['variant_relations']['test']['common_defined']
    assert own['statistics']['variants'] == common['statistics']['variants'] == 13
    assert own['points'][0]['A'] != common['points'][0]['A']
    assert common['statistics']['spearman'] == pytest.approx(1.)


@pytest.mark.parametrize('damage', ['missing_variant', 'seed', 'protocol', 'cohort', 'assay_membership'])
def test_incomplete_or_unmatched_comparisons_are_rejected(all_variants, damage):
    records = all_variants
    path, record = records['frozen', 'A']
    if damage == 'missing_variant': records.pop(('lora64', 'B'))
    elif damage == 'seed': record['job']['seed'] = 43
    elif damage == 'protocol': record['source']['noise_summary_sha256'] = 'changed'
    elif damage == 'cohort': record['native_trajectory']['cohorts']['val'] = 'changed'
    else:
        table_path = path.parent / 'checkpoint-50/test/groups_ranking_score.parquet'
        table = pd.read_parquet(table_path); table.loc[0,'group_id'] = 'wrong'; table.to_parquet(table_path, index=False)
    with pytest.raises(ValueError): module.comparisons(records)


def test_plot_retains_all_requested_variants_and_exports_png_svg(tmp_path, all_variants):
    result = module.comparisons(all_variants)
    destination = tmp_path / 'A_vs_B'; module.draw(result, destination)
    assert destination.with_suffix('.png').read_bytes().startswith(b'\x89PNG')
    svg = destination.with_suffix('.svg').read_text()
    assert 'seed42, fixed base LR' in svg and 'common defined' in svg
