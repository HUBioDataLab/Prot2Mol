import copy
from dataclasses import asdict, replace

import pytest

from experiments.paired_generalization import adaptation_protocol as protocol
from experiments.paired_generalization import adaptation_design as design
from experiments.paired_generalization.adaptation_views import view_requests
from experiments.paired_generalization.control_plan import GPU_UUIDS, digest_json
from experiments.paired_generalization.qualify_controls import make_model
from test_paired_generalization_adaptation_design import inventories
from test_paired_generalization_checkpoint_evaluation import config_fixture


def fixture_protocol():
    primary = dict(tuning_seed=42, confirmation_seeds=[43, 44], gpu_uuids=list(GPU_UUIDS),
        bases={r:dict(training=dict(per_device_train_batch_size=12,
            gradient_accumulation_steps=4, num_train_epochs=15)) for r in ('A', 'B')})
    records, groups = {}, {}
    for regime in ('A', 'B'):
        frame, _, _ = design.build_design(inventories(regime), regime)
        groups[regime] = frame
        records[regime] = dict(k_values=list(design.K_VALUES), **view_requests(frame))
    return primary, records, groups


def test_accounting_includes_each_available_unit_seed_scope_and_no_extra_lr():
    primary, records, groups = fixture_protocol()
    plan = protocol.build_protocol(primary, records, groups)
    assert plan['seeds'] == [42, 43, 44]
    assert plan['accounting']['additional_learning_rates'] == 0
    assert not plan['model_fitting'] and not plan['production_runtime_qualified']
    expected = 0
    for cell in plan['cells']:
        assert cell['selection_regime'] == 'B' and cell['selection_seed'] == 42
        assert cell['validation_groups']
        for regime in ('A', 'B'):
            ids = [identity for values in cell['task_bindings'][regime].values() for identity in values]
            assert len(ids) == len(set(ids)) == cell['fits_per_seed'][regime]
            for identity in ids:
                task = records[regime]['tasks'][identity]
                assert task['kind'] == cell['kind'] and task['k'] == cell['k']
                assert task['support_view'] is not None
                if task['kind'] == 'within':
                    support = set(records[regime]['views'][task['support_view']]['group_ids'])
                    query = set(records[regime]['views'][task['query_view']]['group_ids'])
                    assert not support & query
            expected += len(ids)*3
        if cell['kind'] == 'within':
            assert cell['fits_per_seed']['A'] == 0
    assert plan['accounting']['support_fits'] == expected
    assert any(row['k'] == 64 and row['kind'] == 'within' for row in plan['unavailable'])


def cell_and_rows():
    mapping = {'a': 'unit0', 'b': 'unit0', 'c': 'unit1'}
    cell = dict(id='within__heads__k1', kind='within', scope='heads', k=1,
                validation_groups=mapping, validation_groups_sha256=digest_json(mapping))
    rows = [dict(regime='B', seed=42, role='validation', kind='within', scope='heads', k=1,
                 epoch=epoch, group_id=group, unit=unit, rho=.1)
            for epoch in range(1, 16) for group, unit in mapping.items()]
    return cell, rows


def test_selection_is_macro_over_assays_not_unit_means_and_earliest_ties():
    cell, rows = cell_and_rows()
    for row in rows:
        if row['epoch'] in (2, 3):
            row['rho'] = -1. if row['group_id'] == 'c' else 1.
        elif row['epoch'] == 1:
            row['rho'] = .2
    result = protocol.select_epoch(rows, cell)
    assert result['selected']['epoch'] == 2
    assert result['selected']['macro_spearman'] == pytest.approx(1/3)
    assert result['selected']['defined_fraction'] == 1
    assert not result['test_queries_used'] and not result['confirmation_seeds_used']
    assert protocol.select_epoch(rows[::-1], cell) == result


def test_undefined_correlations_keep_native_metric_and_explicit_coverage():
    cell, rows = cell_and_rows()
    for row in rows:
        row['rho'] = .8
        if row['epoch'] == 1:
            row['rho'] = 1. if row['group_id'] == 'a' else float('nan')
    result = protocol.select_epoch(rows, cell)
    assert result['selected']['epoch'] == 1
    assert result['selected']['defined_assays'] == 1
    assert result['selected']['defined_fraction'] == pytest.approx(1/3)
    for row in rows:
        row['rho'] = None
    result = protocol.select_epoch(rows, cell)
    assert result['status'] == 'not_selectable' and result['selected'] is None
    assert all(candidate['macro_spearman'] is None for candidate in result['candidates'])


@pytest.mark.parametrize('field,value', [('regime', 'A'), ('seed', 43), ('role', 'test'),
    ('scope', 'heads_cross_attention'), ('kind', 'across'), ('k', 2)])
def test_other_split_seed_or_test_queries_cannot_select(field, value):
    cell, rows = cell_and_rows()
    rows[0][field] = value
    with pytest.raises(ValueError, match='Only the registered B seed42'):
        protocol.select_epoch(rows, cell)


def test_missing_duplicate_wrong_unit_and_zero_epoch_are_rejected():
    cell, rows = cell_and_rows()
    with pytest.raises(ValueError, match='Every epoch'):
        protocol.select_epoch(rows[:-1], cell)
    with pytest.raises(ValueError, match='duplicate'):
        protocol.select_epoch(rows+[rows[0]], cell)
    for field, value in [('unit', 'other_unit'), ('epoch', 0), ('group_id', 'test_assay')]:
        changed = copy.deepcopy(rows)
        changed[0][field] = value
        with pytest.raises(ValueError, match='Wrong epoch'):
            protocol.select_epoch(changed, cell)


def test_invalid_correlation_and_changed_inventory_are_rejected():
    cell, rows = cell_and_rows()
    rows[0]['rho'] = float('inf')
    with pytest.raises(ValueError, match='Invalid assay correlation'):
        protocol.select_epoch(rows, cell)
    cell['validation_groups']['a'] = 'changed'
    with pytest.raises(ValueError, match='frozen nonempty'):
        protocol.select_epoch(rows, cell)


def test_config_preserves_optimizer_losses_and_schedule_with_nominal_batch_192(tmp_path):
    base = config_fixture(make_model(None, {}), tmp_path)
    base = replace(base, training=replace(base.training, num_train_epochs=15,
        gradient_accumulation_steps=4, learning_rate=.001, encoder_learning_rate=.0001,
        projection_learning_rate=.0003, weight_decay=.1, warmup_ratio=.2))
    original = copy.deepcopy(base)
    adapted = protocol.adaptation_config(base, tmp_path/'fit')
    assert base == original and adapted.model == base.model and adapted.data == base.data
    assert adapted.training.per_device_train_batch_size*adapted.training.gradient_accumulation_steps == 192
    changed = {key for key, value in asdict(base.training).items()
               if asdict(adapted.training)[key] != value}
    assert changed <= {'output_dir', 'training_mode', 'gradient_accumulation_steps', 'report_to',
                       'evaluate_test', 'fixed_train_eval_assays', 'export_evaluation_predictions',
                       'objective_gradient_diagnostics_steps'}
    with pytest.raises(ValueError, match='registered full-run'):
        protocol.adaptation_config(replace(base, training=replace(base.training, max_steps=3)), tmp_path/'bad')
