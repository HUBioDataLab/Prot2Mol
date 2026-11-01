import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization import adaptation_design as design


def frames_fixture():
    frames, unit_map = {}, {}
    for role, sizes in [('train', [2, 3]), ('val', [2, 4, 6, 8]), ('test', [1, 3, 6, 10])]:
        rows = []
        for u, assays in enumerate(sizes):
            for a in range(assays):
                assay = f'{role}_u{u}_a{a}'
                unit_map[assay] = len(unit_map) if role == 'train' else (100 if role == 'val' else 200)+u
                for m, value in enumerate([5., 6., 7.]):
                    rows.append(dict(assay_id=assay, target_chembl_id='protein',
                                     compound_id=f'{assay}_m{m}', pchembl_value=value))
        frames[role] = pd.DataFrame(rows)
    return frames, pd.Series(unit_map)


def inventories(regime='B'):
    frames, units = frames_fixture()
    return {role:design.group_inventory(frame, regime, role, units) for role,frame in frames.items()}


def test_disjoint_units_nested_whole_assays_and_original_test_preserved():
    inv = inventories()
    groups, _, summary = design.build_design(inv, 'B', k_values=(1, 2, 4, 64))
    roles = {role:set(groups.loc[groups.probe_role == role, 'unit']) for role in ('adaptation','validation','test')}
    assert not roles['adaptation'] & roles['validation']
    assert not roles['adaptation'] & roles['test']
    assert not roles['validation'] & roles['test']
    assert set(groups.loc[groups.probe_role == 'test', 'group_id']) == set(inv['test'].group_id)
    previous = []
    for k in (1, 2, 4):
        selected = summary['across_unit'][str(k)]['selected_group_ids']
        assert selected[:len(previous)] == previous and len(selected) == k
        assert summary['across_unit'][str(k)]['rows'] == 3*k
        previous = selected
    assert not summary['across_unit']['64']['available']
    assert summary['across_unit']['64']['selected_group_ids'] == []
    assert summary['fixed_test_groups_preserved']


def test_membership_is_invariant_to_labels_and_input_row_order():
    frames, units = frames_fixture()
    original = {role:design.group_inventory(f, 'B', role, units) for role,f in frames.items()}
    changed = {role:design.group_inventory(f.assign(pchembl_value=0.).iloc[::-1], 'B', role, units) for role,f in frames.items()}
    a, _, sa = design.build_design(original, 'B', k_values=(1, 2, 4))
    b, _, sb = design.build_design(changed, 'B', k_values=(1, 2, 4))
    columns = ['group_id','unit','probe_role','adaptation_order','within_role','within_support_order']
    pd.testing.assert_frame_equal(a[columns], b[columns])
    assert sa['within_unit']['1']['test']['rank_evaluable_units'] > 0
    assert sb['within_unit']['1']['test']['rank_evaluable_units'] == 0
    assert not sb['within_unit']['1']['validation_selection_possible']


def test_fixed_within_unit_queries_remain_disjoint_from_every_support_prefix():
    groups, units, summary = design.build_design(inventories(), 'B', k_values=(1, 2, 4, 64))
    for (_, unit), g in groups[groups.probe_role != 'adaptation'].groupby(['probe_role','unit']):
        support = g[g.within_role == 'support'].sort_values('within_support_order')
        query = g[g.within_role == 'query']
        assert len(support) == len(g)//2 and len(query) == len(g)-len(g)//2
        assert support.within_support_order.tolist() == list(range(len(support)))
        for k in (1, 2, 4):
            assert not set(support.head(k).group_id) & set(query.group_id)
    assert not summary['within_unit']['64']['validation_selection_possible']
    assert not summary['within_unit']['64']['test_rank_evaluation_possible']
    assert units.query_groups.min() >= 1


def test_a_single_assay_units_cannot_supply_disjoint_within_unit_groups():
    groups, units, summary = design.build_design(inventories('A'), 'A', k_values=(1, 2))
    assert units.support_capacity.eq(0).all()
    assert all(v['test']['units'] == 0 and v['validation']['units'] == 0 for v in summary['within_unit'].values())
    assert groups.loc[groups.probe_role != 'adaptation', 'within_role'].eq('query').all()


def test_cross_partition_unit_overlap_and_invalid_budgets_are_rejected():
    inv = inventories()
    inv['test'].loc[0, 'unit'] = inv['train'].unit.iloc[0]
    with pytest.raises(ValueError, match='units cross partitions'):
        design.build_design(inv, 'B')
    for values in [(0, 1), (2, 1), (1, 1), (1, 1.5)]:
        with pytest.raises(ValueError, match='k must be'):
            design.build_design(inventories(), 'B', k_values=values)


def test_inventory_uses_native_float32_eligibility_without_splitting_assays():
    frames, units = frames_fixture()
    frame = frames['val'].iloc[:3].copy()
    frame['pchembl_value'] = [0., .2, .499999999]
    group = design.group_inventory(frame, 'B', 'val', units)
    assert len(group) == 1 and group.rows.iloc[0] == 3 and group.ranking_eligible.iloc[0]
    with pytest.raises(ValueError, match='Repeated curated'):
        design.group_inventory(pd.concat([frame, frame.iloc[:1]]), 'B', 'val', units)
    with pytest.raises(ValueError, match='Missing scaffold'):
        design.group_inventory(frame, 'B', 'val', pd.Series(dtype=int))
