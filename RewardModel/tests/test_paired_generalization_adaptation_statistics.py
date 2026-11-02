import copy
import json

import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization import adaptation_statistics as stats


def inventory():
    return pd.DataFrame(dict(group_id=['g1','g2','g3','g4','g5'], unit=['u1','u1','u2','u3','u4'],
        rows=[3,3,3,2,3], ranking_eligible=[True,True,True,False,True]))


def records(groups, values):
    return [dict(scorer=scorer, group_id=g.group_id, unit=g.unit, rows=int(g.rows),
        eligible=bool(g.ranking_eligible), defined=values[g.group_id] is not None,
        rho=values[g.group_id]) for scorer in stats.SCORERS for g in groups.itertuples()]


def fixture():
    full = inventory()
    smaller = full[full.unit.isin(['u1','u3'])]
    base = records(full, dict(g1=0.,g2=.8,g3=-.5,g4=None,g5=.2))
    adapted = {1:records(full, dict(g1=.5,g2=None,g3=0.,g4=None,g5=.4)),
               2:records(smaller, dict(g1=.6,g2=.9,g4=None))}
    return base, adapted, {1:full,2:smaller}


def test_macro_is_assay_weighted_and_changes_are_paired_on_defined_assays():
    report = stats.compare_budgets(*fixture())
    result = report['available']['ranking_score']['1']
    assert result['baseline']['macro_spearman'] == pytest.approx(.125)
    assert result['adapted']['macro_spearman'] == pytest.approx(.3)
    # Pairing excludes g2 from both sides; it is not the difference of raw macros.
    assert result['adapted_minus_baseline']['mean_change'] == pytest.approx((.5+.5+.2)/3)
    assert result['adapted_minus_baseline']['mutually_defined_groups'] == 3
    assert result['adapted']['groups'] == 5 and result['adapted']['eligible_groups'] == 4
    assert result['adapted']['defined_groups'] == 3
    assert result['adapted']['eligible_defined_fraction'] == .75
    assert result['adapted_minus_baseline']['unit_bootstrap']['units'] == 4
    assert result['adapted_minus_baseline']['unit_bootstrap']['repetitions'] == 2000
    json.dumps(report, allow_nan=False)


def test_common_unit_and_common_defined_sensitivities_preserve_zero_eligible_units():
    report = stats.compare_budgets(*fixture())
    assert report['common_units'] == ['u1','u3']
    assert report['common_group_ids'] == ['g1','g2','g4']
    common = report['common_units_statistics']['ranking_score']
    assert common['1']['baseline']['macro_spearman'] == .4
    assert common['1']['adapted']['macro_spearman'] == .5
    assert common['2']['adapted']['macro_spearman'] == .75
    fixed = report['common_defined_statistics']['ranking_score']
    assert fixed['defined_group_ids'] == ['g1']
    assert fixed['1']['baseline']['macro_spearman'] == 0.
    assert fixed['2']['baseline']['macro_spearman'] == 0.
    assert fixed['2']['adapted']['macro_spearman'] == .6
    for k in ('1','2'):
        assert fixed[k]['adapted']['groups'] == 3
        assert fixed[k]['adapted']['units'] == 2
        assert fixed[k]['adapted_minus_baseline']['unit_bootstrap']['units'] == 2
    change = report['adjacent_budget_changes']['1->2']
    assert change['units'] == ['u1','u3']
    assert change['pathways']['ranking_score']['mean_change'] == pytest.approx(.1)


def test_reordering_input_is_invariant_and_inputs_are_not_mutated():
    base, adapted, groups = fixture()
    original = copy.deepcopy((base, adapted, groups))
    expected = stats.compare_budgets(base, adapted, groups)
    actual = stats.compare_budgets(base[::-1], {k:v[::-1] for k,v in adapted.items()},
                                  {k:v.iloc[::-1] for k,v in groups.items()})
    assert actual == expected
    assert base == original[0] and adapted == original[1]
    for k in groups:pd.testing.assert_frame_equal(groups[k], original[2][k])


def test_disjoint_budget_cohorts_are_reported_without_a_fake_paired_effect():
    full = inventory()
    groups = {1:full[full.unit=='u1'],2:full[full.unit=='u2']}
    union = pd.concat(groups.values())
    values = dict(g1=0.,g2=.5,g3=.2)
    result = stats.compare_budgets(records(union, values),
        {k:records(g, values) for k,g in groups.items()}, groups)
    assert result['common_units'] == []
    assert all(v is None for v in result['common_units_statistics'].values())
    assert result['adjacent_budget_changes']['1->2']['pathways'] is None


def test_all_undefined_pathway_remains_null_not_zero_and_no_seed_interval_is_invented():
    base, adapted, groups = fixture()
    for recs in [base, *adapted.values()]:
        for row in recs:
            if row['scorer']=='cosine_similarity':row.update(defined=False, rho=None)
    result = stats.compare_budgets(base, adapted, groups)
    for k in ('1','2'):
        value = result['available']['cosine_similarity'][k]
        assert value['adapted']['macro_spearman'] is None
        assert value['adapted_minus_baseline']['mean_change'] is None
        assert value['adapted_minus_baseline']['unit_bootstrap']['interval'] is None
    seeds = stats.summarize_seeds({str(s):copy.deepcopy(result) for s in stats.SEEDS})
    assert seeds['seed_spread']['available']['cosine_similarity']['1']['all_three']['adapted']['defined']==0
    assert 'interval' not in seeds['seed_spread']['available']['ranking_score']['1']['all_three']['adapted']


@pytest.mark.parametrize('change', ['missing','duplicate','unit','size','eligible','defined','nan','range','bool','null','scorer'])
def test_malformed_query_records_are_rejected(change):
    base, _, groups = fixture()
    if change=='missing':base.pop()
    elif change=='duplicate':base.append(copy.deepcopy(base[0]))
    elif change=='unit':base[0]['unit']='another_unit'
    elif change=='size':base[0]['rows']=4
    elif change=='eligible':base[0]['eligible']=False
    elif change=='defined':base[3].update(defined=True,rho=.5)
    elif change=='nan':base[0]['rho']=float('nan')
    elif change=='range':base[0]['rho']=1.01
    elif change=='bool':base[0]['rho']=True
    elif change=='null':base[0].update(defined=False,rho=.1)
    elif change=='scorer':base[0]['scorer']='unregistered'
    with pytest.raises(ValueError):stats.assay_tables(base, groups[1])


@pytest.mark.parametrize('change', ['partial_unit','eligibility','size','unit','missing_budget','zero','empty'])
def test_changing_cohorts_or_budgets_are_rejected(change):
    base, adapted, groups = fixture()
    groups = {k:v.copy() for k,v in groups.items()}
    if change=='partial_unit':groups[2]=groups[2][groups[2].group_id!='g2']
    elif change=='eligibility':groups[2].loc[groups[2].group_id=='g1','ranking_eligible']=False
    elif change=='size':groups[2].loc[groups[2].group_id=='g1','rows']=4
    elif change=='unit':groups[2].loc[groups[2].group_id=='g1','unit']='changed'
    elif change=='missing_budget':adapted.pop(2)
    elif change=='zero':groups[0]=groups.pop(2);adapted[0]=adapted.pop(2)
    elif change=='empty':groups[2]=groups[2].iloc[:0]
    with pytest.raises(ValueError):stats.compare_budgets(base, adapted, groups)


def test_seed_spread_separates_tuning_exposure_and_confirmation_without_pooling_units():
    results = {}
    for seed, delta in zip(stats.SEEDS, [0.,.1,-.1]):
        base, adapted, groups = fixture()
        for rows in adapted.values():
            for row in rows:
                if row['rho'] is not None:row['rho'] += delta
        results[str(seed)] = stats.compare_budgets(base, adapted, groups)
    summary = stats.summarize_seeds(results)['seed_spread']['available']['ranking_score']['1']
    assert summary['individual']['42']['adapted']==pytest.approx(.3)
    assert summary['all_three']['adapted']['sample_sd']==pytest.approx(.1)
    assert summary['confirmation_only']['adapted']['sample_sd']==pytest.approx(np.sqrt(.02))
    assert summary['confirmation_only']['adapted']['total']==2
    with pytest.raises(ValueError,match='all three'):stats.summarize_seeds({'42':results['42']})
    changed = copy.deepcopy(results)
    changed['43']['query_inventory_sha256']='changed'
    with pytest.raises(ValueError,match='cohorts differ'):stats.summarize_seeds(changed)


def test_native_query_worker_schema_is_accepted_without_rounding_float32_eligibility():
    from experiments.paired_generalization.adaptation_query import pathway_records
    frame = pd.DataFrame(dict(group_id=['p__g1']*3+['p__g2']*3+['p__g3']*2,
        pchembl_value=np.asarray([6,6.25,6.5-1e-8,5,6,7,5,8],dtype=np.float32),
        binary_label=[0,1,1,0,0,1,0,1],ranking_score=[1,1,1,3,2,1,1,2],
        activity_logits=[1,2,3,1,2,3,1,2],cosine_similarity=[0.]*8))
    groups = pd.DataFrame(dict(group_id=['p__g1','p__g2','p__g3'],unit=['u1','u1','u2'],
        rows=[3,3,2],ranking_eligible=[True,True,False]))
    native, rows = pathway_records(frame, groups, .5)
    result = stats.compare_budgets(rows, {1:rows}, {1:groups})
    for scorer in stats.SCORERS:
        point = result['available'][scorer]['1']['adapted']
        assert point['macro_spearman']==native[scorer]['macro_spearman']
        assert point['eligible_groups']==native[scorer]['spearman_num_groups']
        assert point['defined_groups']==native[scorer]['spearman_num_defined_groups']
