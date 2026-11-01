import json

import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization.similarity_composition import (
    descriptors, composition_weights, score_similarity, compare_composition, bootstrap_scores)


def rows_fixture():
    rows = []
    for assay,labels,scaffolds in [('mixed',[5.,6.,7.,5.,6.,7.],['S']*3+['U']*3),
                                  ('unseen',[5.,6.,7.],['U']*3),
                                  ('shared_small',[5.,7.],['S']*2)]:
        for i,(label,scaffold) in enumerate(zip(labels,scaffolds)):
            rows.append(dict(assay_id=assay,target_chembl_id='query_protein',compound_id=f'{assay}{i}',
                group_id='query_protein__'+assay,pchembl_value=label,binary_label=int(label>=6.),scaffold=scaffold))
    frame = pd.DataFrame(rows)
    mapping = pd.Series(['S',*frame.scaffold],index=['train_different_protein',*frame.compound_id])
    units = pd.Series([0,1,2],index=['mixed','unseen','shared_small'])
    return frame.drop(columns='scaffold'),mapping,units


def test_binary_relation_is_global_and_keeps_exact_identity_separate():
    frame,mapping,units = rows_fixture()
    rows,groups,summary = descriptors(frame,['train_different_protein'],mapping,units)
    assert set(rows.max_similarity) == {0,1}
    assert not rows.molecule_seen.any()
    assert summary['similarity'] == {'0':{'rows':6,'unique_molecules':6},'1':{'rows':5,'unique_molecules':5}}
    assert summary['whole_groups']['mixed'] == dict(groups=1,eligible_groups=1)
    assert summary['whole_groups']['all_shared'] == dict(groups=1,eligible_groups=0)
    assert groups.set_index('group_id').loc['query_protein__mixed','positive_bin'] == '(0.5,0.75]'
    with pytest.raises(ValueError,match='Missing scaffold'):
        descriptors(frame,['absent'],mapping,units)


def test_within_similarity_recomputes_eligibility_and_keeps_all_bootstrap_units():
    frame,mapping,units = rows_fixture()
    rows,groups,_ = descriptors(frame,['train_different_protein'],mapping,units)
    prediction = frame.assign(ranking_score=[3.,2.,1.,1.,2.,3.,1.,2.,3.,1.,2.])
    summary,table = score_similarity(rows,groups,prediction)
    assert summary['full']['eligible_groups'] == 2
    assert summary['whole_groups']['mixed']['eligible_groups'] == 1
    assert summary['within_stratum']['1']['eligible_groups'] == 1
    assert summary['within_stratum']['1']['original_eligible_macro_spearman'] == pytest.approx(-1.)
    assert summary['within_stratum']['0']['eligible_groups'] == 2
    assert summary['within_stratum']['0']['original_eligible_macro_spearman'] == pytest.approx(1.)
    assert summary['within_stratum']['1']['bootstrap']['units'] == 3
    assert len(table) == 3
    # A changed order or label cannot be silently joined back into a partition.
    with pytest.raises(AssertionError):
        score_similarity(rows,groups,prediction.iloc[::-1].reset_index(drop=True))
    changed = prediction.copy()
    changed.loc[0,'pchembl_value'] += 1e-9
    with pytest.raises(AssertionError):
        score_similarity(rows,groups,changed)


def population(cells,scores):
    return pd.DataFrame(dict(group_id=[f'g{i}' for i in range(len(cells))],unit_id=np.arange(len(cells)),
        cell=cells,spearman=scores,eligible=True,size=3,positive_rate=[0. if c=='low' else 1. for c in cells],
        label_mean=[5. if c=='low' else 7. for c in cells],label_span=2.))


def test_fixed_cell_weights_match_target_mass_and_have_analytic_standardized_score():
    a = population(['low','low','high'],[1.,1.,0.])
    b = population(['low','high','high','high'],[0.,0.,0.,0.])
    cells = composition_weights(a,b)
    w = cells.set_index('cell').A_group_weight
    assert w['low'] == .125 and w['high'] == .75
    result = compare_composition(a,b,cells)
    assert result['A']['raw']['score'] == pytest.approx(2/3)
    assert result['A']['standardized']['score'] == pytest.approx(.25)
    assert result['A']['standardization_change']['score'] == pytest.approx(.25-2/3)
    assert result['A']['standardized']['effective_groups'] == pytest.approx(1/(2*.125**2+.75**2))
    for descriptor in ('positive_rate','label_mean'):
        assert result['balance'][descriptor]['A_standardized'] == result['balance'][descriptor]['B_common']
    # The design itself cannot depend on prediction scores.
    pd.testing.assert_frame_equal(cells,composition_weights(a.assign(spearman=-a.spearman),b.assign(spearman=1.)))


def test_unsupported_target_cells_are_reported_and_never_extrapolated():
    a = population(['low','low'],[1.,0.])
    b = population(['low','missing'],[.2,-1.])
    cells = composition_weights(a,b)
    result = compare_composition(a,b,cells)
    assert result['target_supported_mass'] == .5
    assert result['B']['raw']['score'] == -.4
    assert result['B']['common']['score'] == .2
    assert result['standardized_common_A_minus_B'] == pytest.approx(.3)
    assert cells.set_index('cell').loc['missing','A_group_weight'] == 0


def test_undefined_spearman_is_excluded_with_weighted_coverage_not_replaced_by_zero():
    a = population(['low','high'],[1.,np.nan])
    b = population(['low','high'],[0.,0.])
    result = compare_composition(a,b,composition_weights(a,b))
    assert result['A']['standardized']['score'] == 1.
    assert result['A']['standardized']['defined_weight_fraction'] == .5
    all_missing = a.assign(spearman=np.nan)
    empty = compare_composition(all_missing,b,composition_weights(all_missing,b))
    assert empty['A']['standardized']['score'] is None
    assert empty['A']['standardized']['defined_weight_fraction'] == 0.
    assert empty['A']['standardization_change']['score'] is None
    json.dumps(empty,allow_nan=False)


def test_whole_units_include_units_with_no_eligible_stratum_and_share_draws():
    table = population(['low','low','high'],[1.,1.,-1.])
    table['unit_id'] = [0,0,1]
    result = bootstrap_scores(table,{'common':[1,1,1],'standardized':[1,1,1]},[0,1,2],repetitions=2000)
    assert result['common']['units'] == 3
    draw = np.random.default_rng(42).integers(3,size=(2000,3))
    assert result['common']['defined_repetitions'] == int(np.any(draw != 2,axis=1).sum())
    assert result['common']['score'] == pytest.approx(1/3)
    assert result['standardization_change']['score'] == 0.
    assert result['standardization_change']['interval'] == [0.,0.]
