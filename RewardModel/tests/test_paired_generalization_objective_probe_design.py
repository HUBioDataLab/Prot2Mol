import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization.objective_probe_design import cold_units


def fixture():
    def rows(assay,molecules):
        return [dict(assay_id=assay,target_chembl_id='p',compound_id=m,pchembl_value=5.+i) for i,m in enumerate(molecules)]
    train=pd.DataFrame(rows('train',['t']))
    val=pd.DataFrame(rows('cold',['a','b','c'])+rows('mixed',['d','e','f'])+rows('connected',['g','h','i']))
    scaffolds=pd.Series({'t':'seen','a':'s1','b':'s2','c':'s3','d':'seen','e':'s4','f':'s5','g':'s6','h':'s7','i':'s8'})
    components=pd.Series({'train':0,'cold':1,'mixed':2,'connected':2})
    return train,val,scaffolds,components


def test_cold_selection_keeps_whole_assays_or_components_not_just_unseen_rows():
    train,val,scaffolds,components=fixture()
    a,_,sa=cold_units(train,val,scaffolds,components,'A')
    b,_,sb=cold_units(train,val,scaffolds,components,'B')
    assert set(a.assay_id)=={'cold','connected'}
    assert set(b.assay_id)=={'cold'}
    assert sa['ranking_eligible_assays']==2 and sb['ranking_eligible_assays']==1
    assert not a.scaffold_seen_in_training.any() and not b.scaffold_seen_in_training.any()
    assert b.source_row_index.tolist()==[0,1,2]


def test_labels_change_feasibility_but_never_membership():
    train,val,scaffolds,components=fixture()
    first,_,_=cold_units(train,val,scaffolds,components,'A')
    second,_,summary=cold_units(train,val.assign(pchembl_value=0.),scaffolds,components,'A')
    pd.testing.assert_frame_equal(first,second)
    assert summary['ranking_eligible_assays']==0 and summary['strict_active_rows']==0


def test_missing_relations_and_partition_overlap_are_refused():
    train,val,scaffolds,components=fixture()
    with pytest.raises(ValueError,match='Missing molecule scaffold'):
        cold_units(train,val,scaffolds.drop('a'),components,'A')
    with pytest.raises(ValueError,match='units/assays overlap'):
        cold_units(train,val,scaffolds,components.replace({1:0}),'B')
    with pytest.raises(ValueError,match='Missing scaffold component'):
        cold_units(train,val,scaffolds,components.drop('cold'),'B')


def test_float32_threshold_boundary_is_only_a_reported_feasibility_rule():
    train,val,scaffolds,components=fixture()
    val.loc[:2,'pchembl_value']=[0.,.2,.499999999]
    _,_,result=cold_units(train,val,scaffolds,components,'B')
    assert result['ranking_eligible_assays']==1
    # Trainer promotes float32 labels to Python floats before subtracting;
    # subtracting in float32 would incorrectly round this span back to .5.
    val.loc[:2,'pchembl_value']=[.2,.4,.7]
    _,_,result=cold_units(train,val,scaffolds,components,'B')
    assert result['ranking_eligible_assays']==0
