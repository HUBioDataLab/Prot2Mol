import numpy as np
import pandas as pd
import json

from experiments.frozen_embedding_diagnostics.analysis import (
    Evaluation,directions,neighbour_readouts,permutation_indices,profile_edges,
    second_order,unit_vectors,weighted_neighbours,
)


def frame(partition='train'):
    return pd.DataFrame([dict(partition=partition,x=f'X{q:06d}',y=f'Y{q*10+i:06d}',
        g=f'G{q:06d}',unit=f'U{q:06d}',category=f'C{i%3}',s=float(i),centered=float(i)-4.5,full_group_mean=4.5)
        for q in range(3) for i in range(10)])


def test_metric_is_macro_and_constant_groups_are_undefined():
    f=frame('val');e=Evaluation(f)
    perfect=f.s.to_numpy().copy()
    assert e.summarize(perfect)['macro_spearman']>0.999999
    perfect[:10]=0
    out=e.summarize(perfect)
    assert out['defined_groups']==2 and out['coverage']==2/3
    assert out['macro_spearman']>0.999999
    difference=e.difference(f.s.to_numpy(),-f.s.to_numpy())
    assert difference['difference']>1.999999


def test_neighbours_use_no_evaluation_labels_and_exclude_own_query():
    train=frame();val=frame('val')
    vectors=unit_vectors(np.random.default_rng(1).normal(size=(30,8)))
    lookup={f'Y{i:06d}':i for i in range(30)}
    pred,_=neighbour_readouts(train,val,vectors,lookup)
    val['s']=12345;val['centered']=-333
    again,_=neighbour_readouts(train,val,vectors,lookup)
    for name in pred:np.testing.assert_array_equal(pred[name],again[name])
    changed=train.copy();changed.loc[changed.x=='X000000','centered']+=100
    altered,_=neighbour_readouts(changed,val,vectors,lookup)
    np.testing.assert_allclose(pred['nearest_other'][:10],altered['nearest_other'][:10],atol=1e-10)
    np.testing.assert_allclose(altered['nearest_same'][:10]-pred['nearest_same'][:10],100)


def test_direction_cancels_group_offsets_and_uses_within_group_pairs():
    f=frame();v=unit_vectors(np.random.default_rng(3).normal(size=(30,8)))
    lookup={f'Y{i:06d}':i for i in range(30)}
    a,support=directions(f,v,lookup)
    f.loc[f.x=='X000001','s']+=1000
    b,other=directions(f,v,lookup)
    assert support==other
    for q in a:np.testing.assert_allclose(a[q],b[q])


def test_category_swap_is_consistent_by_identity_and_preserves_categories():
    f=frame();ids=sorted(f.y.unique());mapping,summary=permutation_indices(f,ids)
    classes=f.set_index('y').reindex(ids).category.to_numpy()
    np.testing.assert_array_equal(classes,classes[mapping])
    assert summary['actually_changed']>0 and sorted(mapping)==list(range(30))


def test_no_shared_candidates_means_no_profile_edges():
    profile=profile_edges(frame(),'query')
    assert profile['edges']==[]
    assert second_order(profile,np.eye(3))['status']=='insufficient_overlap'


def test_shared_profiles_serialize_without_turning_missing_values_into_zero():
    f=frame()
    f['y']=[f'Y{i%10:06d}' for i in range(len(f))]
    profile=profile_edges(f,'query')
    assert len(profile['edges'])==3
    assert all(np.isclose(e[2],1) and e[3]==10 for e in profile['edges'])
    json.dumps(profile,allow_nan=False)


def test_conditional_null_retains_groups_and_detects_perfect_order():
    from experiments.frozen_embedding_diagnostics.report import conditional_label_null
    f=frame('val');e=Evaluation(f)
    result=conditional_label_null(e,{'perfect':f.s.to_numpy(),'constant':np.zeros(len(f))},draws=99)
    assert result['perfect']['p_one_sided']==.01
    assert result['constant']['p_one_sided']==1


def test_inverse_distance_readout_and_missing_reference():
    p,d=weighted_neighbours(np.array([[1.,.5]]),np.array([3.,9.]),k=1)
    assert p[0]==3 and d[0]==0
    p,d=weighted_neighbours(np.empty((2,0)),np.array([]))
    assert np.isnan(p).all() and np.isnan(d).all()
