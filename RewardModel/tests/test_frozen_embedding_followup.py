import numpy as np
import pandas as pd

from experiments.frozen_embedding_diagnostics.followup import (
    derangement,choose_supported_queries,matched_group_plan,dense_predictions,
    independent_metric,comparison,
)
from experiments.frozen_embedding_diagnostics.analysis import Evaluation,unit_vectors


def fixture(partition='train'):
    rows=[]
    for q in range(5):
        for group in range(3):
            for i in range(10):
                rows.append(dict(partition=partition,x=f'X{q}',x_source=f'X{q}',
                    g=f'{partition}G{q}_{group}',g_source=f'{partition}G{q}_{group}',
                    y=f'Y{q}_{group}_{i}',y_source=f'Y{q}_{group}_{i}',unit=f'U{q}',
                    s=float(i),centered=float(i)-4.5))
    return pd.DataFrame(rows)


def test_swap_preserves_identity_pool_and_has_no_fixed_points():
    mapping=derangement(['X0','X1','X2','X3','X0'])
    assert set(mapping)==set(mapping.values())
    assert all(q!=v for q,v in mapping.items())
    assert mapping==derangement(list(reversed(list(mapping))))


def test_supported_query_membership_ignores_labels():
    tr=fixture();va=fixture('val')
    expected,_=choose_supported_queries(tr,va,n=3,training_groups=3,validation_groups=2)
    tr['s']=0;va['s']=np.arange(len(va))*1000
    actual,_=choose_supported_queries(tr,va,n=3,training_groups=3,validation_groups=2)
    assert expected==actual


def test_matched_global_has_equal_usable_groups_and_excludes_own_query():
    tr=fixture();tr.loc[tr.g=='trainG0_0','s']=0
    plan=matched_group_plan(tr)
    owners=tr.drop_duplicates('g').set_index('g').x.to_dict()
    assert len(plan['X0']['own_groups'])==2
    for q,p in plan.items():
        assert len(p['own_groups'])==len(p['global_groups'])
        assert len(set(p['global_queries']))==len(p['global_groups'])
        assert all(owners[g]==q for g in p['own_groups'])
        assert all(owners[g]!=q for g in p['global_groups'])


def test_direction_predictions_do_not_consult_validation_labels():
    tr=fixture();va=fixture('val');ids=sorted(tr.y.unique())
    lookup={v:i for i,v in enumerate(ids)}
    vectors=unit_vectors(np.random.default_rng(9).normal(size=(len(ids),7)))
    plan=matched_group_plan(tr)
    first=dense_predictions(tr,va,vectors,lookup,plan)
    va['s']=123456
    second=dense_predictions(tr,va,vectors,lookup,plan)
    for k in first:np.testing.assert_array_equal(first[k],second[k])
    # Additive offsets within training groups cannot change a direction.
    shifted=tr.copy();shifted['s']+=shifted.groupby('g').ngroup()*100
    third=dense_predictions(shifted,va,vectors,lookup,plan)
    for k in first:np.testing.assert_allclose(first[k],third[k])


def test_paired_comparison_and_independent_metric_keep_undefined_groups():
    f=fixture();evaluation=Evaluation(f)
    p=f.s.to_numpy().copy();p[:10]=0
    metric=evaluation.summarize(p);independent_metric(f,p,metric)
    result=comparison(evaluation,p,-f.s.to_numpy())
    assert result['common_groups']==14
    assert np.isclose(result['difference'],2)
    assert np.isclose(result['equal_query_difference'],2)
    assert result['query_cluster_interval95'][0]>1.99
