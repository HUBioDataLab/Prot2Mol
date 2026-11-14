import numpy as np
import pandas as pd

from experiments.frozen_embedding_diagnostics.study import balanced_groups,select_candidates


def fixture():
    return pd.DataFrame([dict(x_source=x,g_source=f'{x}_{g}',y_source=f'y{i}',
        category_source=f'c{i%4}',source_row=g*100+i,s=float(i))
        for x in ('x1','x2','x3') for g in range(4) for i in range(20)])


def test_group_selection_covers_queries_and_is_label_independent():
    f=fixture()
    a=balanced_groups(f,['x1','x2','x3'],count=8)
    f['s']=-100*f.s
    b=balanced_groups(f,['x1','x2','x3'],count=8)
    assert a==b and len(set(a))==8
    counts=pd.Series([x for x,g in a]).value_counts()
    assert set(counts.index)=={'x1','x2','x3'} and counts.max()-counts.min()<=1


def test_candidate_selection_never_uses_scores_and_preserves_identity_uniqueness():
    f=fixture().iloc[:20].copy()
    f=pd.concat([f,f.iloc[:5]],ignore_index=True)
    a=select_candidates(f,np.random.default_rng(42))
    f['s']=np.random.default_rng(93).normal(size=len(f))
    b=select_candidates(f,np.random.default_rng(42))
    assert a.index.tolist()==b.index.tolist()
    assert len(a)==10 and a.y_source.nunique()==10
    assert a.category_source.nunique()>=2
