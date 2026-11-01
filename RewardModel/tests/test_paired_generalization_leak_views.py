import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization.leak_views import align_subset


def fixture():
    return pd.DataFrame(dict(target_chembl_id=['p']*6,assay_id=['a']*3+['b']*3,
        compound_id=['x','y','z','x','m','n'],pchembl_value=[5.,6.,7.,9.,8.,7.],
        binary_label=[1,1,1,1,1,1],unit=['a']*3+['b']*3))


def test_subset_maps_cache_order_and_recomputes_labels_without_omitted_assay():
    full = fixture().iloc[[5,2,1,4,0,3]].reset_index(drop=True)
    selected = fixture().iloc[:3].copy()
    selected['binary_label'] = [0,1,1]
    indices,labels = align_subset(full,selected.iloc[::-1])
    assert indices.tolist() == [1,2,4]
    assert labels.tolist() == [1,1,0]


@pytest.mark.parametrize('failure',['missing','partial','labels','continuous'])
def test_rejects_invalid_matched_views(failure):
    full=fixture();selected=full.iloc[:3].copy();selected['binary_label']=[0,1,1]
    if failure=='missing':selected.loc[0,'compound_id']='absent'
    if failure=='partial':selected=selected.iloc[:2]
    if failure=='labels':selected.loc[0,'binary_label']=1
    if failure=='continuous':selected.loc[0,'pchembl_value']=4.
    with pytest.raises(ValueError):align_subset(full,selected)
