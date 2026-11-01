import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization.leak_design import construct, pair_independent_assays, reserve_b_donors


def assay(name,target,scaffold,component,values=(5.,6.,7.),replicas=(1,2,1)):
    return pd.DataFrame(dict(assay_id=[name]*3,target_chembl_id=[target]*3,
        compound_id=[f'{name}_{i}' for i in range(3)],pchembl_value=values,
        binary_label=np.asarray(values)>=6,measurement_count=replicas,
        scaffold=[scaffold]*3,component=[component]*3))


def inputs(regime):
    train = pd.concat([assay('out','p','novel',100),assay('common','q','other',200)],ignore_index=True)
    test = assay('held','p','test',500)
    if regime == 'A':
        train = pd.concat([train,assay('donor','p','test',500)],ignore_index=True)
    else:
        test = pd.concat([test,assay('reserved','p','test',500)],ignore_index=True)
    frames = dict(train=train,val=assay('valid','p','validation',300),test=test)
    for role,frame in frames.items():
        frame['source_partition'],frame['source_row_index'] = role,np.arange(len(frame))
    return frames


@pytest.mark.parametrize('regime',['A','B'])
def test_exchange_preserves_composition_and_only_introduces_relatives_in_training(regime):
    frames = inputs(regime)
    summary,tables = construct(frames,regime)
    assert summary['matched_assay_exchanges'] == 1
    assert summary['usable_leak_contrast']
    assert summary['exposed_test_rows'] == 3
    assert summary['common_binary_labels_unchanged']
    assert summary['training']['leak_out'] == summary['training']['leak_in']
    assert set(tables['leak_out'].assay_id) == {'out','common'}
    assert not set(tables['leak_in'].assay_id)&set(tables['test'].assay_id)
    assert not tables['leak_out'].scaffold.isin(tables['test'].scaffold).any()
    if regime == 'B':
        assert summary['original_full_test_relative_rows_outside_test'] == 0
        assert set(tables['test'].assay_id)|set(tables['donors'].assay_id) == set(frames['test'].assay_id)


def test_b_test_reservation_is_label_blind_and_complete_assays():
    frame = inputs('B')['test']
    donors,test = reserve_b_donors(frame)
    changed = frame.assign(pchembl_value=np.arange(len(frame))*100,binary_label=0,measurement_count=99)
    donors2,test2 = reserve_b_donors(changed)
    assert donors.assay_id.tolist() == donors2.assay_id.tolist()
    assert test.assay_id.tolist() == test2.assay_id.tolist()
    assert len(donors) == len(test) == 3


def test_mismatched_replicates_or_protein_are_not_accepted_as_matched():
    for field,value in [('measurement_count',9),('target_chembl_id','different')]:
        frames = inputs('A')
        frames['train'].loc[frames['train'].assay_id=='donor',field] = value
        summary,tables = construct(frames,'A')
        assert summary['matched_assay_exchanges'] == 0
        assert not summary['usable_leak_contrast']
        pd.testing.assert_frame_equal(tables['leak_in'],tables['leak_out'])


def test_shared_pair_labels_cannot_change_when_an_assay_is_removed():
    frames = inputs('A')
    train = frames['train']
    train.loc[train.assay_id=='common','target_chembl_id'] = 'p'
    train.loc[(train.assay_id=='common') & (train.source_row_index==3),'compound_id'] = 'out_0'
    train.loc[(train.assay_id=='common') & (train.source_row_index==3),'pchembl_value'] = 9.
    summary,tables = construct(frames,'A')
    assert summary['matched_assay_exchanges'] == 0
    assert summary['common_binary_labels_unchanged']
    pd.testing.assert_frame_equal(tables['leak_in'],tables['leak_out'])


def test_donor_independence_uses_pair_identities_and_keeps_whole_assays():
    frames = inputs('A')
    frame = frames['train'].copy()
    frame.loc[frame.assay_id=='common','target_chembl_id'] = 'p'
    frame.loc[frame.assay_id=='common','compound_id'] = ['out_0','common_1','common_2']
    selected = pair_independent_assays(frame)
    assert not selected.duplicated(['target_chembl_id','compound_id']).any()
    assert selected.groupby('assay_id').size().eq(3).all()
    altered = pair_independent_assays(frame.assign(pchembl_value=0,binary_label=0))
    assert selected.assay_id.tolist() == altered.assay_id.tolist()


def test_b_partial_component_proposal_is_explicit_and_preserves_matching():
    frames = inputs('B')
    frames['train']['component'] = 100
    strict,_ = construct(frames,'B')
    assert strict['matched_assay_exchanges'] == 0
    proposal,tables = construct(frames,'B',allow_b_assay_exchanges=True)
    assert proposal['matched_assay_exchanges'] == 1
    assert proposal['requires_user_decision']
    assert proposal['partial_original_training_components'] == 1
    assert proposal['training']['leak_out'] == proposal['training']['leak_in']
    assert not set(tables['test'].assay_id)&set(tables['leak_in'].assay_id)
