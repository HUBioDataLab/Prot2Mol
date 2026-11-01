import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization.supervision_audit import composition


def fixture():
    labels = {'a':[5.,6.,7.],'b':[7.,8.],'c':[5.9,6.,6.1],
              'd':[6.1],'e':[7.,8.,9.],'f':[2.,3.,4.]}
    return pd.DataFrame([dict(assay_id=assay,target_chembl_id='p2' if assay == 'f' else 'p1',
        compound_id=f'{assay}{i}',pchembl_value=value,binary_label=int(value >= 6.),measurement_count=1)
        for assay,values in labels.items() for i,value in enumerate(values)])


def test_complete_partition_reasons_homogeneous_labels_and_target_composition():
    frame = fixture()
    result,groups,targets,units = composition(frame,.75,frame.assay_id)
    assert result['groups'] == 6
    assert result['eligibility']['eligible_groups'] == 3
    assert result['eligibility']['too_small_groups'] == 2
    assert result['eligibility']['enough_items_but_insufficient_span_groups'] == 1
    assert result['all_groups']['all_positive_groups'] == 3
    assert result['all_groups']['all_negative_groups'] == 1
    assert result['eligible']['same_side_fraction'] == pytest.approx(2/3)
    assert result['targets']['eligible_vs_ineligible_group_target_total_variation'] == pytest.approx(1/3)
    assert result['units']['row_count']['count'] == 6
    assert len(groups) == 6 and len(targets) == 2 and int(units.rows.sum()) == len(frame)
    assert result['threshold']['observation_fraction_within_rms'] == pytest.approx(5/len(frame))
    assert result['threshold']['unique_pair_median_fraction_within_rms'] == pytest.approx(5/len(frame))


def test_binary_label_policy_and_missing_units_are_not_silently_accepted():
    frame = fixture()
    frame.loc[0,'binary_label'] = 1
    with pytest.raises(ValueError,match='within the current partition'):
        composition(frame,.75,frame.assay_id)
    frame = fixture()
    units = frame.assay_id.copy()
    units.iloc[0] = np.nan
    with pytest.raises(ValueError,match='complete row-level units'):
        composition(frame,.75,units)
