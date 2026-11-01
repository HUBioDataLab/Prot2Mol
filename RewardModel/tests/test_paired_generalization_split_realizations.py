import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization.split_realization_audit import assignment_from_units, compare_assignments


def test_assignment_keeps_giant_and_target_rescue_units_in_train_across_seeds():
    rows = []
    for component, count, target in [(0,40,'p0'),(1,3,'cold'), *[(i, i%4+1,'p0') for i in range(2,20)]]:
        for row in range(count):
            rows.append(dict(assay_id=f'a{component}',target_chembl_id=target,binary_label=row%2,component=component))
    frame = pd.DataFrame(rows)
    units = frame[['assay_id','component']].drop_duplicates()
    first, original = assignment_from_units(frame, units, seed=42, restarts=8)
    assert set(first.loc[first.split=='train','component']) == {0,1}
    for seed in (43,44):
        current, roles = assignment_from_units(frame, units, seed=seed,restarts=8)
        np.testing.assert_array_equal(roles=='train',original=='train')
        assert compare_assignments(first,current)['training_units_identical']
        assert len(roles)==len(frame)
        assert set(roles)=={'train','val','test'}
        varied, varied_roles = assignment_from_units(frame, units, seed=seed, restarts=8, randomize_equal_size_order=True)
        np.testing.assert_array_equal(varied_roles=='train', original=='train')
        assert compare_assignments(first,varied)['training_units_identical']
        repeat, _ = assignment_from_units(frame, units.sample(frac=1,random_state=10), seed=seed, restarts=8, randomize_equal_size_order=True)
        pd.testing.assert_frame_equal(varied,repeat)
    with pytest.raises(ValueError,match='exactly one'):
        assignment_from_units(frame,pd.concat([units,units.iloc[:1]]),seed=42)
