import copy

import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization.realization_protocol import align_rows, make_views, example_frame


def fixture():
    source = pd.DataFrame([
        dict(assay_id=f'a{unit}', target_chembl_id='p', compound_id=f'm{unit}_{i}',
             pchembl_value=4.+i, binary_label=int(i > 1))
        for unit in range(6) for i in range(3)])
    units = pd.DataFrame(dict(assay_id=[f'a{i}' for i in range(6)], component=np.arange(6)))
    allocations = dict(original_seed42=([0,1],[2,3]), equal_size_seed43=([0,2],[1,3]), equal_size_seed44=([0,3],[1,2]))
    indices = {name: {role: np.array([3*u+i for u in group for i in range(3)])
                     for role, group in zip(('val','test'), groups)} for name, groups in allocations.items()}
    raw = {role: source.iloc[values].sample(frac=1, random_state=3).reset_index(drop=True)
           for role, values in indices['original_seed42'].items()}
    # Partition median binary labels may differ from raw source labels.
    raw['val'].loc[0, 'binary_label'] = 1 - raw['val'].loc[0, 'binary_label']
    tokens = {role: example_frame(frame).iloc[::-1].reset_index(drop=True) for role, frame in raw.items()}
    return source, raw, tokens, indices, units


def test_views_bind_by_identity_preserve_whole_units_and_do_not_assume_row_order():
    source, raw, tokens, indices, units = fixture()
    rows, views, counts = make_views(source, raw, tokens, indices, units)
    assert len(rows) == 12
    for realization in indices:
        for split in ('val','test'):
            selected = rows.iloc[views[f'{realization}_{split}']]
            assert set(selected.source_index) == set(indices[realization][split])
            assert counts[realization][split] == dict(rows=6, units=2, groups=2)
            assert set(selected.groupby('group_id').size()) == {3}
    assert rows.original_row_index.tolist() == list(range(6))*2
    for split in ('val','test'):
        saved = rows[rows.original_split == split]
        assert saved.compound_id.tolist() == tokens[split].compound_id.tolist()
        np.testing.assert_array_equal(saved.binary_label, tokens[split].binary_label)


def test_rejects_corrupt_cache_labels_missing_rows_and_duplicate_identities():
    source, raw, tokens, indices, units = fixture()
    changed = copy.deepcopy(tokens)
    changed['test'].loc[0, 'pchembl_value'] += .25
    with pytest.raises(ValueError, match='labels disagree'):
        make_views(source, raw, changed, indices, units)
    changed = copy.deepcopy(tokens)
    changed['val'] = changed['val'].iloc[:-1]
    with pytest.raises(ValueError, match='complete permutation'):
        make_views(source, raw, changed, indices, units)
    with pytest.raises(ValueError, match='duplicate'):
        align_rows(raw['val'], pd.concat([raw['val'],raw['val'].iloc[:1]]))


def test_rejects_split_units_role_overlap_and_changed_training_support():
    source, raw, tokens, indices, units = fixture()
    changed = copy.deepcopy(indices)
    changed['equal_size_seed43']['val'][0] = 3
    changed['equal_size_seed43']['test'][0] = 0
    with pytest.raises(ValueError, match='splits an indivisible'):
        make_views(source, raw, tokens, changed, units)
    changed = copy.deepcopy(indices)
    changed['equal_size_seed43']['val'][0] = 12  # A training observation.
    with pytest.raises(ValueError, match='changes held-out support'):
        make_views(source, raw, tokens, changed, units)
    changed = copy.deepcopy(indices)
    changed['equal_size_seed43']['test'] = changed['equal_size_seed43']['val']
    with pytest.raises(ValueError, match='overlaps roles'):
        make_views(source, raw, tokens, changed, units)


def test_rejects_relabelled_identical_allocations():
    source, raw, tokens, indices, units = fixture()
    indices['equal_size_seed44'] = indices['equal_size_seed43']
    with pytest.raises(ValueError, match='not distinct'):
        make_views(source, raw, tokens, indices, units)
