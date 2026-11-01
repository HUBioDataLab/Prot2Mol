import numpy as np
import pandas as pd

from experiments.paired_generalization.lambdamart import query_layout, rank_parameters


def test_noise_levels_preserve_full_large_queries_and_are_offset_invariant():
    large = 49020
    frame = pd.DataFrame({'group_id': ['large'] * large + ['small'] * 3,
                          'pchembl_value': np.r_[np.linspace(0, 7, large), [4., 4.4, 5.6]],
                          'identity': np.arange(large + 3)})
    ordered, sizes, levels = query_layout(frame.sample(frac=1, random_state=42), .75)
    assert sizes.tolist() == [large, 3]
    assert set(ordered.identity) == set(frame.identity)
    np.testing.assert_array_equal(levels, np.floor((ordered.pchembl_value - ordered.groupby('group_id').pchembl_value.transform('min')) / .75))
    shifted = ordered.copy()
    shifted.loc[shifted.group_id == 'small', 'pchembl_value'] += 10.
    np.testing.assert_array_equal(query_layout(shifted, .75)[2], levels)
    params = rank_parameters(int(sizes.max()))
    assert params['lambdarank_num_pair_per_sample'] == large
    assert params['ndcg_exp_gain'] is False
    assert params['lambdarank_normalization'] is True
    assert params['lambdarank_pair_method'] == 'topk'
