import pandas as pd
from pandas.testing import assert_frame_equal

from experiments.paired_generalization.prepare_random import partition_local_labels


def test_heldout_measurements_cannot_change_training_labels():
    frame = pd.DataFrame({"target_chembl_id": ["x", "x"], "compound_id": ["m", "m"],
                          "pchembl_value": [5., 5.4], "binary_label": [0, 0]})
    heldout = frame.assign(pchembl_value=10., binary_label=1)
    first, _ = partition_local_labels({"train": frame, "val": heldout, "test": heldout})
    second, _ = partition_local_labels({"train": frame, "val": heldout.assign(pchembl_value=1.), "test": heldout.assign(pchembl_value=1.)})
    assert_frame_equal(first["train"], second["train"])
    assert first["train"].binary_label.tolist() == [0, 0]
    assert first["val"].binary_label.tolist() == [1, 1]
    assert second["val"].binary_label.tolist() == [0, 0]
