import numpy as np
import pandas as pd

from experiments.paired_generalization.nearest_lookup import fit_lookup, predict_lookup


def test_binary_relation_averages_tied_molecules_not_measurement_frequency():
    training = pd.DataFrame({"target_chembl_id": ["x"] * 4 + ["other"], "compound_id": ["a", "a", "b", "c", "d"],
        "pchembl_value": [1., 3., 8., 11., 100.], "binary_label": [0, 0, 1, 1, 1]})
    scaffolds = pd.Series({"a": "one", "b": "one", "c": "two", "d": "one", "q": "one", "r": "new", "s": "new"})
    query = pd.DataFrame({"target_chembl_id": ["x", "x", "x"], "compound_id": ["q", "r", "s"]})
    result = predict_lookup(query, scaffolds, fit_lookup(training, scaffolds))
    # Mean of two molecules (median(a)=2, b=8), not three raw training rows.
    assert result.ranking_score.tolist() == [5., 7., 7.]
    assert result.nearest_similarity.tolist() == [1, 0, 0]
    assert result.number_of_tied_nearest_molecules.tolist() == [2, 3, 3]
    assert result.query_target_seen.all()
    assert result.classification_score.iloc[0] == 0.


def test_query_labels_cannot_change_fitted_nearest_predictions():
    train = pd.DataFrame({"target_chembl_id": ["x", "x"], "compound_id": ["a", "b"], "pchembl_value": [2., 8.], "binary_label": [0, 1]})
    scaffolds = pd.Series({"a": "one", "b": "two", "q": "one"})
    query = pd.DataFrame({"target_chembl_id": ["x"], "compound_id": ["q"], "pchembl_value": [0.], "binary_label": [0]})
    model = fit_lookup(train, scaffolds)
    first = predict_lookup(query, scaffolds, model)
    query[["pchembl_value", "binary_label"]] = [100., 1]
    second = predict_lookup(query, scaffolds, model)
    np.testing.assert_array_equal(first.ranking_score, second.ranking_score)
    np.testing.assert_array_equal(first.classification_score, second.classification_score)
