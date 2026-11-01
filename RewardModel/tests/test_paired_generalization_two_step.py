import numpy as np
from sklearn.linear_model import Ridge

from experiments.paired_generalization.independent_ridge import ridge_path
from experiments.paired_generalization.two_step_ridge import second_stage_path


def test_coefficient_regression_equals_explicit_virtual_label_two_step_fit():
    rng = np.random.default_rng(53)
    proteins, molecules = rng.normal(size=(8, 3)), rng.normal(size=(12, 5))
    new_proteins, new_molecules = rng.normal(size=(2, 3)), rng.normal(size=(4, 5))
    labels = rng.normal(size=(8, 12))
    first_strength, second_strength = .1, .03
    heads, virtual = [], []
    for i in range(len(proteins)):
        # Different observed subsets: omitted affinities never become zeros.
        observed = np.arange(i % 4, len(molecules))
        coefficient, intercept = ridge_path(molecules[observed], labels[i, observed], np.ones(len(observed)), [first_strength])
        heads.append(np.r_[coefficient[0], intercept[0]])
        virtual.append(new_molecules @ coefficient[0] + intercept[0])
    coefficient, intercept = second_stage_path(proteins, np.array(heads), [second_strength])[0]
    predicted_heads = new_proteins @ coefficient + intercept
    actual = predicted_heads[:, :-1] @ new_molecules.T + predicted_heads[:, -1, None]
    explicit = Ridge(alpha=second_strength * len(proteins)).fit(proteins, np.array(virtual)).predict(new_proteins)
    np.testing.assert_allclose(actual, explicit, atol=1e-10)
