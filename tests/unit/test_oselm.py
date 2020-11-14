"""Unit tests for pyoselm.oselm module"""

import numpy as np
import pytest
from sklearn.datasets import load_digits, make_regression
from sklearn.preprocessing import LabelBinarizer

from pyoselm.oselm import OSELMClassifier, OSELMRegressor, multiple_safe_sparse_dot


@pytest.mark.parametrize("n_hidden", [10, 100])
@pytest.mark.parametrize("activation_func", ["tanh", "sine", "gaussian",
                                             "sigmoid", "softlim"])
@pytest.mark.parametrize("use_woodbury", [False, True])
def test_oselm_regressor(n_hidden, activation_func, use_woodbury):
    # get data
    X, y = make_regression(n_samples=100, n_targets=1, n_features=10, random_state=123)

    # build model
    model = OSELMRegressor(n_hidden=n_hidden,
                           activation_func=activation_func,
                           use_woodbury=use_woodbury,
                           random_state=123)

    # fit model
    model.fit(X, y)

    # predict
    y_pred = model.predict(X)
    min_y, max_y = min(y), max(y)
    assert all([min_y*2 < yy < max_y*2 for yy in y_pred]), \
        "Predicted values out of expected range"

    # score
    score = model.score(X, y)
    assert score > 0.0, "Score of model is lower than expected"

    # partial fit
    model.partial_fit(X, y)

    # predict
    y_pred = model.predict(X)
    min_y, max_y = min(y), max(y)
    assert all([min_y * 2 < yy < max_y * 2 for yy in y_pred]), \
        "Predicted values out of expected range"

    # score
    score = model.score(X, y)
    assert score > 0.0, "Score of model is lower than expected"


@pytest.mark.parametrize("n_hidden", [10, 100])
@pytest.mark.parametrize("activation_func", ["tanh", "sine", "gaussian",
                                             "sigmoid", "softlim"])
@pytest.mark.parametrize("binarizer", [LabelBinarizer(0, 1),
                                       LabelBinarizer(-1, 1)])
def test_oselm_classifier(n_hidden, activation_func, binarizer):
    # get data
    X, y = load_digits(n_class=10, return_X_y=True)

    # build model
    model = OSELMClassifier(n_hidden=n_hidden,
                            activation_func=activation_func,
                            binarizer=binarizer,
                            random_state=123)

    # fit model
    model.fit(X, y)

    # predict
    y_pred = model.predict(X)
    set_y = set(y)
    assert all([yy in set_y for yy in y_pred]), \
        "Predicted values out of expected range"

    # predict proba
    y_proba = model.predict_proba(X)
    assert all([((yy >= 0) & (yy <= 1)).all() for yy in y_proba]), \
        "Predicted values out of expected range"

    # score
    score = model.score(X, y)
    assert score > 0.0, "Score of model is lower than expected"

    # partial fit
    model.partial_fit(X, y)

    # predict
    y_pred = model.predict(X)
    set_y = set(y)
    assert all([yy in set_y for yy in y_pred]), \
        "Predicted values out of expected range"

    # score
    score = model.score(X, y)
    assert score > 0.0, "Score of model is lower than expected"


def test_multiple_safe_sparse_dot():
    n = 10
    a, b = 2, 3
    matrices = [np.ones((n, n)), np.ones((n, n))*a, np.ones((n, n))*b]

    with pytest.raises(ValueError):
        # just 1 matrix, no sparse dot allowed
        multiple_safe_sparse_dot(matrices[0])

    res = multiple_safe_sparse_dot(*matrices)
    print(res)
    assert np.array_equal(res, np.ones((n, n))*n*n*a*b)


def test_oselm_bad_path():
    # predict() without fit
    X, y = make_regression(n_samples=100, n_targets=1, n_features=10)
    model = OSELMRegressor()

    with pytest.raises(ValueError):
        model.predict(X)

    # predict() without fit
    X, y = load_digits(n_class=10, return_X_y=True)
    model = OSELMClassifier()

    with pytest.raises(ValueError):
        model.predict(X)

    with pytest.raises(ValueError):
        model.predict_proba(X)

    # fit OSELM model for first time with not enough rows
    model = OSELMClassifier(use_woodbury=False)
    with pytest.raises(ValueError):
        model.fit(X[:2, :], y[:2])

    model = OSELMClassifier(use_woodbury=True)
    with pytest.raises(ValueError):
        model.fit(X[:2, :], y[:2])


@pytest.mark.skip("Very expensive test")
def test_oselm_fit_woodbury_large_input():
    n = 20e3
    X, y = make_regression(n_samples=int(n), n_targets=1, n_features=10, random_state=123)

    model = OSELMRegressor(use_woodbury=True)

    model.fit(X, y)
    # Second fit throws the warning about large input
    model.fit(X, y)

    score = model.score(X, y)

    assert score > 1.0


# TODO: test with sparse data?

# TODO: test with singular matrix?
