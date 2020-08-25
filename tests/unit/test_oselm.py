"""Unit tests for pyoselm.oselm module"""

import pytest
from sklearn.datasets import load_digits, make_regression
from sklearn.preprocessing import LabelBinarizer

from pyoselm.oselm import (OSELMClassifier, OSELMRegressor,
                           OSELMClassifierSoftmax)


@pytest.mark.parametrize("n_hidden", [10, 100])
@pytest.mark.parametrize("activation_func", ["tanh", "sine", "gaussian",
                                             "sigmoid", "softlim"])
def test_oselm_regressor(n_hidden, activation_func):
    # get data
    X, y = make_regression(n_samples=100, n_targets=1, n_features=10)

    # build model
    model = OSELMRegressor(n_hidden=n_hidden, activation_func=activation_func)

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
                            binarizer=binarizer)

    # fit model
    model.fit(X, y)

    # predict
    y_pred = model.predict(X)
    set_y = set(y)
    assert all([yy in set_y for yy in y_pred]), \
        "Predicted values out of expected range"

    # score
    score = model.score(X, y)
    assert score > 0.0, "Score of model is lower than expected"


@pytest.mark.parametrize("n_hidden", [10, 100])
@pytest.mark.parametrize("activation_func", ["tanh", "sine", "gaussian",
                                             "sigmoid", "softlim"])
def test_oselm_classifier_softmax(n_hidden, activation_func):
    # get data
    X, y = load_digits(n_class=10, return_X_y=True)

    # build model
    model = OSELMClassifierSoftmax(n_hidden=n_hidden,
                                   activation_func=activation_func)

    # fit model
    model.fit(X, y)

    # predict
    y_pred = model.predict(X)
    set_y = set(y)
    assert all([yy in set_y for yy in y_pred]), \
        "Predicted values out of expected range"

    # score
    score = model.score(X, y)
    assert score > 0.0, "Score of model is lower than expected"

