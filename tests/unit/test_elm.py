"""Unit tests for pyoselm.elm module"""

import pytest
from sklearn.datasets import load_digits, make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelBinarizer

from pyoselm.elm import (GenELMRegressor, GenELMClassifier,
                         ELMRegressor, ELMClassifier)
from pyoselm.layer import RBFRandomLayer


@pytest.mark.parametrize("hidden_layer", [None, RBFRandomLayer(random_state=123)])
@pytest.mark.parametrize("regressor", [None, LinearRegression()])
def test_gen_elm_regressor(hidden_layer, regressor):
    # get data
    X, y = make_regression(n_samples=100, n_targets=1, n_features=10, random_state=123)

    # build model
    model = GenELMRegressor(hidden_layer=hidden_layer, regressor=regressor)

    # fit model
    model.fit(X, y)

    # predict
    y_pred = model.predict(X)
    min_y, max_y = min(y), max(y)
    assert all([min_y*2 < yy < max_y*2 for yy in y_pred]), \
        "Predicted values out of expected range"

    # score
    score = model.score(X, y)
    assert -1.0 < score <= 1.0, "Score of model is not in expected range"


@pytest.mark.parametrize("hidden_layer", [None, RBFRandomLayer(random_state=123)])
@pytest.mark.parametrize("binarizer", [None,
                                       LabelBinarizer(neg_label=0, pos_label=1),
                                       LabelBinarizer(neg_label=-1, pos_label=1)])
@pytest.mark.parametrize("regressor", [None, LinearRegression()])
def test_gen_elm_classifier(hidden_layer, binarizer, regressor):
    # get data
    X, y = load_digits(n_class=10, return_X_y=True)

    # build model
    model = GenELMClassifier(hidden_layer=hidden_layer,
                             binarizer=binarizer,
                             regressor=regressor)

    # fit model
    model.fit(X, y)

    # predict
    y_pred = model.predict(X)
    set_y = set(y)
    assert all([yy in set_y for yy in y_pred]), \
        "Predicted values out of expected range"

    # score
    score = model.score(X, y)
    assert 0.0 < score <= 1.0, "Score of model is not in expected range"


@pytest.mark.parametrize("n_hidden", [10, 100])
@pytest.mark.parametrize("activation_func", ["tanh", "sine", "gaussian",
                                             "sigmoid", "softlim"])
@pytest.mark.parametrize("regressor", [None, LinearRegression()])
def test_elm_regressor(n_hidden, activation_func, regressor):
    # get data
    X, y = make_regression(n_samples=100, n_targets=1, n_features=10, random_state=123)

    # build model
    model = ELMRegressor(n_hidden=n_hidden,
                         activation_func=activation_func,
                         regressor=regressor,
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
    assert -1.0 < score <= 1.0, "Score of model is not in expected range"


@pytest.mark.parametrize("n_hidden", [10, 100])
@pytest.mark.parametrize("activation_func", ["tanh", "sine", "gaussian",
                                             "sigmoid", "softlim"])
@pytest.mark.parametrize("binarizer", [LabelBinarizer(0, 1),
                                       LabelBinarizer(-1, 1)])
@pytest.mark.parametrize("regressor", [None, LinearRegression()])
def test_elm_classifier(n_hidden, activation_func, binarizer, regressor):
    # get data
    X, y = load_digits(n_class=10, return_X_y=True)

    # build model
    model = ELMClassifier(n_hidden=n_hidden,
                          activation_func=activation_func,
                          binarizer=binarizer,
                          regressor=regressor,
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
    assert 0.0 < score <= 1.0, "Score of model is not in expected range"


def test_elm_regressor_reproducible_results():
    # get data
    X, y = make_regression(n_samples=100, n_targets=1, n_features=10, random_state=123)

    # build model 1
    model = ELMRegressor(random_state=123)

    # fit model 1
    model.fit(X, y)

    # predict 1
    y_pred1 = model.predict(X)

    # build model 2
    model = ELMRegressor(random_state=123)

    # fit model 2
    model.fit(X, y)

    # predict 2
    y_pred2 = model.predict(X)

    assert all([y1 == y2 for y1, y2 in zip(y_pred1, y_pred2)]), \
        "Results must be deterministic if random_state is not None"


def test_elm_regressor_random_state_different():
    # get data
    X, y = make_regression(n_samples=100, n_targets=1, n_features=10)

    # build model 1
    model = ELMRegressor(random_state=321)

    # fit model 1
    model.fit(X, y)

    # predict 1
    y_pred1 = model.predict(X)

    # build model 2
    model = ELMRegressor(random_state=123)

    # fit model 2
    model.fit(X, y)

    # predict 2
    y_pred2 = model.predict(X)

    assert not all([y1 == y2 for y1, y2 in zip(y_pred1, y_pred2)]), \
        "Results must be different if random_state changes"


def test_elm_bad_path():
    # predict() without fit
    X, y = make_regression(n_samples=100, n_targets=1, n_features=10)
    model = ELMRegressor()

    with pytest.raises(ValueError):
        model.predict(X)

    # predict() without fit
    X, y = make_regression(n_samples=100, n_targets=1, n_features=10)
    model = GenELMRegressor()

    with pytest.raises(ValueError):
        model.predict(X)

    # predict() without fit
    X, y = make_regression(n_samples=100, n_targets=1, n_features=10)
    model = GenELMClassifier()

    with pytest.raises(ValueError):
        model.predict(X)
    
    # decision_function() without fit
    with pytest.raises(ValueError):
        model.decision_function(X)

    # predict() without fit
    X, y = make_regression(n_samples=100, n_targets=1, n_features=10)
    model = ELMClassifier()

    with pytest.raises(ValueError):
        model.predict(X)

    # predict_proba() without fit
    with pytest.raises(ValueError):
        model.predict_proba(X)

    # decision_function() without fit
    with pytest.raises(ValueError):
        model.decision_function(X)

    # bad activation function
    with pytest.raises(ValueError):
        ELMRegressor(activation_func="bad")

    # bad regressor
    with pytest.raises(ValueError):
        GenELMRegressor(regressor="bad")

    # bad hidden_layer
    with pytest.raises(ValueError):
        GenELMRegressor(hidden_layer="bad")
