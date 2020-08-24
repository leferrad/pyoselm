"""Unit tests for pyoselm.layer module"""

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import make_regression

from pyoselm import layer


@pytest.mark.parametrize("n_hidden", [10, 100, 1000])
@pytest.mark.parametrize("activation_func", ["tanh", "sine", "tribas",
                                             "inv_tribas", "linear",
                                             "relu", "softplus", "sigmoid",
                                             "softlim", "hardlim", "gaussian",
                                             "multiquadric",
                                             "inv_multiquadric"])
def test_random_layer(n_hidden, activation_func):
    # get data
    X, y = make_regression(n_samples=100, n_targets=1, n_features=10)

    # build layer
    ly = layer.RandomLayer(n_hidden=n_hidden,
                           activation_func=activation_func)

    # fit layer
    ly.fit(X, y)

    # transform
    X_ = ly.transform(X)
    assert X_.shape == (X.shape[0], n_hidden), \
        "Transform output must be (n_input, n_hidden)"


@pytest.mark.parametrize("n_hidden", [10, 100, 1000])
def test_grbfrandom_layer(n_hidden):
    # get data
    X, y = make_regression(n_samples=100, n_targets=1, n_features=10)

    # build layer
    ly = layer.GRBFRandomLayer(n_hidden=n_hidden)

    # fit layer
    ly.fit(X, y)

    # transform
    X_ = ly.transform(X)
    assert X_.shape == (X.shape[0], n_hidden), \
        "Transform output must be (n_input, n_hidden)"


@pytest.mark.parametrize("n_hidden", [10, 100, 1000])
def test_random_layer_sparse(n_hidden):
    # get data
    X, y = make_regression(n_samples=100, n_targets=1, n_features=10)

    # make it sparse
    X = sp.csc_matrix(X)

    # build layer
    ly = layer.RandomLayer(n_hidden=n_hidden, activation_func="tanh")

    # fit layer
    ly.fit(X, y)

    # transform
    X_ = ly.transform(X)
    assert X_.shape == (X.shape[0], n_hidden), \
        "Transform output must be (n_input, n_hidden)"


@pytest.mark.parametrize("n_hidden", [10, 100, 1000])
def test_random_layer_custom_activation_func(n_hidden):
    # get data
    X, y = make_regression(n_samples=100, n_targets=1, n_features=10)

    # custom activation func
    def softplus(z):
        r"""
        Softplus
        :math:`f(x)=\log{(1+e^z)}`
        """
        return np.log(1.0 + np.exp(z))

    # build layer
    ly = layer.RandomLayer(n_hidden=n_hidden,
                           activation_func=softplus)

    # fit layer
    ly.fit(X, y)

    # transform
    X_ = ly.transform(X)
    assert X_.shape == (X.shape[0], n_hidden), \
        "Transform output must be (n_input, n_hidden)"


def test_random_layer_activation_func_names():
    ly = layer.RandomLayer()

    names = ly.activation_func_names()
    assert set(names) == {'sine', 'tanh', 'tribas', 'inv_tribas', 'linear',
                          'relu', 'softplus', 'sigmoid', 'softlim',
                          'hardlim', 'gaussian', 'multiquadric',
                          'inv_multiquadric'}


def test_random_layer_reproducible_results():
    # get data
    X, y = make_regression(n_samples=100, n_targets=1, n_features=10)

    # build layer 1
    ly1 = layer.RandomLayer(random_state=123)

    # fit layer 1
    ly1.fit(X, y)

    # transform 1
    X1 = ly1.transform(X)

    # build layer 2
    ly2 = layer.RandomLayer(random_state=123)

    # fit layer 2
    ly2.fit(X, y)

    # transform 2
    X2 = ly2.transform(X)

    assert all([(x1 == x2).all() for x1, x2 in zip(X1, X2)]), \
        "Results must be deterministic if random_state is not None"


def test_random_layer_random_state_different():
    # get data
    X, y = make_regression(n_samples=100, n_targets=1, n_features=10)

    # build layer 1
    ly1 = layer.RandomLayer(random_state=123)

    # fit layer 1
    ly1.fit(X, y)

    # transform 1
    X1 = ly1.transform(X)

    # build layer 2
    ly2 = layer.RandomLayer(random_state=321)

    # fit layer 2
    ly2.fit(X, y)

    # transform 2
    X2 = ly2.transform(X)

    assert not all([(x1 == x2).all() for x1, x2 in zip(X1, X2)]), \
        "Results must be different if random_state changes"


def test_random_layer_bad_path():
    # unknown activation function
    with pytest.raises(ValueError):
        layer.RandomLayer(activation_func="unknown")

    # transform() without fit
    X, y = make_regression(n_samples=100, n_targets=1, n_features=10)
    ly = layer.RandomLayer()

    with pytest.raises(ValueError):
        ly.transform(X)
