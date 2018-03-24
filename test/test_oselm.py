#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from pyoselm.core import OSELMClassifier, OSELMRegressor

from sklearn.datasets import load_digits, make_regression
from sklearn.metrics import confusion_matrix

import numpy as np
import time
import random
import sys


def make_batch_predictions(model, x, y, n_batch=50):
    n_samples = len(y)
    n_batches = int(n_samples / n_batch)

    print("Dataset has %i samples" % n_samples)
    print("Testing over %i non-overlapped batches with a max of %i samples..." % (n_batches, n_batch))

    scores, preds = [], []

    tic = time.time()

    for i in range(n_batches):
        x_batch = x[i*n_batch:(i+1)*n_batch]
        y_batch = y[i*n_batch:(i+1)*n_batch]

        model.fit(x_batch, y_batch)
        preds.extend(model.predict(x_batch))
        scores.append(model.score(x_batch, y_batch))
        print("Train score for batch %i: %s" % (i+1, str(scores[-1])))

    print("Train score - online: %s" % str(np.mean(scores)))
    print("Train score - offline: %s" % str(model.score(x, y)))

    toc = time.time()

    print("Total time: %.3f seconds" % (toc-tic))

    return model, preds


def make_sequential_predictions(model, x, y):
    n_samples = len(y)

    # The first batch of data should have the same size as neurons in the model to achieve the 1st phase (boosting)
    batches_x = [x[:model.n_hidden]] + [[x_i] for x_i in x[model.n_hidden:]]
    batches_y = [y[:model.n_hidden]] + [[y_i] for y_i in y[model.n_hidden:]]

    print("Testing over %i samples in a online way..." % n_samples)

    preds = []

    tic = time.time()

    for b_x, b_y in zip(batches_x, batches_y):

        model.fit(b_x, b_y)
        preds.extend(model.predict(b_x))

    print("Train score of total: %s" % str(model.score(x, y)))

    toc = time.time()

    print("Total time: %.3f seconds" % (toc-tic))

    return model, preds


def test_oselm_regression_batch(n_samples=400, n_hidden=25, activation_func='tanh', plot=True):
    x, y = make_regression(n_samples=n_samples, n_targets=1, n_features=10)

    oselmr = OSELMRegressor(n_hidden=n_hidden, activation_func=activation_func)

    oselmr, y_pred = make_batch_predictions(model=oselmr, x=x, y=y, n_batch=n_hidden)

    if plot is True:
        import matplotlib.pyplot as plt

        axis_x = range(len(y))
        plt.plot(axis_x, y, axis_x, y_pred)
        plt.show()


def test_oselm_regression_sequential(n_samples=2000, n_hidden=20, activation_func='tanh', plot=True):
    x, y = make_regression(n_samples=n_samples, n_targets=1, n_features=10)

    oselmr = OSELMRegressor(n_hidden=n_hidden, activation_func=activation_func)

    oselmr, y_pred = make_sequential_predictions(model=oselmr, x=x, y=y)

    if plot is True:
        import matplotlib.pyplot as plt

        axis_x = range(len(y))
        plt.plot(axis_x, y, axis_x, y_pred)
        plt.show()


def test_oselm_classification_batch2(n_batches=10, n_hidden=100, activation_func='sigmoid'):
    x, y = load_digits(n_class=10, return_X_y=True)

    # Shuffle data
    zip_x_y = zip(x, y)
    random.shuffle(zip_x_y)
    x, y = [x_y[0] for x_y in zip_x_y], [x_y[1] for x_y in zip_x_y]

    n_samples = len(y)

    print("Data have %i samples" % n_samples)

    n_batch = n_samples/n_batches  # batch size

    oselmc = OSELMClassifierSoftmax(n_hidden=n_hidden, activation_func=activation_func)
    y_pred, scores = [], []

    for i in range(n_batches):
        x_batch = x[i*n_batch:(i+1)*n_batch]
        y_batch = y[i*n_batch:(i+1)*n_batch]
        oselmc.fit(x_batch, y_batch)
        y_pred.extend(oselmc.predict(x_batch))
        score_batch = oselmc.score(x_batch, y_batch)
        scores.append(score_batch)
        print("Train score for batch %i: %s" % (i+1, str(score_batch)))

    print("Train score - online: %s" % str(np.mean(scores)))
    print("Train score - offline: %s" % str(oselmc.score(x, y)))
    print("Confusion matrix: \n %s" % str(confusion_matrix(y[:-1], y_pred)))


def test_oselm_classification_batch(n_hidden=100, activation_func='sigmoid'):
    x, y = load_digits(n_class=10, return_X_y=True)

    # Shuffle data
    zip_x_y = zip(x, y)
    random.shuffle(zip_x_y)
    x, y = [x_y[0] for x_y in zip_x_y], [x_y[1] for x_y in zip_x_y]

    oselmc = OSELMClassifier(n_hidden=n_hidden, activation_func=activation_func)

    oselmc, y_pred = make_batch_predictions(model=oselmc, x=x, y=y, n_batch=n_hidden)

    max_len = min(len(y), len(y_pred))

    print("Confusion matrix: \n %s" % str(confusion_matrix(y[:max_len], y_pred[:max_len])))


def test_oselm_classification_sequential(n_hidden=100, activation_func='sigmoid'):
    x, y = load_digits(n_class=10, return_X_y=True)

    # Shuffle data
    zip_x_y = zip(x, y)
    random.shuffle(zip_x_y)
    x, y = [x_y[0] for x_y in zip_x_y], [x_y[1] for x_y in zip_x_y]

    oselmc = OSELMClassifier(n_hidden=n_hidden, activation_func=activation_func)

    oselmc, y_pred = make_sequential_predictions(model=oselmc, x=x, y=y)

    max_len = min(len(y), len(y_pred))

    print("Confusion matrix: \n %s" % str(confusion_matrix(y[:max_len], y_pred[:max_len])))


test_modes = {
              'classification_batch': lambda: test_oselm_classification_batch(n_hidden=100),
              'classification_sequential': lambda: test_oselm_classification_sequential(n_hidden=100),
              'regression_batch': lambda: test_oselm_regression_batch(n_hidden=50, n_samples=1000,
                                                                      activation_func='tanh', plot=False),
              'regression_sequential': lambda: test_oselm_regression_sequential(n_hidden=50, n_samples=1000,
                                                                                activation_func='tanh',
                                                                                plot=False)
              }


if __name__ == '__main__':
    mode = 'regression_batch'

    if len(sys.argv) > 1:

        argmode = sys.argv[1]

        if argmode in test_modes:
            mode = argmode

    print("Executing test in mode=%s..." % mode)

    test_modes[mode]()

