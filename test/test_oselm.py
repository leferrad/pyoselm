#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from pyoselm.core import OSELMClassifier, OSELMRegressor

from sklearn.datasets import load_digits, make_regression
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
import time
import random
import sys


def make_toy_data(n=20, start=0.25, delta=0.1):
    end = start + n*delta
    x = np.arange(start, end, delta)
    y = x*np.cos(x)+0.5*np.sqrt(x)*np.random.randn(x.shape[0])
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return x, y


def test_oselm_regression(n_samples=400, n_hidden=25, activation_func='tanh', plot=True):
    x, y = make_regression(n_samples=n_samples, n_targets=1, n_features=10)
    #data = load_diabetes()
    #x, y = data['data'], data['target']

    n_batches = n_samples / n_hidden

    n_batch = n_hidden

    oselmr = OSELMRegressor(n_hidden=n_hidden, activation_func=activation_func)

    ypred = []

    tic = time.time()

    for i in range(n_batches):
        x_batch = x[i*n_batch:(i+1)*n_batch]
        y_batch = y[i*n_batch:(i+1)*n_batch]
        oselmr.fit(x_batch, y_batch)
        ypred.extend(oselmr.predict(x_batch))
        print("Train score for batch %i: %s" % (i+1, str(oselmr.score(x_batch, y_batch))))

    toc = time.time()

    print("Total time: %.3f seconds" % (toc-tic))

    if plot is True:
        axis_x = range(len(y))
        plt.plot(axis_x, y, axis_x, ypred)
        plt.show()


def test_oselm_regression_toy():
    stdsc = StandardScaler()

    N = 200  # Total data
    total_batches = 5
    n_batch = N / total_batches  # batch size

    xtoy, ytoy = make_toy_data(n=N)
    xtoy, ytoy = stdsc.fit_transform(xtoy), stdsc.fit_transform(ytoy)

    #xtoy_train, xtoy_test, ytoy_train, ytoy_test = train_test_split(xtoy, ytoy, test_size=0.2)
    #plt.plot(xtoy, ytoy)
    #plt.show()

    oselmr = OSELMRegressor(n_hidden=20, activation_func='tanh')

    ypred = []

    for i in range(total_batches):
        xtoy_batch = xtoy[i*n_batch:(i+1)*n_batch]
        ytoy_batch = ytoy[i*n_batch:(i+1)*n_batch]
        oselmr.fit(xtoy_batch, ytoy_batch)
        ypred.extend(oselmr.predict(xtoy_batch))
        print("Train score for batch %i: %s" % (i+1, str(oselmr.score(xtoy_batch, ytoy_batch))))

    plt.plot(xtoy, ytoy, xtoy, ypred)
    plt.show()


def test_oselm_regression_sequential(n_samples=2000, n_hidden=20, activation_func='tanh', plot=True):
    x, y = make_regression(n_samples=n_samples, n_targets=1, n_features=10)

    start_i = n_hidden+20

    batches_x = [x[:start_i]] + [[x_i] for x_i in x[start_i:]]
    batches_y = [y[:start_i]] + [[y_i] for y_i in y[start_i:]]

    oselmr = OSELMRegressor(n_hidden=n_hidden, activation_func=activation_func)

    ypred = []

    tic = time.time()

    for b_x, b_y in zip(batches_x, batches_y):
        oselmr.fit(b_x, b_y)
        ypred.extend(oselmr.predict(b_x))

    print("Train score for total: %s" % str(oselmr.score(x, y)))

    toc = time.time()

    print("Total time: %.3f seconds" % (toc-tic))

    if plot is True:
        axis_x = range(len(y))
        plt.plot(axis_x, y, axis_x, ypred)
        plt.show()


def test_oselm_classification(n_batches=10, n_classes=8, n_hidden=100, activation_func='sigmoid'):
    data = load_digits(n_class=n_classes)
    x, y = data['data'], data['target']

    # Shuffle data
    zip_x_y = zip(x, y)
    random.shuffle(zip_x_y)
    x, y = [x_y[0] for x_y in zip_x_y], [x_y[1] for x_y in zip_x_y]

    n_samples = len(y)

    print("Data have %i samples" % n_samples)

    n_batch = n_samples/n_batches  # batch size


    oselmc = OSELMClassifier(n_hidden=n_hidden, activation_func=activation_func)
    scores = []

    for i in range(n_batches):
        x_batch = x[i*n_batch:(i+1)*n_batch]
        y_batch = y[i*n_batch:(i+1)*n_batch]
        oselmc.fit(x_batch, y_batch)
        #ypred.extend(oselmc.predict(x_batch))
        score_batch = oselmc.score(x_batch, y_batch)
        scores.append(score_batch)
        print("Train score for batch %i: %s" % (i+1, str(score_batch)))

    print("Train score - online: %s" % str(np.mean(scores)))
    print("Train score - offline: %s" % str(oselmc.score(x, y)))


test_modes = {'regression': lambda: test_oselm_regression_sequential(n_hidden=50, n_samples=1000,
                                                                     activation_func='tanh', plot=True),
              'classification': lambda: test_oselm_classification(n_hidden=100, n_batches=4, n_classes=10)
              }


if __name__ == '__main__':
    mode = 'regression'

    if len(sys.argv) > 1:

        argmode = sys.argv[1]

        if argmode == 'classification':
            mode = 'classification'
        else:
            mode = 'regression'

    print("Executing test in mode=%s..." % mode)

    test_modes[mode]()

