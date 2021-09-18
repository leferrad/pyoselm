<img style="display: inline;" src="docs/img/pyoselm_logo.png" width="300"/>

*A Python implementation of Online Sequential Extreme Machine Learning (OS-ELM) for online machine learning*

[![CI Pipeline](https://github.com/leferrad/pyoselm/actions/workflows/ci_pipeline.yml/badge.svg)](https://github.com/leferrad/pyoselm/actions/workflows/ci_pipeline.yml)
[![Documentation Status](http://readthedocs.org/projects/pyoselm/badge/?version=latest)](http://pyoselm.readthedocs.io/?badge=latest)
[![Coverage Status](https://codecov.io/gh/leferrad/pyoselm/branch/master/graph/badge.svg)](https://codecov.io/gh/leferrad/pyoselm)

### Description

**pyoselm** is a Python library for machine learning models with Extreme Machine Learning (ELM) and Online Sequential Machine Learning (OS-ELM). It allows to fit models for regression and classification tasks, both in batch and online learning (either row-by-row or chunk-by-chunk).

This library offers a scikit-learn like API for easy usage. For more details about setup and usage, check the [documentation](http://readthedocs.org/projects/pyoselm/).

> **IMPORTANT:** This library was developed as a research project. It may not be production-ready, so please be aware of that.

### Setup

The easiest way to install this library is using `pip`:

```
$ pip install pyoselm
```

### Usage

Here a simple but complete example of usage.

```python
from pyoselm import OSELMRegressor, OSELMClassifier
from sklearn.datasets import load_digits, make_regression 
from sklearn.model_selection import train_test_split

print("Regression task")
# Model
oselmr = OSELMRegressor(n_hidden=20, activation_func='sigmoid', random_state=123)
# Data
X, y = make_regression(n_samples=1000, n_targets=1, n_features=10, random_state=123)   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
n_batch = 40

# Fit model with chunks of data
for i in range(20):
    X_batch = X_train[i*n_batch:(i+1)*n_batch]
    y_batch = y_train[i*n_batch:(i+1)*n_batch]
    oselmr.fit(X_batch, y_batch)
    print("Train score for batch %i: %s" % (i+1, str(oselmr.score(X_batch, y_batch))))

# Results
print("Train score of total: %s" % str(oselmr.score(X_train, y_train)))
print("Test score of total: %s" % str(oselmr.score(X_test, y_test)))  
print("")


print("Classification task")
# Model 
oselmc = OSELMClassifier(n_hidden=20, activation_func='sigmoid', random_state=123)
# Data
X, y = load_digits(n_class=5, return_X_y=True) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Sequential learning
# The first batch of data must have the same size as n_hidden to achieve the first phase (boosting)
batches_x = [X_train[:oselmc.n_hidden]] + [[x_i] for x_i in X_train[oselmc.n_hidden:]]
batches_y = [y_train[:oselmc.n_hidden]] + [[y_i] for y_i in y_train[oselmc.n_hidden:]]

for b_x, b_y in zip(batches_x, batches_y):
    oselmc.fit(b_x, b_y)

print("Train score of total: %s" % str(oselmc.score(X_train, y_train)))
print("Test score of total: %s" % str(oselmc.score(X_test, y_test)))
```
