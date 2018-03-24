# pyoselm

*A Python implementation of Online Sequential Extreme Machine Learning (OS-ELM) for online machine learning*


[![Build Status](https://travis-ci.org/leferrad/pyoselm.svg?branch=master)](https://travis-ci.org/leferrad/pyoselm)

### Dependencies

- Numpy
- Scipy
- Scikit-learn

Original publication: 

> Huang, G. B., Liang, N. Y., Rong, H. J., Saratchandran, P., & Sundararajan, N. (2005). 
  On-Line Sequential Extreme Learning Machine. Computational Intelligence, 2005, 232-237.

Link: https://pdfs.semanticscholar.org/2ebd/fa3852e4ad68a0cfde9f0f69b95953d69178.pdf

Implementation strongly based on the following repos:

- https://github.com/ExtremeLearningMachines/ELM-MATLAB-and-Online.Sequential.ELM
- https://github.com/dclambert/Python-ELM

### Usage

```python
from pyoselm import OSELMRegressor, OSELMClassifier
from sklearn.datasets import load_digits, make_regression
import random

# --- Regression problem ---
# Model
oselmr = OSELMRegressor(n_hidden=20, activation_func='tanh')
# Data
x, y = make_regression(n_samples=400, n_targets=1, n_features=10)
n_batch = 20

# Fit model with chunks of data
for i in range(20):
    x_batch = x[i*n_batch:(i+1)*n_batch]
    y_batch = y[i*n_batch:(i+1)*n_batch]

    oselmr.fit(x_batch, y_batch)
    print("Train score for batch %i: %s" % (i+1, str(oselmr.score(x_batch, y_batch))))

# Results
print("Train score of total: %s" % str(oselmr.score(x, y)))

# --- Classification problem ---
# Model 
oselmc = OSELMClassifier(n_hidden=20, activation_func='sigmoid')
# Data
x, y = load_digits(n_class=10, return_X_y=True)

# Shuffle data (to have batches with more than one class)
zip_x_y = zip(x, y)
random.shuffle(zip_x_y)
x, y = [x_y[0] for x_y in zip_x_y], [x_y[1] for x_y in zip_x_y]

# Sequential learning
# The first batch of data should have the same size as neurons in the model to achieve the 1st phase (boosting)
batches_x = [x[:oselmc.n_hidden]] + [[x_i] for x_i in x[oselmc.n_hidden:]]
batches_y = [y[:oselmc.n_hidden]] + [[y_i] for y_i in y[oselmc.n_hidden:]]

for b_x, b_y in zip(batches_x, batches_y):
    oselmc.fit(b_x, b_y)

print("Train score of total: %s" % str(oselmc.score(x, y)))

```

NOTE: Chuck-by-chunk is faster than one-by-one