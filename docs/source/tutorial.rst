Tutorial
========

This sections will explain the concepts behind the implementation of this library, as well as the main usage of ``pyoselm`` in practical examples.

Basic Concepts
--------------

Extreme Learning Machine (ELM) is a simple algorithm for Single-Layer Feed-Forward Neural Network (SLFN). It randomly selects the input weights and biases of the hidden nodes instead of learning these parameters. To do that, it doesn’t require gradient-based `backpropagation <https://en.wikipedia.org/wiki/Backpropagation>`_  but it uses `Moore-Penrose generalized inverse <https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse>`_ to set these weights.

Given a single hidden layer of ELM, suppose that the output function of the i-th hidden node is :math:`{h_{i}(\mathbf{x})=G(\mathbf {a} _{i},b_{i},\mathbf {x} )}`, where :math:`{\mathbf {a}_{i}}` and :math:`{{b}_{i}}` are the parameters of the i-th hidden node and *G* is an activation function. The output function of the ELM for SLFNs with *L* hidden nodes is:

:math:`{\displaystyle f_{L}({\mathbf {x}})=\sum _{i=1}^{L}{\boldsymbol {\beta }}_{i}h_{i}({\mathbf {x}})}`, where :math:`{\displaystyle {\boldsymbol {\beta }}_{i}}` is the output weight of the i-th hidden node, and:

:math:`{\displaystyle \mathbf {h} (\mathbf {x} )=[h_{i}(\mathbf {x} ),...,h_{L}(\mathbf {x} )]}` is the hidden layer output mapping of ELM. Given *N* training samples, the hidden layer output matrix :math:`{\displaystyle \mathbf {H} }` of ELM is given as:

:math:`{\displaystyle {\mathbf {H}}=\left[{\begin{matrix}{\mathbf {h}}({\mathbf {x}}_{1})\\\vdots \\{\mathbf {h}}({\mathbf {x}}_{N})\end{matrix}}\right]=\left[{\begin{matrix}G({\mathbf {a}}_{1},b_{1},{\mathbf {x}}_{1})&\cdots &G({\mathbf {a}}_{L},b_{L},{\mathbf {x}}_{1})\\\vdots &\vdots &\vdots \\G({\mathbf {a}}_{1},b_{1},{\mathbf {x}}_{N})&\cdots &G({\mathbf {a}}_{L},b_{L},{\mathbf {x}}_{N})\end{matrix}}\right]}`

and :math:`{\displaystyle \mathbf {T} }` is the training data target matrix: :math:`{\displaystyle {\mathbf {T}}=\left[{\begin{matrix}{\mathbf {t}}_{1}\\\vdots \\{\mathbf {t}}_{N}\end{matrix}}\right]}`

Generally speaking, ELM is a kind of regularization neural networks but with non-tuned hidden layer mappings (formed by either random hidden nodes, kernels or other implementations), its objective function is:

{\displaystyle {\text{Minimize: }}\|{\boldsymbol {\beta }}\|_{p}^{\sigma _{1}}+C\|{\bf {H}}{\boldsymbol {\beta }}-{\bf {T}}\|_{q}^{\sigma _{2}}} {\displaystyle {\text{Minimize: }}\|{\boldsymbol {\beta }}\|_{p}^{\sigma _{1}}+C\|{\bf {H}}{\boldsymbol {\beta }}-{\bf {T}}\|_{q}^{\sigma _{2}}}

where {\displaystyle \sigma _{1}>0,\sigma _{2}>0,p,q=0,{\frac {1}{2}},1,2,\cdots ,+\infty } {\displaystyle \sigma _{1}>0,\sigma _{2}>0,p,q=0,{\frac {1}{2}},1,2,\cdots ,+\infty }.

Different combinations of {\displaystyle \sigma _{1}} \sigma _{1}, {\displaystyle \sigma _{2}} \sigma _{2}, {\displaystyle p} p and {\displaystyle q} q can be used and result in different learning algorithms for regression, classification, sparse coding, compression, feature learning and clustering.

As a special case, a simplest ELM training algorithm learns a model of the form (for single hidden layer sigmoid neural networks):

{\displaystyle \mathbf {\hat {Y}} =\mathbf {W} _{2}\sigma (\mathbf {W} _{1}x)} {\mathbf  {{\hat  {Y}}}}={\mathbf  {W}}_{2}\sigma ({\mathbf  {W}}_{1}x)
where W1 is the matrix of input-to-hidden-layer weights, {\displaystyle \sigma } \sigma  is an activation function, and W2 is the matrix of hidden-to-output-layer weights. The algorithm proceeds as follows:

Fill W1 with random values (e.g., Gaussian random noise);
estimate W2 by least-squares fit to a matrix of response variables Y, computed using the pseudoinverse ⋅+, given a design matrix X:
{\displaystyle \mathbf {W} _{2}=\sigma (\mathbf {W} _{1}\mathbf {X} )^{+}\mathbf {Y} } {\mathbf  {W}}_{2}=\sigma ({\mathbf  {W}}_{1}{\mathbf  {X}})^{+}{\mathbf  {Y}}


Algorithm

1. Create the random weights matrix W and bias b for the input layer.
The size of the weight matrix and bias is (j x k) and (1 x k) where j is the number of hidden nodes and k is the number of input nodes.

2. Calculate the hidden layer output matrix
The initial hidden layer output matrix is calculated by multiplying X which is training data with transpose of weight matrix



3. Choose activation function
You can choose any activation function that you want. But in this example I will choose sigmoid activation function because it is easy to implement.
Image for post
4. Calculate the Moore-Penrose pseudoinverse
Several methods can be used to calculate the Moore–Penrose generalized inverse of H. These methods may include but are not limited to orthogonal projection, orthogonalization method, iterative method, and singular value decomposition (SVD).
Image for post
5. Calculate the output weight matrix beta
Image for post
6. Repeat step 2 for the testing dataset, creating a new H matrix. After that, create the result matrix called ŷ. We use the already known beta matrix.
Image for post


According to their creators, these models are able to produce good generalization performance and learn thousands of times faster than networks trained using backpropagation. In literature, it also shows that these models can outperform support vector machines in both classification and regression applications.

The traditional implementation of this algorithm needs all the training data available to build the model (**batch learning**). In many applications, it is very common that the training data can only be obtained one by one or chunk by chunk. If batch learning algorithms are performed each time new training data is available, the learning process will be very time consuming. An Online Sequential Extreme Learning Machine (OS-ELM) can learn the sequential training observations online at arbitrary length (one by one or chunk by chunk). New arrived training observations are learned to modify the
model of the SLFNs. As soon as the learning procedure for the arrived observations is
completed, the data is discarded. Moreover, it has no prior knowledge about the amount
of the observations which will be presented. Therefore, OS-ELM is an elegant sequential
learning algorithm which can handle both the RBF and additive nodes in the
same framework and can be used to both the classification and function regression problems. OS-ELM proves to be a very fast and accurate online sequential learning
algorithm[9-11], which can provide better generalization performance in faster speed
compared with other sequential learning algorithms such as GAP-RBF, GGAP-RBF,
SGBP, RAN, RANEKF and MRAN etc



Step-by-step implementation ??
https://medium.com/datadriveninvestor/extreme-learning-machine-for-simple-classification-e776ad797a3c

Least squares explanation
https://medium.com/datadriveninvestor/extreme-learning-machines-9c8be01f6f77


The implementation for this library is strongly based on the following code repositories:

- https://github.com/ExtremeLearningMachines/ELM-MATLAB-and-Online.Sequential.ELM
- https://github.com/dclambert/Python-ELM



Usage
-----

This tutorial will cover 4 scenarios, which are the main use cases for this library:

- :ref:`Regression task with ELM (batch learning)`
- :ref:`Classification task with ELM (batch learning)`
- :ref:`Regression task with OS-ELM (online learning)`
- :ref:`Classification task with OS-ELM (online learning)`

For each case, a dataset is loaded, split, fit and validate with scores. no pre-processing was applied, but normalization usually helps. Configurations were obtained with minimal experimentation, you can try more


Mention that you can find more use cases in tests

Regression task with ELM (batch learning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, the `diabetes dataset <https://scikit-learn.org/stable/datasets/index.html#diabetes-dataset>`_ is loaded to perform a regression task where ELM is compared with other 2 algorithms that normally perform well in regression.
Notice the same scikit-learn API used to fit models and get scores.
We can see that ELM model obtained the best results in the test dataset.

Batch learning, so all training data is used in a single fashion

.. code-block:: python

    from pyoselm import ELMRegressor
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.tree import ExtraTreeRegressor

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    models = {
        "elm": ELMRegressor(n_hidden=20, activation_func='sigmoid', random_state=123),
        "ridge": Ridge(),
        "extra_tree": ExtraTreeRegressor(max_depth=5, random_state=123)
    }

    for name, model in models.items():
        # Fit with train data
        model.fit(X_train, y_train)

        # Validate scores
        print("Train score for '%s': %s" % (name, str(model.score(X_train, y_train))))
        print("Test score for '%s': %s" % (name, str(model.score(X_test, y_test))))
        print("")

Output:

.. code-block:: none

    Train score for 'elm': 0.5212637443701116
    Test score for 'elm': 0.5823062691305605

    Train score for 'ridge': 0.4247361852792363
    Test score for 'ridge': 0.43601545008459586

    Train score for 'extra_tree': 0.5307017943154888
    Test score for 'extra_tree': 0.24042160732597384



Normalization often improve results


Classification task with ELM (batch learning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

the `hand-written digits datasets <https://scikit-learn.org/stable/datasets/index.html#optical-recognition-of-handwritten-digits-dataset>`_

Every feature is in range [0, 255] so scale ...

.. code-block:: python

    from pyoselm import ELMClassifier
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.tree import ExtraTreeClassifier
    from sklearn.svm import SVC

    X, y = load_digits(return_X_y=True)
    X /= 16.  # scale range
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    models = {
        "elm": ELMClassifier(n_hidden=400, rbf_width=0.2, activation_func='sigmoid', random_state=123),
        "svc": SVC(),
        "extra_tree": ExtraTreeClassifier(max_depth=12, random_state=123)
    }

    for name, model in models.items():
        # Fit with train data
        model.fit(X_train, y_train)

        # Validate scores
        print("Train score for '%s': %s" % (name, str(model.score(X_train, y_train))))
        print("Test score for '%s': %s" % (name, str(model.score(X_test, y_test))))
        print("")


Output:

.. code-block:: none

    Train score for 'elm': 0.9993041057759221
    Test score for 'elm': 0.9916666666666667

    Train score for 'svc': 0.9972164231036882
    Test score for 'svc': 0.9888888888888889

    Train score for 'extra_tree': 0.9659011830201809
    Test score for 'extra_tree': 0.7805555555555556



Regression task with OS-ELM (online learning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, online learning algorithms are used

`california housing <https://scikit-learn.org/stable/datasets/index.html#california-housing-dataset>`_


Standard scaling is applied

.. code-block:: python

    import numpy as np
    from pyoselm import OSELMRegressor
    from sklearn.datasets import fetch_california_housing
    from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import time


    def prepare_datasets(X, y):
        """Get train and test datasets from data 'X' and 'y',
        with proper standard scaling"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

        # Scale data
        scaler = StandardScaler()
        scaler.fit(X_train, y_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test


    def fit_sequential(model, X, y, n_hidden, chunk_size=1):
        """Fit 'model' with data 'X' and 'y', sequentially with mini-batches of
        'chunk_size' (starting with a batch of 'n_hidden' size)"""
        # Sequential learning
        N = len(y)
        # The first batch of data must have the same size as n_hidden to achieve the first phase (boosting)
        batches_x = [X[:n_hidden]] + [X[i:i+chunk_size] for i in np.arange(n_hidden, N, chunk_size)]
        batches_y = [y[:n_hidden]] + [y[i:i+chunk_size] for i in np.arange(n_hidden, N, chunk_size)]

        for b_x, b_y in zip(batches_x, batches_y):
            if isinstance(model, OSELMRegressor):
                model.fit(b_x, b_y)
            else:
                model.partial_fit(b_x, b_y)

        return model

    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = prepare_datasets(X, y)

    n_hidden = 50

    models = {
        "elm": OSELMRegressor(n_hidden=n_hidden, activation_func='sigmoid', random_state=123),
        "sgd": SGDRegressor(random_state=123),
        "par": PassiveAggressiveRegressor(random_state=123),
    }

    chunk_sizes = [1, 100, 1000]

    for name, model in models.items():
        for chunk_size in chunk_sizes:
            print("Chunk size: %i" % chunk_size)

            # Fit with train data
            tic = time.time()
            fit_sequential(model, X_train, y_train, n_hidden, chunk_size)
            toc = time.time()

            # Validate scores
            print("Train score for '%s': %s" % (name, str(model.score(X_train, y_train))))
            print("Test score for '%s': %s" % (name, str(model.score(X_test, y_test))))
            print("Time elapsed: %.3f seconds" % (toc - tic))
            print("")

Output:

.. code-block:: none

    Chunk size: 1
    Train score for 'elm': 0.6772104248443173
    Test score for 'elm': 0.6892117330471859
    Time elapsed: 30.612 seconds

    Chunk size: 100
    Train score for 'elm': 0.6772104248443171
    Test score for 'elm': 0.6892117330472057
    Time elapsed: 0.355 seconds

    Chunk size: 1000
    Train score for 'elm': 0.6772104248443173
    Test score for 'elm': 0.6892117330472123
    Time elapsed: 0.076 seconds

    Chunk size: 1
    Train score for 'sgd': -5.329411541998101
    Test score for 'sgd': -4.022169383999319
    Time elapsed: 5.255 seconds

    Chunk size: 100
    Train score for 'sgd': -78.37459505298487
    Test score for 'sgd': -60.95606987138426
    Time elapsed: 0.091 seconds

    Chunk size: 1000
    Train score for 'sgd': -467.17719294082826
    Test score for 'sgd': -363.8240429481794
    Time elapsed: 0.012 seconds

    Chunk size: 1
    Train score for 'par': -0.6580243270843822
    Test score for 'par': -0.5253735567975708
    Time elapsed: 5.141 seconds

    Chunk size: 100
    Train score for 'par': -0.8446600153025225
    Test score for 'par': -0.8747683620177154
    Time elapsed: 0.088 seconds

    Chunk size: 1000
    Train score for 'par': -0.615051709774799
    Test score for 'par': -0.5496495064733955
    Time elapsed: 0.011 seconds


Notice that chunk-by-chunk is faster than row-by-row, and results are almost equal.

ELM is slower but has better performance


Classification task with OS-ELM (online learning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`forest covertypes <https://scikit-learn.org/stable/datasets/index.html#forest-covertypes>`_


.. code-block:: python

    import numpy as np
    from pyoselm import OSELMClassifier
    from sklearn.datasets import fetch_covtype
    from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import time


    def prepare_datasets(X, y):
        """Get train and test datasets from data 'X' and 'y',
        with proper standard scaling"""

        idx = [i for i in range(len(y)) if y[i] in [1, 2, 5]]
        X = X[idx, :]
        y = y[idx]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

        # Scale data
        scaler = StandardScaler()
        scaler.fit(X_train, y_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test


    def fit_sequential(model, X, y, n_hidden, chunk_size=1):
        """Fit 'model' with data 'X' and 'y', sequentially with mini-batches of
        'chunk_size' (starting with a batch of 'n_hidden' size)"""
        # Sequential learning
        N = len(y)
        # The first batch of data must have the same size as n_hidden to achieve the first phase (boosting)
        batches_x = [X[:n_hidden]] + [X[i:i+chunk_size] for i in np.arange(n_hidden, N, chunk_size)]
        batches_y = [y[:n_hidden]] + [y[i:i+chunk_size] for i in np.arange(n_hidden, N, chunk_size)]

        for b_x, b_y in zip(batches_x, batches_y):
            if isinstance(model, OSELMClassifier):
                model.fit(b_x, b_y)
            else:
                model.partial_fit(b_x, b_y, classes=[1, 2, 5])

        return model

    X, y = fetch_covtype(return_X_y=True)
    X_train, X_test, y_train, y_test = prepare_datasets(X, y)

    n_hidden = 100

    models = {
        "elm": OSELMClassifier(n_hidden=n_hidden, activation_func='sigmoid', random_state=123),
        "sgd": SGDClassifier(random_state=123),
        "par": PassiveAggressiveClassifier(random_state=123),
    }

    chunk_sizes = [1000]

    for name, model in models.items():
        for chunk_size in chunk_sizes:
            print("Chunk size: %i" % chunk_size)

            # Fit with train data
            tic = time.time()
            fit_sequential(model, X_train, y_train, n_hidden, chunk_size)
            toc = time.time()

            # Validate scores
            print("Train score for '%s': %s" % (name, str(model.score(X_train, y_train))))
            print("Test score for '%s': %s" % (name, str(model.score(X_test, y_test))))
            print("Time elapsed: %.3f seconds" % (toc - tic))
            print("")

Output:

.. code-block:: none

    Chunk size: 1000
    Train score for 'elm': 0.7624019400208567
    Test score for 'elm': 0.7628285790720025
    Time elapsed: 4.594 seconds

    Chunk size: 1000
    Train score for 'sgd': 0.742144674231559
    Test score for 'sgd': 0.7432104392283532
    Time elapsed: 0.737 seconds

    Chunk size: 1000
    Train score for 'par': 0.6749350395212369
    Test score for 'par': 0.6759340909766465
    Time elapsed: 0.788 seconds


References
----------

[1] Extreme learning machine: Theory and applications Guang-Bin Huang, Qin-Yu Zhu, Chee-Kheong Siew.


**Original publication:**

> Huang, G. B., Liang, N. Y., Rong, H. J., Saratchandran, P., & Sundararajan, N. (2005).
  On-Line Sequential Extreme Learning Machine. Computational Intelligence, 2005, 232-237.


Liang N Y, Huang G B, Saratchandran P, et al. A fast and accurate online sequential learning
algorithm for feedforward networks[J]. Neural Networks, IEEE Transactions on, 2006,
17(6): 1411-1423.

> Huang, G. B., Liang, N. Y., Rong, H. J., Saratchandran, P., & Sundararajan, N. (2005).
  On-Line Sequential Extreme Learning Machine. Computational Intelligence, 2005, 232-237.
