Tutorial
========

This sections will explain the concepts behind the implementation of this library, as well as the main usage of ``pyoselm`` in practical examples.

Basic Concepts
--------------

The basis of the implementation is shown through these sections with theory content:

- :ref:`Extreme Learning Machine`
- :ref:`Online Sequential Extreme Learning Machine`


Extreme Learning Machine
~~~~~~~~~~~~~~~~~~~~~~~~

Extreme Learning Machine (ELM) [1] is a simple algorithm for Single-Layer Feed-Forward Neural Network (SLFN). It randomly selects the input weights and biases of the hidden nodes instead of learning these parameters. To do that, it doesn’t require gradient-based `backpropagation <https://en.wikipedia.org/wiki/Backpropagation>`_  but it uses `Moore-Penrose generalized inverse <https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse>`_ to set these weights.
According to their creators, these models are able to produce good generalization performance and learn thousands of times faster than networks trained using backpropagation. In literature, it also shows that these models can outperform support vector machines in both classification and regression applications.

Given a single hidden layer of ELM, suppose that the output function of the i-th hidden node is :math:`{h_{i}(\mathbf{x})=G(\mathbf {a} _{i},b_{i},\mathbf {x} )}`, where :math:`{\mathbf {a}_{i}}` and :math:`{{b}_{i}}` are the parameters of the i-th hidden node and *G* is an activation function. The output function of the ELM for SLFNs with *L* hidden nodes is:

:math:`{\displaystyle f_{L}({\mathbf {x}})=\sum _{i=1}^{L}{\boldsymbol {\beta }}_{i}h_{i}({\mathbf {x}})}`, where :math:`{\displaystyle {\boldsymbol {\beta }}_{i}}` is the output weight of the i-th hidden node, and:

:math:`{\displaystyle \mathbf {h} (\mathbf {x} )=[h_{i}(\mathbf {x} ),...,h_{L}(\mathbf {x} )]}` is the hidden layer output mapping of ELM. Given *N* training samples, the hidden layer output matrix :math:`{\displaystyle \mathbf {H} }` of ELM is given as:

:math:`{\displaystyle {\mathbf {H}}=\left[{\begin{matrix}{\mathbf {h}}({\mathbf {x}}_{1})\\\vdots \\{\mathbf {h}}({\mathbf {x}}_{N})\end{matrix}}\right]=\left[{\begin{matrix}G({\mathbf {a}}_{1},b_{1},{\mathbf {x}}_{1})&\cdots &G({\mathbf {a}}_{L},b_{L},{\mathbf {x}}_{1})\\\vdots &\vdots &\vdots \\G({\mathbf {a}}_{1},b_{1},{\mathbf {x}}_{N})&\cdots &G({\mathbf {a}}_{L},b_{L},{\mathbf {x}}_{N})\end{matrix}}\right]}`

and :math:`{\displaystyle \mathbf {T} }` is the training data target matrix: :math:`{\displaystyle {\mathbf {T}}=\left[{\begin{matrix}{{t}}_{1}\\\vdots \\{{t}}_{N}\end{matrix}}\right]}`

and :math:`{\displaystyle \mathbf {\beta} }` is the weight vector: :math:`{\displaystyle {\mathbf {\beta}}=\left[{\begin{matrix}{ {\beta}}_{1}\\\vdots \\{ {\beta}}_{N}\end{matrix}}\right]}`

The values for weight matrix :math:`{\displaystyle {\mathbf {\beta}}}` are obtained through random sampling. It has been mathematically proved that SLFNs with random hidden nodes have the universal approximation capability, the hidden nodes can be randomly generated independent of the training data and remain fixed; then the hidden layer output matrix :math:`{\mathbf H}` is a constant matrix. Thus, training an SLFN is simply equivalent to finding a least squares solution :math:`{\displaystyle \mathit{\mathbf{\widehat{\beta}}} }` of the linear system:

:math:`{\displaystyle \| \mathbf {H} \mathit{\mathbf{\widehat{\beta}}} -  \mathbf {T} \| = min_\beta \| \mathbf {H} \mathit{\mathbf{\beta}} -  \mathbf {T} \|}`

where :math:`{\displaystyle \| . \|}`  is a norm in Euclidean space. The ELM adopts the smallest norm least squares solution of the above linear system as the output weights; that is,

:math:`{\displaystyle \mathit{\mathbf{\widehat{\beta}}} = \mathbf{H}^{+}\mathbf{T}}`,

where :math:`\mathbf{H}^{+}`  is the Moore-Penrose generalized inverse of matrix :math:`\mathbf{H}`.
If :math:`\mathbf{H}^{T} \mathbf{H}` is nonsingular, then the equation can be written as

:math:`{\displaystyle \mathit{\mathbf{\widehat{\beta}}} = \mathbf{H}^{+}\mathbf{T}= \left( \mathbf{H}^{T} \mathbf{H} \right)^{-1} \mathbf{H}^{T} \mathbf{T}}`

The traditional implementation of this algorithm needs all the training data available to build the model (**batch learning**). In many applications, it is very common that the training data can only be obtained one by one or chunk by chunk. If batch learning algorithms are performed each time new training data is available, the learning process will be very time consuming.

Online Sequential Extreme Learning Machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An online version of ELM named OS-ELM is implemented to learn the training samples successively and incrementally. The learning procedure of OS-ELM consists of an initialization phase and a following sequential learning phase, and the one-by-one OS-ELM is summarized as follows.

In initialization phase, given an initial training set, the initial output weights are given by

:math:`{\displaystyle \mathit{\mathbf{\beta}}_{k-1} = \mathbf{P}_{k-1} \mathbf{H^{T}}_{k-1} \mathbf{T}_{k-1}}`,

where :math:`{\displaystyle \mathbf{P}_{k-1}} = \left( \mathbf{H^{T}}_{k-1} \mathbf{H}_{k-1} \right)^{-1}`, :math:`{\displaystyle \mathbf{H}_{k-1}} = \left[ \mathbf{h^{T}}_{1} \mathbf{h^{T}}_{2} \dots \mathbf{h^{T}}_{k-1} \right]^T` and :math:`{\displaystyle \mathbf{T}_{k-1}} = \left[ \mathbf{t^{T}}_{1} \mathbf{t^{T}}_{2} \dots \mathbf{t^{T}}_{k-1} \right]^T`

In the sequential learning phase, the recursive least square (RLS) algorithm is used to update the output weights in a recursive way. Then for another new sample *k*, the output weights update equations are determined by

:math:`{\displaystyle \mathbf{P}_{k} = \mathbf{P}_{k-1} - \frac{\mathbf{P}_{k-1} \mathbf{h^{T}}_{k} \mathbf{h}_k \mathbf{P}_{k-1}} {1 + \mathbf{h}_k \mathbf{P}_{k-1} \mathbf{h^{T}}_{k}} }`,

:math:`{\displaystyle \mathit{\mathbf{\beta}}_{k} = \mathit{\mathbf{\beta}}_{k-1} + \mathbf{P}_{k} \mathbf{h^{T}}_{k} \left( \mathit{t}_{k} - \mathbf{h}_{k} \mathit{\mathbf{\beta}}_{k-1} \right) }`,

It can be seen that the output weights of OS-ELM are recursively updated based on the intermediate results in the last iteration and the newly arrived data, which can be discarded immediately as soon as they have been learnt, so the computation overhead and the memory requirement of the algorithm are greatly reduced. The above one-by-one OSELM algorithm can be easily extended to chunk-by-chunk type.

As soon as the learning procedure for the arrived observations is completed, the data is discarded. Moreover, it has no prior knowledge about the amount
of the observations which will be presented. Therefore, OS-ELM is an elegant sequential
learning algorithm which can handle both the RBF and additive nodes in the
same framework and can be used to both the classification and function regression problems.


Implementation
~~~~~~~~~~~~~~

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


Classification task with ELM (batch learning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, the dataset used is `hand-written digits dataset <https://scikit-learn.org/stable/datasets/index.html#optical-recognition-of-handwritten-digits-dataset>`_ for a task of images classification.

For ELM algorithms, normalization often improve results (since it can avoid large numbers processed in algebraic operations). In this dataset, every feature is in range [0, 16] so we will scale values to range [0, 1].

Here the test results for ELM are slightly better than SVC.

.. code-block:: python

    from pyoselm import ELMClassifier
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.tree import ExtraTreeClassifier
    from sklearn.svm import SVC

    X, y = load_digits(return_X_y=True)
    X /= 16.  # scale values
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

Here, online learning algorithms are used to learn a dataset in different chunk settings: from one-by-one to chunk-by-chunk.

The dataset used is `california housing <https://scikit-learn.org/stable/datasets/index.html#california-housing-dataset>`_, a hard one.

As pre-processing, standard scaling is applied over the feature values.

We can see that, for these settings, OS-ELM is by far the slowest one but at least it has wide better results than the other algorithms in every setting.
Notice that chunk-by-chunk is faster than row-by-row, and results are almost equal across settings.

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


Classification task with OS-ELM (online learning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, let's try OS-ELM in a classification task, in this case with `forest covertypes <https://scikit-learn.org/stable/datasets/index.html#forest-covertypes>`_

We can see that again OS-ELM has slower but better results than the other online algorithms.

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

- Extreme learning machine: Theory and applications Guang-Bin Huang, Qin-Yu Zhu, Chee-Kheong Siew.
- "Extreme learning machine" in `Wikipedia <https://en.wikipedia.org/wiki/Extreme_learning_machine>`_
- Huang, G. B., Liang, N. Y., Rong, H. J., Saratchandran, P., & Sundararajan, N. (2005).
  On-Line Sequential Extreme Learning Machine. Computational Intelligence, 2005, 232-237.
- Guo, W., Xu, T., Tang, K., Yu, J., & Chen, S. (2018). Online sequential extreme learning machine
  with generalized regularization and adaptive forgetting factor for time-varying system prediction.
  Mathematical Problems in Engineering, 2018.
- Liang N Y, Huang G B, Saratchandran P, et al. A fast and accurate online sequential learning
  algorithm for feedforward networks[J]. Neural Networks, IEEE Transactions on, 2006,
  17(6): 1411-1423.
