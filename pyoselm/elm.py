"""Module to build Extreme Learning Machine (ELM) models"""

# ===================================================
# Acknowledgement:
# Author: David C. Lambert [dcl -at- panix -dot- com]
# Copyright(c) 2013
# License: Simple BSD
# ===================================================

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import as_float_array

from pyoselm.layer import RandomLayer, MLPRandomLayer


__all__ = [
    "ELMRegressor",
    "ELMClassifier",
    "GenELMRegressor",
    "GenELMClassifier",
]


class BaseELM(BaseEstimator):
    """Abstract Base class for ELMs"""
    __metaclass__ = ABCMeta

    def __init__(self, hidden_layer, regressor):
        self.regressor = regressor
        self.hidden_layer = hidden_layer

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model using X, y as training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape [n_samples, n_outputs]
            Target values (class labels in classification, real numbers in
            regression)

        Returns
        -------
        self : object
            Returns an instance of self.
        """

    @abstractmethod
    def predict(self, X):
        """
        Predict values using the model

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]

        Returns
        -------
        C : numpy array of shape [n_samples, n_outputs]
            Predicted values.
        """


class GenELMRegressor(BaseELM, RegressorMixin):
    """
    Regression model based on Extreme Learning Machine.

    Parameters
    ----------
    `hidden_layer` : random_layer instance, optional
        (default=MLPRandomLayer(random_state=0))

    `regressor`    : regressor instance, optional
        (default=sklearn.linear_model.LinearRegression())

    Attributes
    ----------
    `coefs_` : numpy array
        Fitted regression coefficients if no regressor supplied.

    `fitted_` : bool
        Flag set when fit has been called already.

    `hidden_activations_` : numpy array of shape [n_samples, n_hidden]
        Hidden layer activations for last input.

    See Also
    --------
    ELMRegressor, MLPRandomLayer
    """

    def __init__(self, hidden_layer=None, regressor=None):
        if hidden_layer is None:
            # Default value
            hidden_layer = MLPRandomLayer(random_state=0)
        elif not isinstance(hidden_layer, RandomLayer):
            raise ValueError("Argument 'hidden_layer' must be a RandomLayer instance")

        if regressor is None:
            # Default value
            regressor = LinearRegression()
        elif not isinstance(regressor, RegressorMixin):
            raise ValueError("Argument 'regressor' must be a RegressorMixin instance")

        super(GenELMRegressor, self).__init__(hidden_layer, regressor)

        self.coefs_ = None
        self.fitted_ = False
        self.hidden_activations_ = None

    def _fit_regression(self, y):
        """Fit regression with the supplied regressor"""
        self.regressor.fit(self.hidden_activations_, y)
        self.fitted_ = True

    @property
    def is_fitted(self):
        """Check if model was fitted

        Returns
        -------
            boolean, True if model is fitted
        """
        return self.fitted_

    def fit(self, X, y):
        """
        Fit the model using X, y as training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape [n_samples, n_outputs]
            Target values (class labels in classification, real numbers in
            regression)

        Returns
        -------
        self : object

            Returns an instance of self.
        """
        # fit random hidden layer and compute the hidden layer activations
        self.hidden_activations_ = self.hidden_layer.fit_transform(X)

        # solve the regression from hidden activations to outputs
        self._fit_regression(as_float_array(y, copy=True))

        return self

    def predict(self, X):
        """
        Predict values using the model

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]

        Returns
        -------
        C : numpy array of shape [n_samples, n_outputs]
            Predicted values.
        """
        if not self.is_fitted:
            raise ValueError("GenELMRegressor not fitted")

        # compute hidden layer activations
        self.hidden_activations_ = self.hidden_layer.transform(X)

        # compute output predictions for new hidden activations
        predictions = self.regressor.predict(self.hidden_activations_)

        return predictions


class GenELMClassifier(BaseELM, ClassifierMixin):
    """
    Classification model based on Extreme Learning Machine.
    Internally, it uses a GenELMRegressor.

    Parameters
    ----------
    `hidden_layer` : random_layer instance, optional
        (default=MLPRandomLayer(random_state=0))

    `binarizer`    : LabelBinarizer, optional
        (default=sklearn.preprocessing.LabelBinarizer(-1, 1))

    `regressor`    : regressor instance, optional
        (default=LinearRegression())
        Used to perform the regression from hidden unit activations
        to the outputs and subsequent predictions.

    Attributes
    ----------
    `classes_` : numpy array of shape [n_classes]
        Array of class labels

    `genelm_regressor_` : ELMRegressor instance
        Performs actual fit of binarized values

    See Also
    --------
    GenELMRegressor, ELMClassifier, MLPRandomLayer
    """
    def __init__(self, hidden_layer=None, binarizer=None, regressor=None):

        # Default values
        if hidden_layer is None:
            hidden_layer = MLPRandomLayer(random_state=0)
        if binarizer is None:
            binarizer = LabelBinarizer(neg_label=-1, pos_label=1)

        super(GenELMClassifier, self).__init__(hidden_layer, regressor)

        self.binarizer = binarizer

        self.classes_ = None
        self._genelm_regressor = GenELMRegressor(hidden_layer, regressor)

    def fit(self, X, y):
        """
        Fit the model using X, y as training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape [n_samples, n_outputs]
            Target values (class labels in classification, real numbers in
            regression)

        Returns
        -------
        self : object

            Returns an instance of self.
        """
        self.classes_ = np.unique(y)

        y_bin = self.binarizer.fit_transform(y)

        self._genelm_regressor.fit(X, y_bin)
        return self

    @property
    def is_fitted(self):
        """Check if model was fitted

        Returns
        -------
            boolean, True if model is fitted
        """
        return self._genelm_regressor is not None and self._genelm_regressor.is_fitted

    def decision_function(self, X):
        """
        This function return the decision function values related to each
        class on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]

        Returns
        -------
        C : array of shape [n_samples, n_classes] or [n_samples,]
            Decision function values related to each class, per sample.
            In the two-class case, the shape is [n_samples,]
        """
        if not self.is_fitted:
            raise ValueError("GenELMClassifier not fitted")

        return self._genelm_regressor.predict(X)

    def predict(self, X):
        """Predict values using the model

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]

        Returns
        -------
        C : numpy array of shape [n_samples, n_outputs]
            Predicted values.
        """
        if not self.is_fitted:
            raise ValueError("GenELMClassifier not fitted")

        raw_predictions = self.decision_function(X)
        class_predictions = self.binarizer.inverse_transform(raw_predictions)

        return class_predictions


class ELMRegressor(BaseEstimator, RegressorMixin):
    """
    Regression model based on Extreme Learning Machine.

    An Extreme Learning Machine (ELM) is a single layer feedforward
    network with a random hidden layer components and ordinary linear
    least squares fitting of the hidden->output weights by default.
    [1][2]

    ELMRegressor is a wrapper for an GenELMRegressor that creates a
    RandomLayer based on the given parameters.

    Parameters
    ----------
    `n_hidden` : int, optional (default=20)
        Number of units to generate in the SimpleRandomLayer

    `alpha` : float, optional (default=0.5)
        Mixing coefficient for distance and dot product input activations:
        activation = alpha*mlp_activation + (1-alpha)*rbf_width*rbf_activation

    `rbf_width` : float, optional (default=1.0)
        multiplier on rbf_activation

    `activation_func` : {callable, string} optional (default='sigmoid')
        Function used to transform input activation

        It must be one of 'tanh', 'sine', 'tribas', 'inv_tribase', 'sigmoid',
        'hardlim', 'softlim', 'gaussian', 'multiquadric', 'inv_multiquadric' or
        a callable.  If none is given, 'tanh' will be used. If a callable
        is given, it will be used to compute the hidden unit activations.

    `activation_args` : dictionary, optional (default=None)
        Supplies keyword arguments for a callable activation_func

    `user_components`: dictionary, optional (default=None)
        dictionary containing values for components that woud otherwise be
        randomly generated.  Valid key/value pairs are as follows:
           'radii'  : array-like of shape [n_hidden]
           'centers': array-like of shape [n_hidden, n_features]
           'biases' : array-like of shape [n_hidden]
           'weights': array-like of shape [n_hidden, n_features]

    `regressor`    : regressor instance, optional
        (default=sklearn.linear_model.LinearRegression())
        Used to perform the regression from hidden unit activations
        to the outputs and subsequent predictions.

    `random_state`  : int, RandomState instance or None (default=None)
        Control the pseudo random number generator used to generate the
        hidden unit weights at fit time.

    Attributes
    ----------
    `genelm_regressor_` : GenELMRegressor object
        Wrapped object that actually performs the fit.

    Examples
    --------
    >>> from pyoselm import ELMRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_targets=1, n_features=10)
    >>> model = ELMRegressor(n_hidden=20,
    ...                      activation_func="tanh",
    ...                      random_state=123)
    >>> model.fit(X, y)
    ELMRegressor(random_state=123)
    >>> model.score(X, y)
    0.8600650083210614

    See Also
    --------
    GenELMRegressor, RandomLayer, MLPRandomLayer,

    References
    ----------
    .. [1] http://www.extreme-learning-machines.org
    .. [2] G.-B. Huang, Q.-Y. Zhu and C.-K. Siew, "Extreme Learning Machine:
          Theory and Applications", Neurocomputing, vol. 70, pp. 489-501,
              2006.
    """

    def __init__(self,
                 n_hidden=20,
                 alpha=0.5,
                 rbf_width=1.0,
                 activation_func='sigmoid',
                 activation_args=None,
                 user_components=None,
                 regressor=None,
                 random_state=None,):

        self.n_hidden = n_hidden
        self.alpha = alpha
        self.random_state = random_state
        self.activation_func = activation_func
        self.activation_args = activation_args
        self.user_components = user_components
        self.rbf_width = rbf_width
        self.regressor = regressor

        # Just to validate input arguments
        self._create_random_layer()

        self._genelm_regressor = None

    def _create_random_layer(self):
        """Pass init params to RandomLayer"""

        return RandomLayer(n_hidden=self.n_hidden,
                           alpha=self.alpha, random_state=self.random_state,
                           activation_func=self.activation_func,
                           activation_args=self.activation_args,
                           user_components=self.user_components,
                           rbf_width=self.rbf_width)

    def fit(self, X, y):
        """
        Fit the model using X, y as training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape [n_samples, n_outputs]
            Target values (class labels in classification, real numbers in
            regression)

        Returns
        -------
        self : object

            Returns an instance of self.
        """
        rhl = self._create_random_layer()
        self._genelm_regressor = GenELMRegressor(hidden_layer=rhl,
                                                 regressor=self.regressor)
        self._genelm_regressor.fit(X, y)
        return self

    @property
    def is_fitted(self):
        """Check if model was fitted

        Returns
        -------
            boolean, True if model is fitted
        """
        return self._genelm_regressor is not None and self._genelm_regressor.is_fitted

    def predict(self, X):
        """
        Predict values using the model

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]

        Returns
        -------
        C : numpy array of shape [n_samples, n_outputs]
            Predicted values.
        """
        if not self.is_fitted:
            raise ValueError("ELMRegressor is not fitted")

        return self._genelm_regressor.predict(X)


# TODO: inherit from BaseELMClassifier
class ELMClassifier(ELMRegressor):
    """
    Classification model based on Extreme Learning Machine.

    An Extreme Learning Machine (ELM) is a single layer feedforward
    network with a random hidden layer components and ordinary linear
    least squares fitting of the hidden->output weights by default.
    [1][2]

    ELMClassifier is an ELMRegressor subclass that first binarizes the
    data, then uses the superclass to compute the decision function that
    is then unbinarized to yield the prediction.

    The params for the RandomLayer used in the input transform are
    exposed in the ELMClassifier constructor.

    Parameters
    ----------
    `n_hidden` : int, optional (default=20)
        Number of units to generate in the SimpleRandomLayer

    `activation_func` : {callable, string} optional (default='sigmoid')
        Function used to transform input activation

        It must be one of 'tanh', 'sine', 'tribas', 'inv_tribase', 'sigmoid',
        'hardlim', 'softlim', 'gaussian', 'multiquadric', 'inv_multiquadric' or
        a callable. If a callable is given, it will be used to compute
        the hidden unit activations.

    `activation_args` : dictionary, optional (default=None)
        Supplies keyword arguments for a callable activation_func

    `random_state`  : int, RandomState instance or None (default=None)
        Control the pseudo random number generator used to generate the
        hidden unit weights at fit time.

    Attributes
    ----------
    `classes_` : numpy array of shape [n_classes]
        Array of class labels

    Examples
    --------
    >>> from pyoselm import ELMClassifier
    >>> from sklearn.datasets import load_digits
    >>> X, y = load_digits(n_class=10, return_X_y=True)
    >>> model = ELMClassifier(n_hidden=50,
    ...                       activation_func="sigmoid",
    ...                       random_state=123)
    >>> model.fit(X, y)
    ELMClassifier(activation_func='sigmoid', n_hidden=50, random_state=123)
    >>> model.score(X, y)
    0.8241513633834168

    See Also
    --------
    RandomLayer, RBFRandomLayer, MLPRandomLayer,
    GenELMRegressor, GenELMClassifier, ELMClassifier

    References
    ----------
    .. [1] http://www.extreme-learning-machines.org
    .. [2] G.-B. Huang, Q.-Y. Zhu and C.-K. Siew, "Extreme Learning Machine:
          Theory and Applications", Neurocomputing, vol. 70, pp. 489-501,
              2006.
    """

    def __init__(self,
                 n_hidden=20,
                 alpha=0.5,
                 rbf_width=1.0,
                 activation_func='sigmoid',
                 activation_args=None,
                 user_components=None,
                 regressor=None,
                 binarizer=LabelBinarizer(neg_label=-1, pos_label=1),
                 random_state=None,):

        super(ELMClassifier, self).__init__(n_hidden=n_hidden,
                                            alpha=alpha,
                                            random_state=random_state,
                                            activation_func=activation_func,
                                            activation_args=activation_args,
                                            user_components=user_components,
                                            rbf_width=rbf_width,
                                            regressor=regressor)

        self.classes_ = None
        self.binarizer = binarizer

    def fit(self, X, y):
        """
        Fit the model using X, y as training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape [n_samples, n_outputs]
            Target values (class labels in classification, real numbers in
            regression)

        Returns
        -------
        self : object

            Returns an instance of self.
        """
        self.classes_ = np.unique(y)

        y_bin = self.binarizer.fit_transform(y)
        super(ELMClassifier, self).fit(X, y_bin)

        return self

    def decision_function(self, X):
        """
        This function return the decision function values related to each
        class on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]

        Returns
        -------
        C : array of shape [n_samples, n_classes] or [n_samples,]
            Decision function values related to each class, per sample.
            In the two-class case, the shape is [n_samples,]
        """
        if not self.is_fitted:
            raise ValueError("ELMClassifier is not fitted")

        return super(ELMClassifier, self).predict(X)

    def predict(self, X):
        """
        Predict values using the model

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]

        Returns
        -------
        C : numpy array of shape [n_samples, n_outputs]
            Predicted values.
        """
        if not self.is_fitted:
            raise ValueError("ELMClassifier is not fitted")

        raw_predictions = self.decision_function(X)
        class_predictions = self.binarizer.inverse_transform(raw_predictions)

        return class_predictions

    def predict_proba(self, X):
        """
        Predict probability values using the model

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]

        Returns
        -------
        P : numpy array of shape [n_samples, n_outputs]
            Predicted probability values.
        """
        if not self.is_fitted:
            raise ValueError("ELMClassifier not fitted")

        raw_predictions = self.decision_function(X)
        # using softmax to translate raw predictions into probability values
        proba_predictions = softmax(raw_predictions)

        return proba_predictions

    def score(self, X, y, **kwargs):
        """Force use of accuracy score since
        it doesn't inherit from ClassifierMixin"""
        return accuracy_score(y, self.predict(X))
