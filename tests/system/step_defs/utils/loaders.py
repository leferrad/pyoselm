"""Module to share utils for the tests regarding loaders of data and models"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pyoselm import ELMRegressor, OSELMRegressor, ELMClassifier, OSELMClassifier

# Datasets to use for tests
AVAILABLE_DATASETS = {
    "boston": datasets.load_boston,  # regression
    "california": datasets.fetch_california_housing,  # regression
    "iris": datasets.load_iris,  # classification
    "covertype": datasets.fetch_covtype,  # classification
}

# Pipelines to use for tests
AVAILABLE_PIPELINES = {
    "boston": lambda: Pipeline([("scaler", StandardScaler())]),
    "california": lambda: Pipeline([("scaler", StandardScaler())]),
    "iris": lambda: Pipeline([("scaler", StandardScaler())]), 
    "covertype": lambda: Pipeline([("scaler", StandardScaler())]),
}

# Models from pyoselm.elm to use for tests
AVAILABLE_ELM_MODELS = {
    "boston": lambda: ELMRegressor(n_hidden=50, activation_func="sigmoid", random_state=123),
    "california": lambda: ELMRegressor(n_hidden=50, activation_func="sigmoid", random_state=123),
    "iris": lambda: ELMClassifier(n_hidden=20, activation_func="sigmoid", random_state=123),
    "covertype": lambda: ELMClassifier(n_hidden=50, activation_func="sigmoid", random_state=123),
}

# Models from pyoselm.oselm to use for tests
AVAILABLE_OSELM_MODELS = {
    "boston": lambda: OSELMRegressor(n_hidden=70, activation_func="sigmoid", random_state=123),
    "california": lambda: OSELMRegressor(n_hidden=50, activation_func="sigmoid", random_state=123),
    "iris": lambda: OSELMClassifier(n_hidden=20, activation_func="sigmoid", random_state=123),
    "covertype": lambda: OSELMClassifier(n_hidden=50, activation_func="sigmoid", random_state=123),
}


class Dataset:
    """
    Util to store a dataset for tests. 
    It splits data in train and test sets for cross validation.

    Args:
        name (string): name of dataset to load
        X (np.array): features
        y (np.array): target
        test_size (float, optional): Ratio for test set. Defaults to 0.3.
    """
    def __init__(self, name, X, y, test_size=0.3):
        self.name = name
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            random_state=123)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.y = y


class Results:
    def __init__(self, train, test):
        self.train = train
        self.test = test


def load_dataset(key):
    """Load a dataset based on key"""
    if key not in AVAILABLE_DATASETS:
        raise ValueError(f"Not supported dataset for key '{key}'")

    loader = AVAILABLE_DATASETS[key]
    X, y = loader(return_X_y=True)

    return Dataset(key, X, y)


def load_pipeline(key):
    """Load a pipeline based on key"""
    if key not in AVAILABLE_PIPELINES:
        raise ValueError(f"Not supported pipeline for key '{key}'")

    return AVAILABLE_PIPELINES[key]()


def load_elm_model(key):
    """Load an ELM model based on key"""
    if key not in AVAILABLE_ELM_MODELS:
        raise ValueError(f"Not supported ELM model for key '{key}'")

    return AVAILABLE_ELM_MODELS[key]()


def load_oselm_model(key):
    """Load an OSELM model based on key"""
    if key not in AVAILABLE_OSELM_MODELS:
        raise ValueError(f"Not supported OSELM model for key '{key}'")

    return AVAILABLE_OSELM_MODELS[key]()
