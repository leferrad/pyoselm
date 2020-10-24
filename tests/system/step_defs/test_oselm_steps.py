"""Steps for oselm.feature"""

import pytest
from pytest_bdd import scenarios, given, when

from .utils.loaders import load_oselm_model


scenarios("../features/oselm.feature")


@given("an OSELMRegressor model")
def oselm_regressor_model(dataset):
    key = dataset.name
    yield load_oselm_model(key)


@given("an OSELMClassifier model")
def oselm_classifier_model(dataset):
    key = dataset.name
    yield load_oselm_model(key)

@given("a model")
def model(dataset):
    key = dataset.name
    yield load_oselm_model(key)


@pytest.fixture
@when("I fit the pipeline and the OSELMRegressor")
def pipeline_model2(dataset, pipeline, oselm_regressor_model):
    pipeline.steps.append(("regressor", oselm_regressor_model))
    pipeline.fit(dataset.X_train, dataset.y_train)
    yield pipeline


@pytest.fixture
@when("I fit the pipeline and the OSELMClassifier")
def pipeline_model2(dataset, pipeline, oselm_classifier_model):
    assert 1 == 2
    pipeline.steps.append(("classifier", oselm_classifier_model))
    pipeline.fit(dataset.X_train, dataset.y_train)
    yield pipeline


# TODO: train sequentially as in tutorial

@pytest.fixture
@when("I fit the pipeline and the OSELMRegressor in online fashion")
def pipeline_model_online(dataset, pipeline, oselm_regressor_model):
    # First fit the pipeline
    pipeline.fit(dataset.X_train, dataset.y_train)

    X_train = pipeline.transform(dataset.X_train)
    y_train = dataset.y_train

    # Model should be fitted with a batch of at least model.n_hidden rows the first time
    n = oselm_regressor_model.n_hidden
    oselm_regressor_model.fit(X_train[:n], y_train[:n])

    # Now fit the model row by row
    for X, y in zip(X_train[n:], y_train[n:]):
        # Reshape arrays to have a matrix and vector
        X = X.reshape(1, -1)
        y = [y]
        oselm_regressor_model.fit(X, y) 

    # Add the model to the pipeline
    pipeline.steps.append(("regressor", oselm_regressor_model))

    yield pipeline
