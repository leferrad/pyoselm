"""Steps for elm.feature"""

import pytest
from pytest_bdd import scenarios, given, when

from .utils.loaders import load_elm_model


scenarios("../features/elm.feature")


@given("an ELMRegressor model")
def elm_regressor_model(dataset):
    key = dataset.name
    yield load_elm_model(key)


@pytest.fixture
@when("I fit the pipeline and the ELMRegressor")
def pipeline_model(dataset, pipeline, elm_regressor_model):
    pipeline.steps.append(("regressor", elm_regressor_model))
    pipeline.fit(dataset.X_train, dataset.y_train)
    yield pipeline
