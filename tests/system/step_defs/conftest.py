"""Steps that are shared between scenarios"""

import pytest
from pytest_bdd import given, when, then, parsers

from .utils.loaders import load_dataset, load_pipeline, Results


@given(parsers.parse("the dataset '{key}'"))
def dataset(key):
    yield load_dataset(key)


@given("a pre-processing pipeline")
def pipeline(dataset):
    key = dataset.name
    yield load_pipeline(key)


@pytest.fixture
@when("I fit the pipeline and the model")
def pipeline_model(dataset, pipeline, model):
    pipeline.steps.append(("model", model))
    pipeline.fit(dataset.X_train, dataset.y_train)

    yield pipeline


@pytest.fixture
@then("I compute the score in train and test sets")
def score_model(pipeline_model, dataset):
    score_train = pipeline_model.score(dataset.X_train, dataset.y_train)
    score_test = pipeline_model.score(dataset.X_test, dataset.y_test)

    yield Results(score_train, score_test)


@then("the results are very good")
def results_really_good(score_model):
    assert score_model.train > 0.7
    assert score_model.test > 0.7


@then("the results are good enough")
def results_good_enough(score_model):
    assert score_model.train > 0.5
    assert score_model.test > 0.5

