"""Steps that are shared between scenarios"""

import pytest
from pytest_bdd import given, when, then, parsers

from .utils.loaders import load_dataset, load_pipeline, load_elm_model, load_oselm_model, Results


@given(parsers.parse("the dataset '{key}'"))
def dataset(key):
    yield load_dataset(key)


@given("a pre-processing pipeline")
def pipeline(dataset):
    key = dataset.name
    yield load_pipeline(key)


@given(parsers.parse("an {model_type} model"))
def model(model_type, dataset):
    key = dataset.name
    loader = load_elm_model if model_type.startswith("ELM") else load_oselm_model
    yield loader(key)


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
    # Thresholds valid for r1 and accuracy score
    assert score_model.train > 0.7
    assert score_model.test > 0.7


@then("the results are good enough")
def results_good_enough(score_model):
    # Thresholds valid for r1 and accuracy score
    assert score_model.train > 0.5
    assert score_model.test > 0.5
