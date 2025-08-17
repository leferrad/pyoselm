"""Steps that are shared between scenarios"""

import pytest
from pytest_bdd import given, parsers, then, when

from .utils.loaders import (
    Results,
    load_dataset,
    load_elm_model,
    load_oselm_model,
    load_pipeline,
)

# Global variables to store state between steps
_test_context = {}


@pytest.fixture(autouse=True)
def clear_test_context():
    """Clear test context before each test"""
    _test_context.clear()
    yield
    _test_context.clear()


@given(parsers.parse("the dataset '{key}'"))
def step_dataset(key):
    """Load dataset and store it in global context"""
    dataset = load_dataset(key)
    _test_context["dataset"] = dataset
    return dataset


@pytest.fixture
def dataset():
    """Dataset fixture that retrieves from global context"""
    return _test_context.get("dataset")


@given("a pre-processing pipeline")
def step_pipeline():
    """Load pipeline based on dataset"""
    dataset = _test_context.get("dataset")
    if not dataset:
        raise ValueError("Dataset must be loaded first")

    key = dataset.name
    pipeline = load_pipeline(key)
    _test_context["pipeline"] = pipeline
    return pipeline


@pytest.fixture
def pipeline():
    """Pipeline fixture that retrieves from global context"""
    return _test_context.get("pipeline")


@given(parsers.parse("an {model_type} model"))
def step_model(model_type):
    """Load model based on dataset and model type"""
    dataset = _test_context.get("dataset")
    if not dataset:
        raise ValueError("Dataset must be loaded first")

    key = dataset.name
    loader = load_elm_model if model_type.startswith("ELM") else load_oselm_model
    model = loader(key)
    _test_context["model"] = model
    return model


@pytest.fixture
def model():
    """Model fixture that retrieves from global context"""
    return _test_context.get("model")


@when("I fit the pipeline and the model")
def step_fit_pipeline_model():
    """Fit the pipeline and model"""
    dataset = _test_context.get("dataset")
    pipeline = _test_context.get("pipeline")
    model = _test_context.get("model")

    if not all([dataset, pipeline, model]):
        raise ValueError("Dataset, pipeline, and model must be loaded first")

    pipeline.steps.append(("model", model))
    pipeline.fit(dataset.X_train, dataset.y_train)
    _test_context["pipeline_model"] = pipeline
    return pipeline


@when("I fit the pipeline and the model in online fashion, row by row,")
def step_fit_pipeline_model_online_row():
    """Fit the pipeline and model in online fashion, row by row"""
    dataset = _test_context.get("dataset")
    pipeline = _test_context.get("pipeline")
    model = _test_context.get("model")

    if not all([dataset, pipeline, model]):
        raise ValueError("Dataset, pipeline, and model must be loaded first")

    # For online learning, we need to fit the preprocessing pipeline first
    # then fit the model incrementally
    from sklearn.pipeline import Pipeline

    # First fit the preprocessing pipeline on all data
    pipeline.fit(dataset.X_train, dataset.y_train)

    # Transform data with fitted preprocessing pipeline
    X_transformed = pipeline.transform(dataset.X_train)

    # For OS-ELM, the first batch must have at least n_hidden samples
    n_hidden = model.n_hidden

    # First fit with n_hidden samples
    model.fit(X_transformed[:n_hidden], dataset.y_train[:n_hidden])

    # Then fit the remaining samples row by row
    for i in range(n_hidden, len(X_transformed)):
        X_row = X_transformed[i : i + 1]
        y_row = dataset.y_train[i : i + 1]
        model.fit(X_row, y_row)

    # Create final pipeline with fitted model
    final_pipeline = Pipeline(pipeline.steps + [("model", model)])
    _test_context["pipeline_model"] = final_pipeline
    return final_pipeline


@when("I fit the pipeline and the model in online fashion, chunk by chunk,")
def step_fit_pipeline_model_online_chunk():
    """Fit the pipeline and model in online fashion, chunk by chunk"""
    dataset = _test_context.get("dataset")
    pipeline = _test_context.get("pipeline")
    model = _test_context.get("model")

    if not all([dataset, pipeline, model]):
        raise ValueError("Dataset, pipeline, and model must be loaded first")

    # For online learning, we need to fit the preprocessing pipeline first
    # then fit the model incrementally
    from sklearn.pipeline import Pipeline

    # First fit the preprocessing pipeline on all data
    pipeline.fit(dataset.X_train, dataset.y_train)

    # Transform data with fitted preprocessing pipeline
    X_transformed = pipeline.transform(dataset.X_train)

    # For OS-ELM, the first batch must have at least n_hidden samples
    n_hidden = model.n_hidden
    chunk_size = 10

    # First fit with n_hidden samples
    model.fit(X_transformed[:n_hidden], dataset.y_train[:n_hidden])

    # Then fit the remaining samples chunk by chunk
    for i in range(n_hidden, len(X_transformed), chunk_size):
        X_chunk = X_transformed[i : i + chunk_size]
        y_chunk = dataset.y_train[i : i + chunk_size]
        model.fit(X_chunk, y_chunk)

    # Create final pipeline with fitted model
    final_pipeline = Pipeline(pipeline.steps + [("model", model)])
    _test_context["pipeline_model"] = final_pipeline
    return final_pipeline


@pytest.fixture
def pipeline_model():
    """Pipeline model fixture that retrieves from global context"""
    return _test_context.get("pipeline_model")


@then("I compute the score in train and test sets")
def step_compute_scores():
    """Compute scores on train and test sets"""
    dataset = _test_context.get("dataset")
    pipeline_model = _test_context.get("pipeline_model")

    assert all(
        [dataset, pipeline_model]
    ), "Dataset and fitted pipeline model must be available"

    score_train = pipeline_model.score(dataset.X_train, dataset.y_train)
    score_test = pipeline_model.score(dataset.X_test, dataset.y_test)
    scores = Results(score_train, score_test)
    _test_context["score_model"] = scores
    return scores


@pytest.fixture
def score_model():
    """Score model fixture that retrieves from global context"""
    return _test_context.get("score_model")


@then("the results are very good")
def results_really_good(score_model):
    # Thresholds valid for r1 and accuracy score
    assert score_model.train > 0.6
    assert score_model.test > 0.6


@then("the results are good enough")
def results_good_enough(score_model):
    # Thresholds valid for r1 and accuracy score
    assert score_model.train > 0.5
    assert score_model.test > 0.5
