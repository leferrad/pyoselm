"""Steps for oselm.feature"""

import pytest
from pytest_bdd import scenarios, when

scenarios("../features/oselm.feature")


def pipeline_model_online_chunk(dataset, pipeline, model, chunk_size):
    # First fit the pipeline
    pipeline.fit(dataset.X_train, dataset.y_train)

    # Get data
    X_train = pipeline.transform(dataset.X_train)
    y_train = dataset.y_train

    # Model should be fitted with a batch of at least model.n_hidden rows the first time
    n = model.n_hidden
    model.fit(X_train[:n], y_train[:n])
    N = len(y_train)

    # Now fit the model chunk-by-chunk
    for i in range(n, N, chunk_size):
        X = X_train[i:i+chunk_size]
        y = y_train[i:i+chunk_size]
        model.fit(X, y)

    # Add the model to the pipeline
    pipeline.steps.append(("model", model))

    return pipeline


@pytest.fixture
@when("I fit the pipeline and the model in online fashion, chunk by chunk,")
def pipeline_model(dataset, pipeline, model):
    yield pipeline_model_online_chunk(dataset, pipeline, model, chunk_size=100)


@pytest.fixture
@when("I fit the pipeline and the model in online fashion, row by row,")
def pipeline_model(dataset, pipeline, model):
    yield pipeline_model_online_chunk(dataset, pipeline, model, chunk_size=1)
