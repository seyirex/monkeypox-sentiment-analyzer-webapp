"""Tests that use the real pickled model and vectoriser from model/."""
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import helper


@pytest.fixture(scope="module")
def model(real_model_and_vectoriser):
    return real_model_and_vectoriser[0]


@pytest.fixture(scope="module")
def vectoriser(real_model_and_vectoriser):
    return real_model_and_vectoriser[1]


def test_model_is_binary_linear_svc(model):
    assert isinstance(model, LinearSVC)
    assert list(model.classes_) == [0, 1]


def test_vectoriser_is_fitted_tfidf(vectoriser):
    assert isinstance(vectoriser, TfidfVectorizer)
    assert vectoriser.ngram_range == (1, 2)
    assert len(vectoriser.vocabulary_) > 0


def test_vectoriser_and_model_dimensions_match(model, vectoriser):
    assert model.coef_.shape[1] == len(vectoriser.vocabulary_)


def test_inference_end_to_end(model, vectoriser):
    df = helper.inference(
        vectoriser,
        model,
        "I love this, it is great;This is terrible and awful",
        ["tweet"],
    )

    assert list(df["Sentiment"]) == ["Positive", "Negative"]
    for value in df["Probability(Confidence Level)"]:
        assert value.endswith("%")
        assert 50.0 <= float(value.rstrip("%")) <= 100.0


def test_predict_url_end_to_end(model, vectoriser):
    df = helper.predict_url(
        vectoriser, model, ["I love this, it is great", "This is terrible and awful"]
    )

    assert list(df["Sentiment"]) == ["Positive", "Negative"]
