"""Tests for TextBlob labelling and the inference wrappers (with a fake model)."""
import pytest

import helper

CONFIDENCE = "Probability(Confidence Level)"


class TestGetSentiment:
    @pytest.mark.parametrize(
        "tweet, expected",
        [
            ("I love this, it is great", 1),
            ("This is terrible and awful", -1),
            ("The meeting is on Tuesday", 0),
        ],
    )
    def test_polarity_to_label(self, tweet, expected):
        assert helper.get_sentiment(tweet) == expected

    def test_ignores_mentions_and_links(self):
        assert helper.get_sentiment("@bob https://t.co/x great day") == 1


class TestInference:
    def test_splits_tweets_on_semicolons(self, fake_vectoriser, make_fake_model):
        model = make_fake_model([1, 0, 1], [[0.1, 0.9], [0.7, 0.3], [0.4, 0.6]])
        df = helper.inference(fake_vectoriser, model, "first;second;third", ["tweet"])

        assert list(df["Tweet"]) == ["first", "second", "third"]

    def test_output_columns(self, fake_vectoriser, make_fake_model):
        model = make_fake_model([1], [[0.1, 0.9]])
        df = helper.inference(fake_vectoriser, model, "hello", ["tweet"])

        assert list(df.columns) == ["Tweet", "Sentiment", CONFIDENCE]

    def test_maps_labels_to_words(self, fake_vectoriser, make_fake_model):
        model = make_fake_model([1, 0], [[0.2, 0.8], [0.9, 0.1]])
        df = helper.inference(fake_vectoriser, model, "a;b", ["tweet"])

        assert list(df["Sentiment"]) == ["Positive", "Negative"]

    def test_reports_confidence_of_predicted_class(self, fake_vectoriser, make_fake_model):
        model = make_fake_model([1, 0], [[0.2, 0.8], [0.9, 0.1]])
        df = helper.inference(fake_vectoriser, model, "a;b", ["tweet"])

        assert list(df[CONFIDENCE]) == ["80.0%", "90.0%"]

    def test_rounds_confidence_to_two_decimals(self, fake_vectoriser, make_fake_model):
        model = make_fake_model([1], [[0.12345, 0.87655]])
        df = helper.inference(fake_vectoriser, model, "a", ["tweet"])

        assert df[CONFIDENCE][0] == "87.66%"

    def test_vectoriser_gets_cleaned_lemmatized_text(self, fake_vectoriser, make_fake_model):
        model = make_fake_model([1], [[0.1, 0.9]])
        helper.inference(fake_vectoriser, model, "I LOVE dogs!!!", ["tweet"])

        assert fake_vectoriser.received == ["love dog"]


class TestPredictUrl:
    def test_output_columns(self, fake_vectoriser, make_fake_model):
        model = make_fake_model([0], [[0.9, 0.1]])
        df = helper.predict_url(fake_vectoriser, model, ["hello"])

        assert list(df.columns) == ["tweet", "Sentiment", CONFIDENCE]

    def test_keeps_tweets_and_maps_labels(self, fake_vectoriser, make_fake_model):
        model = make_fake_model([1, 0], [[0.2, 0.8], [0.9, 0.1]])
        df = helper.predict_url(fake_vectoriser, model, ["good", "bad"])

        assert list(df["tweet"]) == ["good", "bad"]
        assert list(df["Sentiment"]) == ["Positive", "Negative"]

    def test_negative_confidence(self, fake_vectoriser, make_fake_model):
        model = make_fake_model([0], [[0.9, 0.1]])
        df = helper.predict_url(fake_vectoriser, model, ["bad"])

        assert df[CONFIDENCE][0] == "90.0%"

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: predict_url always reports the Negative-class probability, "
        "so Positive tweets show a low confidence (e.g. 20% instead of 80%).",
    )
    def test_positive_confidence(self, fake_vectoriser, make_fake_model):
        model = make_fake_model([1], [[0.2, 0.8]])
        df = helper.predict_url(fake_vectoriser, model, ["good"])

        assert df[CONFIDENCE][0] == "80.0%"
