"""Tests for text cleaning: preprocess, lemmatize_process, clean_tweet, POS mapping."""
import pytest
from nltk.corpus import wordnet

import helper


def run(text):
    """Preprocess a single tweet."""
    return helper.preprocess([text])[0]


class TestPreprocess:
    def test_returns_one_result_per_input(self):
        assert len(helper.preprocess(["one", "two", "three"])) == 3

    def test_empty_string(self):
        assert run("") == ""

    def test_lowercases_text(self):
        assert run("HELLO World") == "hello world "

    @pytest.mark.parametrize(
        "url", ["http://example.com", "https://t.co/abc123"]
    )
    def test_replaces_urls(self, url):
        assert run(f"vaccine {url}") == "vaccine URL "

    def test_replaces_www_links(self):
        assert run("visit www.who.int") == "visit URL "

    def test_replaces_user_mentions(self):
        assert run("@WHO update please") == "USER update please "

    @pytest.mark.parametrize(
        "emoticon, meaning",
        [(":)", "smile"), (":(", "sad"), (";)", "wink"), (":-(", "sad")],
    )
    def test_replaces_lowercase_emoticons(self, emoticon, meaning):
        assert run(f"news {emoticon}") == f"news EMOJI{meaning} "

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: text is lowercased before emoticon replacement, so "
        "uppercase keys such as ':D' and ':P' never match and are dropped.",
    )
    @pytest.mark.parametrize("emoticon, meaning", [(":D", "smile"), (":P", "raspberry")])
    def test_replaces_uppercase_emoticons(self, emoticon, meaning):
        assert run(f"news {emoticon}") == f"news EMOJI{meaning} "

    def test_collapses_repeated_letters_to_two(self):
        assert run("soooo goooood") == "soo good "

    def test_removes_digits_and_punctuation(self):
        assert run("covid19 is #bad!!!") == "covid bad "

    def test_removes_stopwords_and_single_letters(self):
        assert run("i am a b c happy") == "happy "

    def test_keeps_negations(self):
        # The custom stopword list deliberately keeps negations like "not".
        assert run("not good") == "not good "


class TestGetWordnetPosTag:
    @pytest.mark.parametrize(
        "tag, expected",
        [
            ("JJ", wordnet.ADJ),
            ("JJS", wordnet.ADJ),
            ("VB", wordnet.VERB),
            ("VBG", wordnet.VERB),
            ("NN", wordnet.NOUN),
            ("NNS", wordnet.NOUN),
            ("RB", wordnet.ADV),
            ("DT", wordnet.NOUN),  # anything unknown falls back to noun
            ("", wordnet.NOUN),
        ],
    )
    def test_maps_treebank_tags(self, tag, expected):
        assert helper.get_wordnet_pos_tag(tag) == expected


class TestLemmatizeProcess:
    def test_lemmatizes_with_part_of_speech(self):
        assert helper.lemmatize_process(["running dogs better"]) == ["run dog well"]

    def test_plural_nouns(self):
        assert helper.lemmatize_process(["cases"]) == ["case"]

    def test_empty_string(self):
        assert helper.lemmatize_process([""]) == [""]

    def test_returns_one_result_per_input(self):
        assert len(helper.lemmatize_process(["a cat", "two dogs"])) == 2


class TestCleanTweet:
    def test_removes_mentions_links_and_symbols(self):
        tweet = "@bob Check this out! https://t.co/abc #monkeypox"
        assert helper.clean_tweet(tweet) == "Check this out monkeypox"

    def test_keeps_case_and_numbers(self):
        assert helper.clean_tweet("COVID-19 cases") == "COVID 19 cases"

    def test_collapses_whitespace(self):
        assert helper.clean_tweet("  lots   of\tspace  ") == "lots of space"

    def test_empty_string(self):
        assert helper.clean_tweet("") == ""
