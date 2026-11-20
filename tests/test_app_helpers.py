"""Tests for the Streamlit/IO helpers: plot, progressbar, convert_df,
read_markdown_file and twint_to_pandas. Streamlit calls are replaced with fakes."""
from types import SimpleNamespace

import pandas as pd

import helper
from conftest import ROOT


class TestPlot:
    def _draw(self, monkeypatch, sentiments):
        shown = []
        monkeypatch.setattr(helper.st, "pyplot", shown.append)
        helper.plot(pd.DataFrame({"Sentiment": sentiments}))
        assert len(shown) == 1
        return shown[0].axes[0]

    def test_pie_shows_percentages(self, monkeypatch):
        ax = self._draw(monkeypatch, ["Positive", "Positive", "Positive", "Negative"])
        texts = [t.get_text() for t in ax.texts]

        assert "Positive" in texts and "Negative" in texts
        assert "75.00%" in texts and "25.00%" in texts

    def test_title(self, monkeypatch):
        ax = self._draw(monkeypatch, ["Positive", "Negative"])

        assert ax.get_title() == "Positive vs Negative Tweet(%)"


class TestProgressbar:
    def test_fills_from_0_to_100(self, monkeypatch):
        calls = []

        class FakeBar:
            def progress(self, value):
                calls.append(value)

        def fake_progress(value):
            calls.append(value)
            return FakeBar()

        monkeypatch.setattr(helper.st, "progress", fake_progress)
        monkeypatch.setattr(helper.time, "sleep", lambda _seconds: None)

        helper.progressbar()

        assert calls == list(range(0, 101))


class TestConvertDf:
    def test_returns_utf8_csv_with_index(self):
        df = pd.DataFrame({"tweet": ["héllo"], "Sentiment": ["Positive"]})

        result = helper.convert_df(df)

        assert isinstance(result, bytes)
        assert result.decode("utf-8").splitlines() == [
            ",tweet,Sentiment",
            "0,héllo,Positive",
        ]


class TestReadMarkdownFile:
    def test_reads_file_contents(self, tmp_path):
        path = tmp_path / "about.md"
        path.write_text("# Title\n\nBody text")

        assert helper.read_markdown_file(path) == "# Title\n\nBody text"

    def test_project_about_page_exists(self):
        assert "monkeypox" in helper.read_markdown_file(ROOT / "about.md").lower()


class TestTwintToPandas:
    def test_selects_requested_columns(self, monkeypatch):
        tweets = pd.DataFrame(
            {"date": ["2022-08-27"], "username": ["who"], "tweet": ["hi"], "id": [1]}
        )
        fake_twint = SimpleNamespace(
            output=SimpleNamespace(panda=SimpleNamespace(Tweets_df=tweets))
        )
        monkeypatch.setattr(helper, "twint", fake_twint)

        df = helper.twint_to_pandas(["date", "username", "tweet"])

        assert list(df.columns) == ["date", "username", "tweet"]
        assert df.iloc[0].tolist() == ["2022-08-27", "who", "hi"]
