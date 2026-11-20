"""Shared test setup.

* Puts the project root on ``sys.path`` so ``import helper`` works.
* Stubs out ``twint`` when it isn't installed. twint scrapes Twitter, needs the
  network and no longer works against X, so unit tests never touch it.
* Uses a non-interactive Matplotlib backend.
* Makes sure the NLTK data that ``helper`` needs is available.
"""
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

try:  # pragma: no cover - depends on the environment
    import twint  # noqa: F401
except Exception:  # twint missing or broken on import
    sys.modules["twint"] = types.ModuleType("twint")

import nltk  # noqa: E402

# (resource path used by nltk.data.find, package name for nltk.download)
_NLTK_RESOURCES = [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("corpora/wordnet", "wordnet"),
]
for resource, package in _NLTK_RESOURCES:
    try:
        nltk.data.find(resource)
    except LookupError:  # pragma: no cover - only runs on a fresh machine
        nltk.download(package, quiet=True)


@pytest.fixture(autouse=True)
def _close_figures():
    """Stop Matplotlib figures leaking between tests."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


class FakeVectoriser:
    """Stands in for the TF-IDF vectoriser and records what it was given."""

    def __init__(self):
        self.received = None

    def transform(self, texts):
        self.received = list(texts)
        return self.received


class FakeModel:
    """Stands in for LinearSVC with fixed predictions and probabilities."""

    def __init__(self, predictions, probabilities):
        self._predictions = predictions
        self._probabilities = probabilities

    def predict(self, _features):
        import numpy as np

        return np.array(self._predictions)

    def _predict_proba_lr(self, _features):
        import numpy as np

        return np.array(self._probabilities)


@pytest.fixture
def fake_vectoriser():
    return FakeVectoriser()


@pytest.fixture
def make_fake_model():
    return FakeModel


@pytest.fixture(scope="session")
def real_model_and_vectoriser():
    """Load the pickled model and vectoriser shipped in ``model/``.

    ``helper`` opens them with paths relative to the project root, so the
    working directory is switched there while loading.
    """
    import helper

    previous = os.getcwd()
    os.chdir(ROOT)
    try:
        return helper.load_models(), helper.load_vectoriser()
    finally:
        os.chdir(previous)
