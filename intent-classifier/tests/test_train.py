import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import build_model, load_examples, predict_with_fallback


def test_dataset_has_multiple_examples_per_class():
    texts, labels = load_examples()
    assert len(texts) == len(labels)
    assert len(set(labels)) >= 3
    assert all(labels.count(label) >= 4 for label in set(labels))


def test_model_predicts_known_training_example():
    texts, labels = load_examples()
    model = build_model().fit(texts, labels)
    predicted, confidence = predict_with_fallback(model, "I forgot my password", 0)
    assert predicted in set(labels)
    assert 0 <= confidence <= 1


def test_threshold_one_returns_fallback():
    texts, labels = load_examples()
    model = build_model().fit(texts, labels)
    predicted, _ = predict_with_fallback(model, "unrelated words", 1)
    assert predicted == "fallback"


def test_invalid_threshold():
    texts, labels = load_examples()
    model = build_model().fit(texts, labels)
    with pytest.raises(ValueError):
        predict_with_fallback(model, "hello", -0.1)
