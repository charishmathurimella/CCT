import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import build_model, load_examples, predict_with_fallback, evaluate


@pytest.fixture(scope="module")
def fitted_model():
    texts, labels = load_examples()
    return build_model().fit(texts, labels)


def test_dataset_integrity():
    texts, labels = load_examples()
    assert len(texts) == len(labels) == 40
    assert len(set(labels)) == 4


def test_known_training_example(fitted_model):
    label, confidence = predict_with_fallback(fitted_model, "I forgot my password", 0)
    assert label == "account_access"
    assert 0 <= confidence <= 1


def test_vpn_example(fitted_model):
    label, confidence = predict_with_fallback(fitted_model, "I cannot connect to the VPN", 0.4)
    assert label == "technical_support"
    assert confidence >= 0.4


def test_blank_text_falls_back(fitted_model):
    assert predict_with_fallback(fitted_model, "  ") == ("fallback", 0.0)


def test_threshold_one_returns_fallback(fitted_model):
    assert predict_with_fallback(fitted_model, "unrelated words", 1)[0] == "fallback"


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_threshold(fitted_model, threshold):
    with pytest.raises(ValueError):
        predict_with_fallback(fitted_model, "hello", threshold)


def test_cross_validation_returns_all_predictions():
    texts, labels = load_examples()
    assert len(evaluate(texts, labels)) == len(labels)
