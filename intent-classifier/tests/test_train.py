import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import (
    DATA_PATH, TEST_PATH, build_model, evaluate, load_examples,
    predict_with_fallback, validate_datasets,
)


@pytest.fixture(scope="module")
def datasets():
    train_x, train_y = load_examples(DATA_PATH)
    test_x, test_y = load_examples(TEST_PATH)
    return train_x, train_y, test_x, test_y


@pytest.fixture(scope="module")
def fitted_model(datasets):
    train_x, train_y, _, _ = datasets
    return build_model().fit(train_x, train_y)


def test_dataset_sizes_and_balance(datasets):
    train_x, train_y, test_x, test_y = datasets
    assert len(train_x) == len(train_y) == 180
    assert len(test_x) == len(test_y) == 40
    assert len(set(train_y)) == 4
    validate_datasets(*datasets)


def test_no_exact_train_test_overlap(datasets):
    train_x, _, test_x, _ = datasets
    assert set(x.casefold() for x in train_x).isdisjoint(x.casefold() for x in test_x)


def test_known_training_example(fitted_model):
    label, confidence = predict_with_fallback(fitted_model, "I forgot my account password", 0)
    assert label == "account_access"
    assert 0 <= confidence <= 1


def test_vpn_example(fitted_model):
    label, confidence = predict_with_fallback(fitted_model, "The VPN connection keeps dropping", 0)
    assert label == "technical_support"
    assert 0 <= confidence <= 1


def test_blank_text_falls_back(fitted_model):
    assert predict_with_fallback(fitted_model, "  ") == ("fallback", 0.0)


def test_threshold_one_returns_fallback(fitted_model):
    assert predict_with_fallback(fitted_model, "unrelated words", 1)[0] == "fallback"


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_threshold(fitted_model, threshold):
    with pytest.raises(ValueError):
        predict_with_fallback(fitted_model, "hello", threshold)


def test_evaluate_returns_40_predictions(datasets):
    _, predictions = evaluate(*datasets)
    assert len(predictions) == 40
