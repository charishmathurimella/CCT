"""Train an intent classifier and evaluate it on a separate test dataset."""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "intents.csv"
TEST_PATH = ROOT / "data" / "test_intents.csv"


def load_examples(path=DATA_PATH):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not row.get("text", "").strip() or not row.get("intent", "").strip() for row in rows):
        raise ValueError("Dataset must contain nonempty text and intent columns")
    texts = [row["text"].strip() for row in rows]
    labels = [row["intent"].strip() for row in rows]
    if len(set(text.casefold() for text in texts)) != len(texts):
        raise ValueError("Duplicate utterances detected")
    return texts, labels


def validate_datasets(train_x, train_y, test_x, test_y):
    if len(train_x) != 180 or len(test_x) != 40:
        raise ValueError("Expected exactly 180 training and 40 test examples")
    if set(train_y) != set(test_y):
        raise ValueError("Train and test intent labels must match")
    if set(train_x_i.casefold() for train_x_i in train_x) & set(test_x_i.casefold() for test_x_i in test_x):
        raise ValueError("Training and test sets must not contain identical utterances")
    if any(count != 45 for count in Counter(train_y).values()):
        raise ValueError("Expected 45 training examples per intent")
    if any(count != 10 for count in Counter(test_y).values()):
        raise ValueError("Expected 10 test examples per intent")


def build_model():
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])


def predict_with_fallback(model, text, threshold=0.4):
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    if not isinstance(text, str) or not text.strip():
        return "fallback", 0.0
    probabilities = model.predict_proba([text])[0]
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    return (str(model.classes_[index]) if confidence >= threshold else "fallback"), confidence


def evaluate(train_x, train_y, test_x, test_y):
    model = build_model()
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    print("Independent test accuracy:", round(accuracy_score(test_y, predictions), 3))
    print("Independent test macro F1:", round(f1_score(test_y, predictions, average="macro"), 3))
    print(classification_report(test_y, predictions, zero_division=0))
    print("Labels:", list(model.classes_))
    print("Confusion matrix:\n", confusion_matrix(test_y, predictions, labels=model.classes_))
    return model, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Classify a sample after training")
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    train_x, train_y = load_examples(DATA_PATH)
    test_x, test_y = load_examples(TEST_PATH)
    validate_datasets(train_x, train_y, test_x, test_y)
    model, _ = evaluate(train_x, train_y, test_x, test_y)
    if args.text is not None:
        label, confidence = predict_with_fallback(model, args.text, args.threshold)
        print(f"Prediction: {label} (confidence: {confidence:.3f})")


if __name__ == "__main__":
    main()
