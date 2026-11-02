"""Train and evaluate a simple text intent classifier on synthetic examples."""

import argparse
import csv
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).parent / "data" / "intents.csv"


def load_examples(path=DATA_PATH):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not row.get("text") or not row.get("intent") for row in rows):
        raise ValueError("Dataset must contain nonempty text and intent columns.")
    return [row["text"] for row in rows], [row["intent"] for row in rows]


def build_model():
    from sklearn.pipeline import FeatureUnion
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("characters", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def predict_with_fallback(model, text, threshold=0.65):
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    probabilities = model.predict_proba([text])[0]
    index = probabilities.argmax()
    confidence = float(probabilities[index])
    label = str(model.classes_[index]) if confidence >= threshold else "fallback"
    return label, confidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Optional utterance to classify after training")
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    texts, labels = load_examples()
    train_x, test_x, train_y, test_y = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    model = build_model()
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    print(f"Accuracy: {accuracy_score(test_y, predictions):.3f}")
    print(f"Macro F1: {f1_score(test_y, predictions, average='macro'):.3f}")
    print(classification_report(test_y, predictions, zero_division=0))
    print("Labels:", list(model.classes_))
    print("Confusion matrix:\n", confusion_matrix(test_y, predictions, labels=model.classes_))
    if args.text:
        label, confidence = predict_with_fallback(model, args.text, args.threshold)
        print(f"Prediction: {label} (confidence: {confidence:.3f})")


if __name__ == "__main__":
    main()
