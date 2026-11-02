"""Evaluate the classifier on a development-only validation dataset."""
from collections import Counter

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from train import DATA_PATH, ROOT, build_model, load_examples

VALIDATION_PATH = ROOT / "data" / "validation_intents.csv"


def run_validation():
    train_x, train_y = load_examples(DATA_PATH)
    valid_x, valid_y = load_examples(VALIDATION_PATH)
    if len(train_x) != 180 or len(valid_x) != 40:
        raise ValueError("Expected 180 training and 40 validation utterances")
    if any(n != 10 for n in Counter(valid_y).values()):
        raise ValueError("Expected 10 validation utterances per intent")
    if set(train_y) != set(valid_y):
        raise ValueError("Train and validation labels differ")
    if set(x.casefold() for x in train_x) & set(x.casefold() for x in valid_x):
        raise ValueError("Training and validation utterances overlap")
    model = build_model().fit(train_x, train_y)
    predictions = model.predict(valid_x)
    accuracy = accuracy_score(valid_y, predictions)
    macro_f1 = f1_score(valid_y, predictions, average="macro")
    print(f"Validation accuracy: {accuracy:.3f}")
    print(f"Validation macro F1: {macro_f1:.3f}")
    print(classification_report(valid_y, predictions, zero_division=0))
    print("Labels:", list(model.classes_))
    print("Confusion matrix:\n", confusion_matrix(valid_y, predictions, labels=model.classes_))
    for utterance, expected, predicted in zip(valid_x, valid_y, predictions):
        if expected != predicted:
            print(f"Error: {utterance!r}; expected={expected}; predicted={predicted}")
    return accuracy, macro_f1


if __name__ == "__main__":
    run_validation()
