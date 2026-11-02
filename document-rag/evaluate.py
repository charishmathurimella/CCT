"""Measure top-1 document retrieval accuracy on labeled synthetic questions."""
import csv
from pathlib import Path

from rag import Retriever

EVALUATION = Path(__file__).resolve().parent / "data" / "evaluation.csv"


def evaluate():
    with EVALUATION.open(newline="", encoding="utf-8") as handle:
        examples = list(csv.DictReader(handle))
    if not examples:
        raise ValueError("No evaluation questions found")
    retriever = Retriever()
    correct = 0
    for row in examples:
        results = retriever.search(row["question"], top_k=1)
        predicted = results[0]["source"] if results else None
        if predicted == row["expected_source"]:
            correct += 1
        else:
            print(f"Mismatch: {row['question']!r}: expected {row['expected_source']}, got {predicted}")
    accuracy = correct / len(examples)
    print(f"Top-1 document retrieval accuracy: {accuracy:.3f} ({correct}/{len(examples)})")
    return accuracy


if __name__ == "__main__":
    evaluate()
