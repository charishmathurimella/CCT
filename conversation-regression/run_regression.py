"""Run a labeled regression suite and print failures."""
import csv
import json
from pathlib import Path

from router import route

CASES = Path(__file__).resolve().parent / "data" / "regression_cases.csv"


def evaluate(path=CASES):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No regression cases found")
    intent_correct = 0
    entity_correct = 0
    for index, row in enumerate(rows, 1):
        actual = route(row["utterance"])
        expected_entities = json.loads(row["expected_entities"])
        intent_ok = actual["intent"] == row["expected_intent"]
        entity_ok = actual["entities"] == expected_entities
        intent_correct += int(intent_ok)
        entity_correct += int(entity_ok)
        if not (intent_ok and entity_ok):
            print(f"Case {index} failed: {row['utterance']!r}")
            print(f"  expected: {row['expected_intent']} / {expected_entities}")
            print(f"  actual:   {actual['intent']} / {actual['entities']}")
    print(f"Intent accuracy: {intent_correct}/{len(rows)} ({intent_correct/len(rows):.3f})")
    print(f"Entity exact-match accuracy: {entity_correct}/{len(rows)} ({entity_correct/len(rows):.3f})")
    return intent_correct, entity_correct, len(rows)


if __name__ == "__main__":
    intent_correct, entity_correct, total = evaluate()
    if intent_correct != total or entity_correct != total:
        raise SystemExit(1)
