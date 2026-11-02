"""Deterministic synthetic employee-support intent router and entity extractor."""
import argparse
import json
import re

INTENTS = ("account_access", "technical_support", "leave_policy", "payroll")

PATTERNS = {
    "account_access": (
        r"\\b(?:password|passcode|login|log\\s?in|sign\\s?in|credentials|authentication|unlock account|account locked)\\b",
    ),
    "technical_support": (
        r"\\b(?:vpn|printer|laptop|wifi|wi-fi|webcam|monitor|software|microphone|computer|headset)\\b",
    ),
    "leave_policy": (
        r"\\b(?:pto|vacation|time off|sick leave|annual leave|parental leave|holiday|leave balance)\\b",
    ),
    "payroll": (
        r"\\b(?:paycheck|payslip|pay stub|paystub|salary|payroll|direct deposit|wages|overtime|tax withholding)\\b",
    ),
}


def extract_entities(text):
    """Return only explicitly supported entities, not inferred facts."""
    entities = {}
    ticket = re.search(r"\\b(?:ticket|case)\\s*#?\\s*([A-Za-z]{2,5}-?\\d{3,8})\\b", text, re.I)
    if ticket:
        entities["ticket_id"] = ticket.group(1).upper()
    days = re.search(r"\\b(\\d{1,2})\\s+(?:days?|business days?)\\b", text, re.I)
    if days:
        entities["days"] = int(days.group(1))
    return entities


def route(text):
    if not isinstance(text, str) or not text.strip():
        return {"intent": "fallback", "entities": {}, "matched_intents": []}
    matched = sorted(
        intent for intent, patterns in PATTERNS.items()
        if any(re.search(pattern, text, flags=re.I) for pattern in patterns)
    )
    if len(matched) > 1:
        intent = "clarification"
    elif len(matched) == 1:
        intent = matched[0]
    else:
        intent = "fallback"
    return {"intent": intent, "entities": extract_entities(text), "matched_intents": matched}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("utterance")
    args = parser.parse_args()
    print(json.dumps(route(args.utterance), indent=2))


if __name__ == "__main__":
    main()
