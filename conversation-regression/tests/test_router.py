import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from router import extract_entities, route
from run_regression import evaluate


@pytest.mark.parametrize(("utterance", "intent"), [
    ("My password expired", "account_access"),
    ("Unlock my account please", "account_access"),
    ("Unlock the account", "account_access"),
    ("My laptop is broken", "technical_support"),
    ("Can I use PTO", "leave_policy"),
    ("My paycheck is late", "payroll"),
    ("My VPN and paycheck are not working", "clarification"),
    ("Tell me a joke", "fallback"),
    ("", "fallback"),
])
def test_intent_routing(utterance, intent):
    assert route(utterance)["intent"] == intent


def test_entity_extraction():
    assert extract_entities("Ticket IT-1234: I need 3 days off") == {
        "ticket_id": "IT-1234",
        "days": 3,
    }


def test_entities_not_invented():
    assert extract_entities("I need time off") == {}


def test_ambiguous_request_reports_all_matches():
    result = route("My VPN and paycheck have issues")
    assert result["intent"] == "clarification"
    assert result["matched_intents"] == ["payroll", "technical_support"]


def test_regression_suite():
    intent_correct, entity_correct, total = evaluate()
    assert total >= 30
    assert intent_correct == total
    assert entity_correct == total
