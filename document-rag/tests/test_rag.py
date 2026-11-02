import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import Retriever, load_documents
from evaluate import evaluate


@pytest.fixture(scope="module")
def retriever():
    return Retriever()


def test_documents_loaded():
    docs = load_documents()
    assert len(docs) >= 8
    assert {d["source"] for d in docs} == {
        "account_access.txt", "technical_support.txt", "leave_policy.txt", "payroll.txt"
    }


def test_retrieval_returns_citations(retriever):
    result = retriever.answer("How do I reset my password?", threshold=0)
    assert result["source"] == "account_access.txt"
    assert "password" in result["answer"].lower()


def test_blank_question_abstains(retriever):
    result = retriever.answer(" ")
    assert result["source"] is None


def test_unrelated_question_abstains(retriever):
    result = retriever.answer("Tell me about quantum astrophysics", threshold=0.15)
    assert result["source"] is None


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_threshold(retriever, threshold):
    with pytest.raises(ValueError):
        retriever.answer("hello", threshold)


def test_evaluation_score_range():
    assert 0 <= evaluate() <= 1


def test_search_is_deterministic(retriever):
    first = retriever.search("Where is my pay statement?")
    second = retriever.search("Where is my pay statement?")
    assert first == second


def test_search_returns_requested_number(retriever):
    results = retriever.search("employee support", top_k=2)
    assert len(results) == 2
    assert all(0 <= item["score"] <= 1 for item in results)


def test_invalid_top_k(retriever):
    with pytest.raises(ValueError):
        retriever.search("password", top_k=0)


def test_high_threshold_abstains(retriever):
    result = retriever.answer("reset password", threshold=1)
    assert result["source"] is None
