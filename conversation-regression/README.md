# Conversation regression tests

This project checks how a simple employee-support router handles ordinary requests, overlapping topics, and unsupported questions.

I kept the router rule-based so that every decision can be inspected. It is a baseline for testing routing behavior, not a replacement for a trained NLU model.

## What it covers

- Four support intents: account access, technical support, leave policy, and payroll
- Ticket IDs and explicitly mentioned numbers of leave days
- Clarification when more than one intent matches
- Fallback when no intent matches
- A CSV regression suite that prints mismatches and exits with an error if a case fails

For example, “Unlock my account please” should route to account access. That phrase originally failed and was added as a regression case after a GitHub Actions run exposed the gap.

## Run it

```bash
python -m pip install -r requirements.txt
python router.py "Unlock my account please"
python run_regression.py
python -m pytest -q
```

## Verified checks

The November 1, 2026 GitHub Actions run passed **13 unit tests** and all **35 authored regression cases** for intent and entity exact matches.

Because the cases were written for this router, a perfect regression result means the implemented behavior matches those expectations. It does not imply perfect classification on unseen language.

## Tradeoffs

The router favors explicit, understandable rules. That makes failures easy to debug, but synonyms, paraphrases, and context across turns remain limited. A future version could compare this baseline with a statistical classifier while keeping the same regression cases.
