# Conversational AI Regression Testing

A reproducible, privacy-safe test harness for intent routing, entity extraction, ambiguous requests, and fallback behavior. This is an **independent synthetic example**, not a reproduction of any employer's virtual assistant.

## Capabilities

- Deterministic rule-based intent routing with explicit priority and ambiguity handling
- Extract PTO day counts and support ticket IDs from example utterances
- Refuse unsupported or unclear queries through a fallback label
- Execute a version-controlled CSV regression suite
- Report exact-match intent accuracy, entity accuracy, and failed test cases

The router is a **rule-based baseline**, not a trained NLU model. It is designed to demonstrate regression testing and routing design. It can later be replaced with a statistical classifier or agent-based router.

## Run

```bash
pip install -r requirements.txt
python router.py "How many PTO days do I have?"
python run_regression.py
pytest -q
```

## Behavior and limitations

A request that simultaneously matches multiple domains returns `clarification`, rather than silently selecting one. Empty or out-of-scope requests return `fallback`. Entities are extracted only from supported formats. The dataset is synthetic and not a production benchmark. Report metrics only after executing the scripts.
