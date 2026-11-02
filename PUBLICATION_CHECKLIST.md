# Publication readiness checklist

**Current state: PRIVATE. Do not change visibility without explicit owner approval.**

## Required before publishing

- [x] Three independent synthetic portfolio projects committed
- [x] README files describe project scope and limitations
- [x] Automated tests and evaluation scripts included for each project
- [x] Consolidated GitHub Actions workflow added at `.github/workflows/portfolio-release.yml`
- [x] Lightweight credential and required-file audit script added
- [x] Confirm the latest consolidated GitHub Actions run completed successfully (run 41370059339)
- [x] Capture measured validation, retrieval, and regression scores from the run logs
- [ ] Review the **entire git history**, not just current files, for confidential code, credentials, internal screenshots, names, URLs, and documents
- [ ] Manually verify the content is original and approved for sharing
- [ ] Choose and add a license, if open-source reuse is desired
- [ ] Review public-facing contact details and biography for accuracy
- [ ] Obtain explicit owner approval before changing repository visibility

## Run checks locally

From the repository root:

```bash
python -m pip install -r intent-classifier/requirements.txt
python -m pip install -r document-rag/requirements.txt
python -m pip install -r conversation-regression/requirements.txt
python scripts/publication_audit.py
python -m pytest intent-classifier/tests -q
python intent-classifier/validate.py
python intent-classifier/train.py
python -m pytest document-rag/tests -q
python document-rag/evaluate.py
python -m pytest conversation-regression/tests -q
python conversation-regression/run_regression.py
```

## Claims and interpretation

- Intent classifier's previously reported test macro F1 is **0.948** on 40 synthetic test utterances. This rounds to **0.95** at two decimal places, but does **not** meet a strict 0.950 threshold.
- The classifier training dataset contains templated variations and does not represent 180 independent real-world language patterns.
- The document QA project is an **extractive TF-IDF retrieval baseline**, not an LLM-backed generative RAG system.
- The conversational regression project is a **rule-based router and test harness**, not a learned intent classifier.
- New evaluation results must be measured from actual runs before being described as verified.
- Synthetic evaluation results should never be presented as production accuracy.

## Owner approval

When all checks above are complete, request an explicit go-ahead to make the repository public. Publishing is a separate action, not implied by preparing it.

## Verified CI results (2026-11-01, run 41370059339)

- Publication audit: passed (pattern-based scan only)
- Intent classifier: 11 tests passed; validation accuracy 0.975 and macro F1 0.975 (40 synthetic examples); held-out test accuracy 0.950 and macro F1 0.948 (40 synthetic examples)
- Document QA: 11 tests passed; top-1 source retrieval 12/12 on synthetic evaluation questions
- Conversational regression: 13 tests passed; intent 35/35 and entity exact-match 35/35 on authored synthetic cases

These are small, synthetic, curated benchmarks and are not production performance claims. The manual history, IP, contact-details, and licensing review remains open.
