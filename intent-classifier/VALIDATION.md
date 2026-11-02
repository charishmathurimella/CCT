# Validation results: 180 training / 40 test

Run on 2026-11-01 using Python, scikit-learn 1.8.0, and pytest 9.0.2.

## Dataset and method

- Training: **180** synthetic examples, 45 per intent
- Testing: **40** separate synthetic examples, 10 per intent
- Model: combined word/character TF-IDF and class-balanced logistic regression
- The model was fitted on the training set only. No test labels were used for fitting.
- Dataset checks confirmed no exact text duplicates across the two sets.

## Measured results

- Independent test accuracy: **0.950** (38/40 correct)
- Independent test macro F1: **0.948**
- Automated tests: **9 passed**

| Intent | Precision | Recall | F1 | Test examples |
|---|---:|---:|---:|---:|
| account_access | 0.91 | 1.00 | 0.95 | 10 |
| leave_policy | 1.00 | 1.00 | 1.00 | 10 |
| payroll | 1.00 | 0.80 | 0.89 | 10 |
| technical_support | 0.91 | 1.00 | 0.95 | 10 |

Confusion matrix, rows actual and columns predicted:

```text
labels: account_access, leave_policy, payroll, technical_support
[[10  0  0  0]
 [ 0 10  0  0]
 [ 1  0  8  1]
 [ 0  0  0 10]]
```

Incorrect predictions:

- "My wages were deposited into the wrong account": expected payroll, predicted account_access
- "What date will this month's wages arrive": expected payroll, predicted technical_support

Sample training utterance: "My laptop will not start" -> technical_support, confidence 0.843, using the default threshold 0.4.

## Reproduce

```bash
pip install -r requirements.txt
python train.py --text "My laptop will not start"
pytest -q
```

## Limitations and next steps

All utterances are synthetic. Training sentences include near-duplicate suffix variations, so the 180 rows are not 180 independent language patterns. The test sentences were separately written, but the set is small and was authored for the same task. Avoid tuning repeatedly against this test set; create a separate validation set for improvements, then evaluate once on a newly collected untouched test set. Add ambiguous, out-of-scope, and realistic examples, and calibrate the fallback threshold. A macro F1 of 1.0 cannot be promised on genuinely unseen data.
