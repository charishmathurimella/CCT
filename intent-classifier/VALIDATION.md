# Baseline validation

Run date: 2026-11-01

Environment: Python with scikit-learn 1.8.0 and pytest 9.0.2.

Commands executed from the `intent-classifier` directory:

```bash
pytest -q
python train.py --text "I cannot connect to the VPN" --threshold 0.65
```

## Results

- Automated tests: **4 passed**
- Holdout split: **30 training examples, 10 test examples** (stratified, random state 42)
- Holdout accuracy: **0.400**
- Holdout macro F1: **0.408**
- Example prediction: **fallback**, confidence **0.514**, at a threshold of **0.65**

Confusion matrix (rows actual, columns predicted; labels in this order: account_access, leave_policy, payroll, technical_support):

```text
[[1 0 0 2]
 [1 1 0 0]
 [0 1 1 1]
 [0 1 0 1]]
```

## Interpretation

The program runs and the included unit tests pass, but classification quality is poor on this very small synthetic holdout set. The unit tests primarily verify functionality and do not establish prediction quality. A ten-example test set is too small for reliable performance estimates. Confidence scores from logistic regression are not calibrated for deployment; the 0.65 fallback threshold has not been tuned.

## Next steps

1. Expand and diversify synthetic or appropriately licensed examples.
2. Reserve an untouched test set and use cross-validation on training data for model selection.
3. Add out-of-scope examples and test routing between overlapping intents.
4. Tune fallback thresholds on validation data and evaluate coverage versus error rate.
5. Add explicit model quality tests before considering deployment.

These results are from local execution of the repository code, not a hosted GitHub CI workflow.
