# Validation: revised baseline

Local run on 2026-11-01 using Python, scikit-learn 1.8.0.

- Dataset: 40 synthetic utterances, 10 for each of four intents
- Evaluation: stratified 5-fold cross-validation with out-of-fold predictions (random_state=42)
- Accuracy: **0.575**
- Macro F1: **0.569**
- VPN example: `technical_support`, confidence **0.543**, threshold **0.4**

Confusion matrix, with rows actual and columns predicted:

```text
labels: account_access, leave_policy, payroll, technical_support
[[6 1 0 3]
 [1 8 0 1]
 [0 4 4 2]
 [3 1 1 5]]
```

## Limitations

This remains a small synthetic baseline with significant classification errors. Five-fold cross-validation uses more evaluation examples than the original 10-example holdout, but it is not an independent external test. The previous 40% holdout accuracy and the new 57.5% cross-validation accuracy use different evaluation protocols and should not be presented as a direct improvement measurement. The fallback threshold has not been calibrated.

Next steps: add varied labeled examples, reserve a never-seen test set, evaluate out-of-scope requests, and tune threshold based on coverage and error rate.
