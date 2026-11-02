# Intent classifier evaluation

The most recent verified results come from the successful [portfolio GitHub Actions run](https://github.com/charishmathurimella/CCT/actions/runs/41370059339) on November 1, 2026.

## Dataset

- Training: 180 synthetic utterances, 45 per intent
- Validation: 40 synthetic utterances, 10 per intent
- Test: 40 synthetic utterances, 10 per intent
- Labels: `account_access`, `technical_support`, `leave_policy`, `payroll`

The model uses word and character TF-IDF features with class-balanced logistic regression. The training set contains repeated sentence structures, so the number of rows should not be mistaken for the number of independent language patterns.

## Results

| Split | Accuracy | Macro F1 |
|---|---:|---:|
| Validation | 0.975 | 0.975 |
| Test | 0.950 (38/40) | 0.948 |

Eleven unit tests passed in the same CI run.

The two test errors involved payroll requests being classified as account access or technical support. They are useful examples of why shared terms such as “account” and payment-related wording need careful coverage.

## Reproduce

```bash
python -m pip install -r requirements.txt
python validate.py
python train.py
python -m pytest -q
```

## Interpretation

These are small, synthetic datasets. The test set should remain untouched during further tuning. For a stronger performance claim, collect a new independent test set, add ambiguous and out-of-domain queries, and assess confidence calibration.
