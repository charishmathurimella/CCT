# Intent Classification Baseline

A reproducible intent classifier for four employee-support categories using **180 synthetic training examples** and **40 separate synthetic test examples**. No employer data is included.

## Data

| Intent | Train | Test |
|---|---:|---:|
| Account access | 45 | 10 |
| Technical support | 45 | 10 |
| Leave policy | 45 | 10 |
| Payroll | 45 | 10 |
| **Total** | **180** | **40** |

Training utterances are generated from 15 seed sentences per intent, each with three wording variants. This creates considerable linguistic similarity within training groups. The test set uses different sentences, but both sets are synthetic and authored together. It is not an external benchmark.

## Approach

- Word and character TF-IDF features
- Class-balanced logistic regression
- Fit only on the training CSV; report accuracy, macro F1, and a confusion matrix on the separate test CSV
- Reject exact duplicates across the datasets
- Confidence-based fallback for individual predictions

## Run

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
python train.py --text "My laptop will not start"
pytest -q
```

## Limitations

The test dataset must not be used for repeated model selection or tuning. A separate validation split and an untouched external test set will be needed for defensible performance claims. A target of F1 = 1.00 is not a substitute for genuine generalization. Confidence values are not calibrated.
