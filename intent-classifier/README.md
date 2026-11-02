# Intent Classification Baseline

A reproducible NLP baseline for routing employee-support questions to four intent categories. The 40 example utterances are synthetic and contain no employer data.

## Method

- Word and character TF-IDF features, combined in a scikit-learn pipeline
- Class-balanced logistic regression
- Stratified five-fold out-of-fold evaluation, rather than a single small holdout
- Confidence-based fallback and explicit handling of empty input
- Dataset integrity checks and automated tests

## Run

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
python train.py --text "I cannot connect to the VPN" --threshold 0.4
pytest -q
```

## Measured baseline

On the 40 synthetic examples, five-fold out-of-fold accuracy is **0.575** and macro F1 is **0.569** (scikit-learn 1.8.0). The example VPN query routes to `technical_support` at confidence **0.543** when the model is trained on all 40 examples.

These results are not comparable one-to-one with the earlier 10-example holdout because the evaluation protocol changed. They are not production-level results. The confidence threshold is a demonstration setting, not a calibrated decision boundary. See [VALIDATION.md](VALIDATION.md).

## Before deployment

Expand the dataset with diverse examples, create a genuinely untouched test set, evaluate out-of-domain queries, tune fallback thresholds against held-out data, and assess per-intent errors.
