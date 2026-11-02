# Intent Classification Baseline

A small, reproducible NLP baseline for routing employee-support questions to the right category. All example utterances are synthetic and unrelated to any employer's internal data.

## Approach

The classifier uses word and character TF-IDF features with logistic regression. A stratified holdout split provides an initial estimate of generalization. The script reports accuracy, macro F1, a classification report, and a confusion matrix. Predictions below a configurable confidence threshold are sent to a fallback route.

This is a baseline, not a production-ready system. The included dataset is deliberately small; results on it should not be interpreted as real-world performance.

## Run

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
python train.py --text "I cannot connect to the VPN" --threshold 0.65
pytest -q
```

## Next improvements

Expand the dataset with consented or synthetic examples, add out-of-scope utterances, tune thresholds against validation data, and evaluate performance on an independently collected test set.
