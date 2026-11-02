# Validation protocol

The current dataset contains **180 synthetic training utterances** (45 per intent) and **40 separate synthetic test utterances** (10 per intent).

Run:

```bash
python train.py
pytest -q
```

The script trains a TF-IDF and logistic regression pipeline on `data/intents.csv` and reports accuracy, macro F1, a classification report, and a confusion matrix on `data/test_intents.csv`. It checks the expected dataset sizes, per-intent balance, and exact train/test duplicates.

**Do not interpret the earlier 57.5% cross-validation score as the result for this dataset.** That score was measured on the former 40-example dataset. No new measured performance score is recorded here until the updated code has been executed.

The examples are synthetic and relatively templated. This test is useful for a reproducible demonstration, but a more challenging independently authored evaluation set is needed before claiming deployment readiness.
