## Why
We want to add a spam email/SMS classification capability to explore machine learning integration in the project and provide a demonstrable baseline model for future improvements.

## What Changes
- **ADDED** new capability: `spam-classification` to add a supervised ML pipeline for spam detection.
- Add data ingestion for the provided dataset URL and a small training script to produce a baseline model using logistic regression.
- Add evaluation reporting (precision/recall/F1) and a simple prediction CLI for local testing.

## Impact
- Affected specs: `spam-classification` (new)
- Affected code: `ml/train.py`, `ml/predict.py`, `data/ingest.py`, plus tests under `tests/ml/`
- Breaks: none (additive)
