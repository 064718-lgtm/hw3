# Spam Classification (baseline)

This folder contains a minimal baseline implementation for spam classification (Phase 1) using scikit-learn and Streamlit.

Quick start

1. Create a Python environment and install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Train the baseline model (will download the dataset):

```powershell
python ml/train.py --output models/spam_baseline.joblib
```

3. Use the web interface (recommended):

```powershell
streamlit run app.py
```

Or predict a single message via CLI:

```powershell
python ml/predict.py --model models/spam_baseline.joblib --text "Free entry in 2 a wkly comp to win"
```

Notes
- The dataset URL used is the Packt SMS spam CSV (no header) referenced in the proposal.
- Phase 2/3 placeholders remain empty in `openspec/changes/add-spam-classification/tasks.md` as requested.
- The Streamlit interface provides an easy way to try different messages and see prediction confidence.
