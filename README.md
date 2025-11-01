# Spam Classification (hw3)

This repository contains a small SMS spam classification demo (TF-IDF + LogisticRegression). It includes training and prediction scripts and a Streamlit web UI.

Demo site
-----------------
- Live demo (deployed): https://9fsq4e88xenv3y9i3d2xxk.streamlit.app/

Quick overview
-----------------
- `ml/train.py` — training script. Produces a joblib model (default path: `models/spam_baseline.joblib`).
- `ml/predict.py` — simple CLI to load the trained model and run predictions.
- `data/ingest.py` — dataset downloader for the SMS spam CSV used in the baseline.
- `app.py` — Streamlit application (single-file) for interactive single/batch prediction, training-on-demand, and model upload.

Requirements
-----------------
Install the Python dependencies (recommended inside a virtualenv):

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Quick start (local)
-----------------
1. Train the baseline model (creates `models/spam_baseline.joblib`):

```powershell
python ml/train.py --output models/spam_baseline.joblib
```

2. Run the Streamlit app locally:

```powershell
streamlit run app.py
```

The app also supports uploading an existing joblib model file and batch predictions via CSV (a `text` column is expected).

Dataset
-----------------
The baseline uses the Packt SMS spam dataset (raw CSV hosted in the original project). See `data/ingest.py` for the ingest URL used by the training script.

Notes
-----------------
- Model artifact `models/spam_baseline.joblib` is not checked into the repo by default. You can train it locally or upload one through the app.
- If you deploy the app (for example to Streamlit Cloud), point the app to the model path or use the app's Train button to create the model on the deployed instance (note: training can be slow on free tiers).

Contributing
-----------------
If you want enhancements (UI polish, evaluation panels, CI checks), open an issue or create a PR. I can add a GitHub Actions workflow to run syntax checks on push if you'd like.

License
-----------------
MIT-style (see repository license or add one).

Contact
-----------------
If you need help running the demo locally or want me to run training and a local smoke test, tell me and I'll run the steps and post logs.
# Spam Classification & Project README

This repository contains an OpenSpec-driven proposal and a minimal implementation for a spam classification baseline (Phase 1). The baseline trains a Logistic Regression model using TF-IDF features and provides a small Streamlit UI for interactive prediction.

## Source Reference
- Dataset and inspiration: Chapter 3 of the Packt repository
  - https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity.git
  - Direct dataset URL used in this project:
    https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv

## Quick setup (Windows PowerShell)
1. Create and activate a virtual environment, install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. (Optional) Download dataset locally:

```powershell
python data/ingest.py --out data/sms_spam_no_header.csv
```

3. Train baseline model (will download dataset if not present):

```powershell
python ml/train.py --output models/spam_baseline.joblib
```

4. Predict a single message via CLI:

```powershell
python ml/predict.py --model models/spam_baseline.joblib --text "Free entry in 2 a wkly comp to win"
```

5. Run Streamlit web UI (recommended for exploration):

```powershell
streamlit run app.py
```

## Preprocess command
- The repository provides `data/ingest.py` to download the raw CSV. The training script (`ml/train.py`) performs minimal preprocessing (drop empty messages, normalize labels to `spam`/`ham`) automatically.

If you'd like an explicit preprocessing step (save cleaned CSV):

```powershell
python - <<'PY'
import pandas as pd
df = pd.read_csv('data/sms_spam_no_header.csv', header=None, encoding_errors='ignore')
df = df.iloc[:, :2]
df.columns = ['label','message']
df = df.dropna(subset=['message'])
df['label'] = df['label'].astype(str).str.strip().str.lower().map(lambda s: 'spam' if s.startswith('spam') or s == '1' else 'ham')
df.to_csv('data/sms_spam_clean.csv', index=False)
print('Saved data/sms_spam_clean.csv')
PY
```

## Visualization & Example Results

The training script prints evaluation metrics (precision, recall, F1) and a confusion matrix. You can also create a confusion matrix plot using matplotlib: below is a suggested script to generate a visualization from your trained model and test split.

```python
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import pandas as pd

# load data
df = pd.read_csv('data/sms_spam_no_header.csv', header=None, encoding_errors='ignore')
df = df.iloc[:, :2]; df.columns=['label','message']
df = df.dropna(subset=['message'])
df['label'] = df['label'].astype(str).str.strip().str.lower().map(lambda s: 'spam' if s.startswith('spam') or s == '1' else 'ham')
X = df['message'].values; y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = joblib.load('models/spam_baseline.joblib')
preds = model.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap='Blues')
plt.title('Confusion Matrix (example)')
plt.savefig('artifacts/confusion_matrix.png')
print('Saved artifacts/confusion_matrix.png')
```

Example output (your run will differ):

```
Evaluation on test split:
              precision    recall  f1-score   support

       ham       0.98      0.99      0.99       965
      spam       0.95      0.91      0.93       140

    accuracy                           0.98      1105
   macro avg       0.97      0.95      0.96      1105
weighted avg       0.98      0.98      0.98      1105

Confusion matrix:
[[955  10]
 [ 13 127]]
```

## Notes & Next steps
- Phase 2 and later phases are intentionally left blank in `openspec/changes/add-spam-classification/tasks.md` for you to fill (e.g., hyperparameter tuning, SVM experiments, model serving, batch prediction).
- See `openspec/changes/add-spam-classification/` for proposal, tasks, and spec deltas.

If you want, I can run the training here and produce the actual confusion matrix and saved artifact, or I can expand Phase 2 with concrete experiments.
