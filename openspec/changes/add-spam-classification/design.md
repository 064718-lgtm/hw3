## Context
We will create a lightweight ML pipeline runnable locally for grading. The baseline uses logistic regression via scikit-learn; data will be fetched from the provided GitHub raw URL.

### Data
- Source: https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
- Format: CSV with two columns: label and message (no header)

### Model
- Baseline model: scikit-learn LogisticRegression with TF-IDF features (sklearn.feature_extraction.text.TfidfVectorizer).
- Persist the model with `joblib` into `models/spam_baseline.joblib`.

### Interfaces
- CLI: `python ml/train.py --data-url <url> --output models/spam_baseline.joblib`
- CLI: `python ml/predict.py --model models/spam_baseline.joblib --text "message text"`

### Constraints
- Keep runtime low and dependencies minimal. Use CPU-only scikit-learn.
