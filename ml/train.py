#!/usr/bin/env python3
"""Train a baseline spam classifier using LogisticRegression and TF-IDF features.

Usage:
    python ml/train.py --data-url <url> --output models/spam_baseline.joblib
"""
import argparse
import os
import sys
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


DEFAULT_URL = (
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
)


def download_dataset(url: str) -> pd.DataFrame:
    # CSV has no header; first column is label, second is message
    df = pd.read_csv(url, header=None, encoding_errors="ignore")
    if df.shape[1] < 2:
        raise ValueError("Expected at least 2 columns (label, message)")
    df = df.iloc[:, :2]
    df.columns = ["label", "message"]
    return df


def prepare(df: pd.DataFrame):
    df = df.dropna(subset=["message"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    # Normalize labels to 'spam' or 'ham'
    df["label"] = df["label"].map(lambda s: "spam" if s.startswith("spam") or s == "1" else "ham")
    X = df["message"].astype(str).values
    y = df["label"].values
    return X, y


def train(data_url: str, output_path: str):
    print(f"Downloading dataset from {data_url}...")
    df = download_dataset(data_url)
    X, y = prepare(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = make_pipeline(TfidfVectorizer(max_features=10000), LogisticRegression(max_iter=200))
    print("Training logistic regression baseline...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print("Evaluation on test split:")
    print(classification_report(y_test, preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"Saved trained model pipeline to {output_path}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-url", default=DEFAULT_URL, help="CSV URL for dataset")
    parser.add_argument("--output", default="models/spam_baseline.joblib", help="Output path for model")
    args = parser.parse_args(argv)
    train(args.data_url, args.output)


if __name__ == "__main__":
    main()
