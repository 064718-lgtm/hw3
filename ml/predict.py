#!/usr/bin/env python3
"""Load a trained model pipeline and predict a label for a given message.

Usage:
    python ml/predict.py --model models/spam_baseline.joblib --text "Free entry..."
"""
import argparse
import joblib


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/spam_baseline.joblib", help="Path to trained model pipeline")
    parser.add_argument("--text", required=True, help="Message text to classify")
    args = parser.parse_args(argv)

    pipeline = joblib.load(args.model)
    pred = pipeline.predict([args.text])[0]
    probs = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([args.text])[0]
    if probs is not None:
        # assume classes_ ordering
        class_idx = list(pipeline.classes_).index(pred)
        confidence = float(probs[class_idx])
    else:
        confidence = 1.0

    print({"label": str(pred), "confidence": confidence})


if __name__ == "__main__":
    main()
