"""Canonical Streamlit app for spam classification (clean replacement).
"""
from pathlib import Path
from typing import Optional, List
import subprocess
import sys

import streamlit as st
import joblib
import pandas as pd


MODEL_DEFAULT = Path("models/spam_baseline.joblib")


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def run_training(output_path: str) -> tuple[int, str, str]:
    cmd = [sys.executable, "ml/train.py", "--output", output_path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def classify(model, text: str) -> tuple[str, Optional[float], Optional[dict]]:
    pred = model.predict([text])[0]
    if hasattr(model, "predict_proba"):
        arr = model.predict_proba([text])[0]
        probs = {c: float(v) for c, v in zip(model.classes_, arr)}
        conf = probs.get(pred)
    else:
        probs = None
        conf = None
    return pred, conf, probs


def ensure_history():
    if "history" not in st.session_state:
        st.session_state.history = []


def main():
    st.set_page_config(page_title="Spam Classifier", layout="wide")
    ensure_history()

    st.title("Spam / Ham Classifier")
    st.write("Enter text or upload CSV (column 'text') to classify messages.")

    sidebar = st.sidebar
    sidebar.header("Model")
    model_path = sidebar.text_input("Model path", value=str(MODEL_DEFAULT))
    model_file = Path(model_path)

    if not model_file.exists():
        sidebar.warning("Model not found.")
        if sidebar.button("Train model now"):
            with st.spinner("Training model..."):
                rc, out, err = run_training(str(model_file))
            st.code(out + ("\n\nERROR:\n" + err if err else ""))
            if rc == 0 and model_file.exists():
                st.success("Model trained and saved.")
                try:
                    load_model.clear()
                except Exception:
                    pass
        uploaded = sidebar.file_uploader("Upload joblib model", type=["joblib"])
        if uploaded is not None:
            models_dir = Path("models")
            models_dir.mkdir(parents=True, exist_ok=True)
            target = models_dir / uploaded.name
            with open(target, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success(f"Saved model to {target}")
            model_file = target

    model = None
    if model_file.exists():
        try:
            model = load_model(str(model_file))
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    left, right = st.columns([2, 1])

    with left:
        st.header("Single message")
        text = st.text_area("Message text", height=160)
        if st.button("Classify"):
            if not model:
                st.error("No model loaded.")
            elif not text or not text.strip():
                st.warning("Please enter text to classify.")
            else:
                pred, conf, probs = classify(model, text)
                color = "red" if str(pred).lower() == "spam" else "green"
                st.markdown(f"### Prediction: <span style='color:{color}'>{pred}</span>", unsafe_allow_html=True)
                if conf is not None:
                    st.metric("Confidence", f"{conf:.1%}")
                if probs is not None:
                    dfp = pd.DataFrame([probs]).T.reset_index()
                    dfp.columns = ["class", "probability"]
                    st.table(dfp)
                st.session_state.history.insert(0, {"text": text, "pred": str(pred), "conf": conf})

        st.markdown("---")
        st.header("Batch (CSV)")
        csv_file = st.file_uploader("Upload CSV (must contain 'text' column)", type=["csv"], key="batch")
        if csv_file is not None:
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                df = None
            if df is not None:
                if "text" not in df.columns:
                    st.error("CSV must contain a 'text' column.")
                elif not model:
                    st.error("Load or train a model first.")
                else:
                    preds = model.predict(df["text"].astype(str).tolist())
                    out = df.copy()
                    out["pred"] = preds
                    if hasattr(model, "predict_proba"):
                        prob_arrays = model.predict_proba(df["text"].astype(str).tolist())
                        out["conf"] = [float(p.max()) for p in prob_arrays]
                    st.dataframe(out.head(200))
                    st.success(f"Classified {len(out)} rows")

    with right:
        st.header("Samples & History")
        st.write("Try sample messages")
        samples: List[str] = [
            "Win a $1000 gift card now! Click here.",
            "Are we still on for lunch tonight?",
        ]
        for s in samples:
            if st.button(s, key=s[:20]):
                # If a model is loaded, classify the sample immediately and show result.
                if model:
                    try:
                        pred, conf, probs = classify(model, s)
                        color = "red" if str(pred).lower() == "spam" else "green"
                        st.markdown(f"**Sample prediction:** <span style='color:{color}'>{pred}</span>", unsafe_allow_html=True)
                        if conf is not None:
                            st.write(f"Confidence: {conf:.1%}")
                        st.session_state.history.insert(0, {"text": s, "pred": str(pred), "conf": conf})
                    except Exception as e:
                        st.error(f"Failed to classify sample: {e}")
                        st.session_state.history.insert(0, {"text": s, "pred": "(error)", "conf": None})
                else:
                    st.warning("No model loaded — click Train or upload a model to get real predictions.")
                    st.session_state.history.insert(0, {"text": s, "pred": "(sample)", "conf": None})

        st.markdown("---")
        st.subheader("Model info")
        st.write(f"Model path: `{model_file}`")
        if model is not None and hasattr(model, "classes_"):
            st.write(f"Classes: {list(model.classes_)}")

        st.markdown("---")
        st.subheader("Prediction history")
        if st.session_state.history:
            hist = pd.DataFrame(st.session_state.history)
            st.table(hist.head(50))
        else:
            st.write("No predictions yet.")


if __name__ == "__main__":
    main()
