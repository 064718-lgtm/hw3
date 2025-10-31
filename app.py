"""Streamlit web interface for spam classification.

Run with:
    streamlit run app.py
"""
import streamlit as st
import joblib
import subprocess
import sys
from pathlib import Path


@st.cache_resource
def load_model(model_path: str):
    """Load a joblib model pipeline from disk. Cached by Streamlit to avoid repeated loads."""
    return joblib.load(model_path)


def run_training(model_path: str):
    """Run the training script as a subprocess and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, "ml/train.py", "--output", model_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def main():
    st.title("Spam Email Classification")
    st.write("Enter a message to check if it's spam or ham (not spam).")

    model_path = "models/spam_baseline.joblib"
    model_file = Path(model_path)

    if not model_file.exists():
        st.warning(f"Model not found at `{model_path}`. You can train it from the app or run `python ml/train.py` locally.")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Train model now"):
                with st.spinner("Training model (this may take a minute)..."):
                    rc, out, err = run_training(model_path)
                st.code(out + ("\n\nERROR:\n" + err if err else ""))
                if rc == 0 and model_file.exists():
                    st.success("Training completed successfully.")
                    # Clear cached resources so the new model is loaded on next call
                    try:
                        st.cache_resource.clear()
                    except Exception:
                        pass
                else:
                    st.error("Training failed. See output above.")

        with col2:
            custom = st.text_input("Or provide an existing model path:", value=str(model_path))
            if st.button("Load model from path"):
                if Path(custom).exists():
                    try:
                        model = load_model(custom)
                        st.success(f"Loaded model from {custom}")
                    except Exception as e:
                        """Streamlit web interface for spam classification.

                        This enhanced UI matches the example layout: sidebar with model info,
                        sample messages, batch CSV upload, prediction history, and evaluation panel.

                        Run with:
                            streamlit run app.py
                        """
                        import io
                        import streamlit as st
                        import joblib
                        import subprocess
                        import sys
                        from pathlib import Path
                        from typing import Optional
                        import pandas as pd
                        from sklearn.metrics import classification_report, ConfusionMatrixDisplay
                        import matplotlib.pyplot as plt


                        MODEL_DEFAULT = "models/spam_baseline.joblib"
                        DATA_DEFAULT = "data/sms_spam_no_header.csv"


                        @st.cache_resource
                        def load_model(model_path: str):
                            return joblib.load(model_path)


                        def run_training(model_path: str):
                            cmd = [sys.executable, "ml/train.py", "--output", model_path]
                            proc = subprocess.run(cmd, capture_output=True, text=True)
                            return proc.returncode, proc.stdout, proc.stderr


                        def evaluate_model_if_possible(model, data_path: str) -> Optional[str]:
                            """If dataset exists, evaluate model on a test split and return classification report text and save confusion matrix to buffer."""
                            p = Path(data_path)
                            if not p.exists():
                                return None
                            try:
                                df = pd.read_csv(p, header=None, encoding_errors="ignore")
                                df = df.iloc[:, :2]
                                df.columns = ["label", "message"]
                                df = df.dropna(subset=["message"])
                                df["label"] = df["label"].astype(str).str.strip().str.lower().map(lambda s: "spam" if s.startswith("spam") or s == "1" else "ham")
                                X = df["message"].values
                                y = df["label"].values
                                from sklearn.model_selection import train_test_split

                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                                preds = model.predict(X_test)
                                report = classification_report(y_test, preds)

                                # Confusion matrix plot
                                fig, ax = plt.subplots(figsize=(4, 4))
                                ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax, cmap="Blues")
                                buf = io.BytesIO()
                                fig.tight_layout()
                                fig.savefig(buf, format="png")
                                buf.seek(0)
                                return report, buf
                            except Exception:
                                return None


                        def sidebar_model_info(model_path: str):
                            st.sidebar.header("Model & Data")
                            st.sidebar.write(f"Model path: `{model_path}`")
                            if Path(model_path).exists():
                                try:
                                    model = load_model(model_path)
                                    st.sidebar.success("Model loaded")
                                    # Try to show some metadata
                                    if hasattr(model, "named_steps") and "tfidfvectorizer" in model.named_steps:
                                        vec = model.named_steps.get("tfidfvectorizer")
                                    else:
                                        # try to find a TfidfVectorizer in pipeline
                                        vec = None
                                        for v in getattr(model, "steps", []) if hasattr(model, "steps") else []:
                                            if "tfidf" in v[0].lower():
                                                vec = v[1]
                                                break
                                    if vec is not None and hasattr(vec, "vocabulary_"):
                                        st.sidebar.write(f"Vocab size: {len(vec.vocabulary_)}")
                                    if hasattr(model, "classes_"):
                                        st.sidebar.write(f"Classes: {list(model.classes_)}")
                                except Exception as e:
                                    st.sidebar.error(f"Failed to load model: {e}")
                            else:
                                st.sidebar.warning("Model file not found. Use Train button on main page.")


                        def main():
                            st.set_page_config(page_title="Spam Email Classification", layout="wide")
                            st.title("📧 Spam Email Classification")

                            sidebar_model_info(MODEL_DEFAULT)

                            left, right = st.columns([2, 1])

                            # Left column: input, batch upload, and results
                            with left:
                                st.header("Try it now")

                                # sample messages
                                samples = [
                                    "Free entry in 2 a wkly comp to win! Call now",
                                    "Hey, are we still meeting tomorrow at 10?",
                                    "URGENT! Your account has been compromised. Reply with your password",
                                ]
                                example = st.selectbox("Example messages", ["(none)"] + samples)
                                text = st.text_area("Message text", value=(example if example != "(none)" else ""), height=120)

                                col_run, col_clear = st.columns([1, 1])
                                with col_run:
                                    if st.button("Classify"):
                                        if not text.strip():
                                            st.warning("Please enter a message to classify.")
                                        else:
                                            if not Path(MODEL_DEFAULT).exists():
                                                st.error("Model not found. Please train the model first or use the Train button in the sidebar.")
                                            else:
                                                model = load_model(MODEL_DEFAULT)
                                                pred = model.predict([text])[0]
                                                probs = model.predict_proba([text])[0] if hasattr(model, "predict_proba") else None
                                                if probs is not None:
                                                    prob = float(probs[list(model.classes_).index(pred)])
                                                else:
                                                    prob = 1.0
                                                st.metric("Prediction", pred.upper(), delta=f"{prob:.1%} confidence")
                                                # record history
                                                history = st.session_state.get("history", [])
                                                history.insert(0, {"text": text, "pred": pred, "conf": prob})
                                                st.session_state["history"] = history[:50]

                                with col_clear:
                                    if st.button("Clear history"):
                                        st.session_state["history"] = []

                                st.markdown("---")
                                st.header("Batch prediction (CSV)")
                                st.write("Upload a CSV with one column of text (no header) or a column named 'message'.")
                                uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
                                if uploaded is not None:
                                    try:
                                        df = pd.read_csv(uploaded, header=0 if "message" in pd.read_csv(uploaded, nrows=0).columns else None, encoding_errors="ignore")
                                    except Exception:
                                        uploaded.seek(0)
                                        df = pd.read_csv(uploaded, header=None, encoding_errors="ignore")
                                    # normalize
                                    if df.shape[1] == 1:
                                        messages = df.iloc[:, 0].astype(str).tolist()
                                    elif "message" in df.columns:
                                        messages = df["message"].astype(str).tolist()
                                    else:
                                        messages = df.iloc[:, 0].astype(str).tolist()

                                    if not Path(MODEL_DEFAULT).exists():
                                        st.error("Model not found. Train the model first to run batch predictions.")
                                    else:
                                        model = load_model(MODEL_DEFAULT)
                                        preds = model.predict(messages)
                                        probs = model.predict_proba(messages) if hasattr(model, "predict_proba") else None
                                        out = pd.DataFrame({"message": messages, "prediction": preds})
                                        if probs is not None:
                                            # attach confidence
                                            confs = [float(p[list(model.classes_).index(preds[i])]) for i, p in enumerate(probs)]
                                            out["confidence"] = confs
                                        st.dataframe(out.head(200))
                                        # provide download
                                        csv = out.to_csv(index=False).encode("utf-8")
                                        st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv")

                            # Right column: model operations and history
                            with right:
                                st.header("Model & Controls")
                                if not Path(MODEL_DEFAULT).exists():
                                    if st.button("Train model now"):
                                        with st.spinner("Training model..."):
                                            rc, out, err = run_training(MODEL_DEFAULT)
                                        st.code(out + ("\n\nERROR:\n" + err if err else ""))
                                        if rc == 0 and Path(MODEL_DEFAULT).exists():
                                            st.success("Training completed. Reload the app to use the model.")
                                else:
                                    st.success("Model is available")
                                    if st.button("Show evaluation on dataset"):
                                        eval_res = evaluate_model_if_possible(load_model(MODEL_DEFAULT), DATA_DEFAULT)
                                        if eval_res is None:
                                            st.warning("Dataset not found locally. Run `python data/ingest.py` to download it or run training first.")
                                        else:
                                            report, buf = eval_res
                                            st.subheader("Classification report")
                                            st.text(report)
                                            st.subheader("Confusion matrix")
                                            st.image(buf)

                                st.markdown("---")
                                st.header("Prediction history")
                                history = st.session_state.get("history", [])
                                if history:
                                    st.table(pd.DataFrame(history))
                                else:
                                    st.write("No predictions yet.")


                        if __name__ == "__main__":
                            main()
                        st.error(f"Failed to load model: {e}")
                else:
                    st.error("Provided path does not exist.")

        return

    # If model exists, load it and show the classifier UI
    try:
        model = load_model(str(model_file))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    message = st.text_area("Message text:", height=100)

    if st.button("Classify"):
        if not message:
            st.warning("Please enter some text to classify.")
            return

        pred = model.predict([message])[0]
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([message])[0]

        if probs is not None:
            class_idx = list(model.classes_).index(pred)
            confidence = float(probs[class_idx])
        else:
            confidence = 1.0

        result_color = "red" if pred == "spam" else "green"
        st.markdown(f"### Prediction: <span style='color:{result_color}'>{pred}</span>", unsafe_allow_html=True)
        st.progress(confidence)
        st.write(f"Confidence: {confidence:.1%}")

        st.write("\nProbabilities:")
        if probs is not None:
            for cls, prob in zip(model.classes_, probs):
                st.write(f"- {cls}: {prob:.1%}")
        else:
            st.write("Probabilities not available for this model.")


if __name__ == "__main__":
    main()