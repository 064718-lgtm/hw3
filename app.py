import sys
import subprocess
from pathlib import Path
from typing import Optional, List

import streamlit as st
import joblib
import pandas as pd


MODEL_DEFAULT_PATH = Path("models/spam_baseline.joblib")


@st.cache_resource
def load_model(model_path: str):
    """Load a joblib model pipeline from disk. Cached by Streamlit to avoid repeated loads."""
    return joblib.load(model_path)


def run_training(model_path: str) -> tuple[int, str, str]:
    """Run the training script as a subprocess and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, "ml/train.py", "--output", model_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def classify_text(model, text: str) -> tuple[str, Optional[float], Optional[dict]]:
    """Return (prediction_label, confidence, probs_dict).

    If predict_proba is not available, confidence is None and probs_dict is None.
    """
    pred = model.predict([text])[0]
    probs = None
    if hasattr(model, "predict_proba"):
        prob_array = model.predict_proba([text])[0]
        probs = {cls: float(p) for cls, p in zip(model.classes_, prob_array)}
        confidence = probs.get(pred, None)
    else:
        confidence = None
    return pred, confidence, probs


def ensure_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []


def main() -> None:
    st.set_page_config(page_title="Spam Classifier", layout="wide")
    ensure_session_state()

    st.title("Spam / Ham Classifier")
    st.write("A small demo that classifies messages as spam or ham using a scikit-learn pipeline.")

    sidebar = st.sidebar
    sidebar.header("Model & Controls")

    model_path = sidebar.text_input("Model path", value=str(MODEL_DEFAULT_PATH))
    model_file = Path(model_path)

    # Training controls if model missing
    if not model_file.exists():
        sidebar.warning(f"Model not found at `{model_path}`.")
        if sidebar.button("Train model now"):
            with st.spinner("Training (runs ml/train.py)..."):
                rc, out, err = run_training(str(model_file))
            st.code(out + ("\n\nERROR:\n" + err if err else ""))
            if rc == 0 and model_file.exists():
                st.success("Training finished and model file created.")
                # clear cache so load_model will reload
                try:
                    load_model.clear()
                except Exception:
                    pass
            else:
                st.error("Training failed — inspect output above.")

        uploaded = sidebar.file_uploader("Or upload an existing model (joblib)", type=["joblib"])
        if uploaded is not None:
            # write to the models directory and update model_path
            models_dir = Path("models")
            models_dir.mkdir(parents=True, exist_ok=True)
            target = models_dir / uploaded.name
            with open(target, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success(f"Saved model to {target}")
            model_file = target

    # Attempt to load model if exists
    model = None
    if model_file.exists():
        try:
            model = load_model(str(model_file))
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    # Main UI columns
    left, right = st.columns([2, 1])

    with left:
        st.header("Single message")
        text = st.text_area("Message text", height=160)
        if st.button("Classify"):
            if not model:
                st.error("No model loaded. Train or provide a model first.")
            elif not text or text.strip() == "":
                st.warning("Please enter some text to classify.")
            else:
                pred, conf, probs = classify_text(model, text)
                color = "red" if pred.lower() == "spam" else "green"
                st.markdown(f"### Prediction: <span style='color:{color}'>{pred}</span>", unsafe_allow_html=True)
                if conf is not None:
                    st.metric("Confidence", f"{conf:.1%}")
                if probs is not None:
                    probs_df = pd.DataFrame([probs]).T.reset_index()
                    probs_df.columns = ["class", "probability"]
                    st.table(probs_df)
                # add to history
                st.session_state.history.insert(0, {"text": text, "pred": pred, "conf": conf})

        st.markdown("---")
        st.header("Batch prediction (CSV)")
        st.write("Upload a CSV with a column named 'text' to classify multiple messages.")
        csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv")
        if csv_file is not None:
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                df = None

            if df is not None:
                if "text" not in df.columns:
                    st.error("CSV must contain a 'text' column.")
                else:
                    if not model:
                        st.error("No model loaded. Train or provide a model first.")
                    else:
                        preds = model.predict(df["text"].astype(str).tolist())
                        result_df = df.copy()
                        result_df["pred"] = preds
                        if hasattr(model, "predict_proba"):
                            prob_arrays = model.predict_proba(df["text"].astype(str).tolist())
                            # attach max probability as confidence
                            result_df["conf"] = [float(p.max()) for p in prob_arrays]
                        st.dataframe(result_df.head(200))
                        st.success(f"Classified {len(result_df)} rows.")

    with right:
        st.header("Samples & Model")
        st.write("Try example messages:")
        samples: List[str] = [
            "Free entry in 2 a weekly competition to win FA Cup final tickets",
            "Hey, are we still meeting for lunch today?",
            "Congratulations! You have won a $1000 Walmart gift card. Click here to claim.",
        ]
        for s in samples:
            if st.button(s, key=s[:20]):
                st.session_state.get("last_sample", None)
                st.experimental_set_query_params()
                st.session_state.history.insert(0, {"text": s, "pred": "(sample)", "conf": None})

        st.markdown("---")
        st.subheader("Model")
        st.write(f"Model path: `{model_file}`")
        if model is not None:
            st.write(f"Model classes: {list(model.classes_)}")

        st.markdown("---")
        st.subheader("Prediction history")
        if st.session_state.history:
            hist_df = pd.DataFrame(st.session_state.history)
            st.table(hist_df.head(50))
        else:
            st.write("No predictions yet.")


if __name__ == "__main__":
    main()
import sys
import subprocess
from pathlib import Path
from typing import Optional, List

import streamlit as st
import joblib
import pandas as pd


MODEL_DEFAULT_PATH = Path("models/spam_baseline.joblib")


@st.cache_resource
def load_model(model_path: str):
    """Load a joblib model pipeline from disk. Cached by Streamlit to avoid repeated loads."""
    return joblib.load(model_path)


def run_training(model_path: str) -> tuple[int, str, str]:
    """Run the training script as a subprocess and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, "ml/train.py", "--output", model_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def classify_text(model, text: str) -> tuple[str, Optional[float], Optional[dict]]:
    """Return (prediction_label, confidence, probs_dict).

    If predict_proba is not available, confidence is None and probs_dict is None.
    """
    pred = model.predict([text])[0]
    probs = None
    if hasattr(model, "predict_proba"):
        prob_array = model.predict_proba([text])[0]
        probs = {cls: float(p) for cls, p in zip(model.classes_, prob_array)}
        confidence = probs.get(pred, None)
    else:
        confidence = None
    return pred, confidence, probs


def ensure_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []


def main() -> None:
    st.set_page_config(page_title="Spam Classifier", layout="wide")
    ensure_session_state()

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
                            """Streamlit web interface for spam classification.

                            This app provides interactive single and batch prediction, training-on-demand,
                            and an evaluation panel when the dataset is available locally.

                            Run with:
                                streamlit run app.py
                            """
                            import io
                            from pathlib import Path
                            from typing import Optional

                            import streamlit as st
                            import joblib
                            import subprocess
                            import sys
                            import pandas as pd
                            import matplotlib.pyplot as plt
                            from sklearn.metrics import classification_report, ConfusionMatrixDisplay


                            MODEL_DEFAULT = "models/spam_baseline.joblib"
                            DATA_DEFAULT = "data/sms_spam_no_header.csv"


                            @st.cache_resource
                            def load_model(model_path: str):
                                return joblib.load(model_path)


                            def run_training(model_path: str):
                                cmd = [sys.executable, "ml/train.py", "--output", model_path]
                                proc = subprocess.run(cmd, capture_output=True, text=True)
                                return proc.returncode, proc.stdout, proc.stderr


                            def evaluate_model_if_possible(model, data_path: str) -> Optional[tuple]:
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
                                        vec = None
                                        if hasattr(model, "named_steps") and "tfidfvectorizer" in model.named_steps:
                                            vec = model.named_steps.get("tfidfvectorizer")
                                        else:
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

                                with left:
                                    st.header("Try it now")
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
                                        uploaded.seek(0)
                                        try:
                                            preview = pd.read_csv(uploaded, nrows=0)
                                            header = 0 if "message" in preview.columns else None
                                            uploaded.seek(0)
                                            df = pd.read_csv(uploaded, header=header, encoding_errors="ignore")
                                        except Exception:
                                            uploaded.seek(0)
                                            df = pd.read_csv(uploaded, header=None, encoding_errors="ignore")

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
                                                confs = [float(p[list(model.classes_).index(preds[i])]) for i, p in enumerate(probs)]
                                                out["confidence"] = confs
                                            st.dataframe(out.head(200))
                                            csv = out.to_csv(index=False).encode("utf-8")
                                            st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv")

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