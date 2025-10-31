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