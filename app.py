"""Streamlit web interface for spam classification.

Run with:
    streamlit run app.py
"""
import streamlit as st
import joblib
from pathlib import Path


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


def main():
    st.title("Spam Email Classification")
    st.write("Enter a message to check if it's spam or ham (not spam).")

    # Load the model (cached)
    model_path = "models/spam_baseline.joblib"
    if not Path(model_path).exists():
        st.error(f"Model not found at {model_path}. Please train the model first:\n```\npython ml/train.py\n```")
        return
    
    model = load_model(model_path)
    
    # Text input
    message = st.text_area("Message text:", height=100)
    
    if st.button("Classify"):
        if not message:
            st.warning("Please enter some text to classify.")
            return
            
        # Get prediction and probability
        pred = model.predict([message])[0]
        probs = model.predict_proba([message])[0]
        
        # Get confidence for the predicted class
        class_idx = list(model.classes_).index(pred)
        confidence = float(probs[class_idx])
        
        # Show result with confidence
        result_color = "red" if pred == "spam" else "green"
        st.markdown(f"### Prediction: <span style='color:{result_color}'>{pred}</span>", unsafe_allow_html=True)
        st.progress(confidence)
        st.write(f"Confidence: {confidence:.1%}")
        
        # Show probabilities for both classes
        st.write("\nProbabilities:")
        for cls, prob in zip(model.classes_, probs):
            st.write(f"- {cls}: {prob:.1%}")


if __name__ == "__main__":
    main()