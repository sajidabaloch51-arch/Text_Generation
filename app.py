
import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

# -------------------------------
# Page configuration
# -------------------------------

st.set_page_config(
    page_title="BERT Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


# -------------------------------
# Load the model once
# -------------------------------

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    device = 0 if torch.cuda.is_available() else -1

    classifier = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    return classifier


# -------------------------------
# Application interface
# -------------------------------

st.title("🎬 BERT Sentiment Analysis")

st.write(
    "Enter a movie review to analyze whether "
    "its sentiment is positive or negative."
)

st.divider()

review = st.text_area(
    "Movie Review",
    placeholder="Example: This movie was amazing!",
    height=180
)

if st.button("Analyze Sentiment", type="primary"):

    if not review.strip():

        st.warning("Please enter a movie review.")

    else:

        try:

            with st.spinner("Analyzing your review..."):

                classifier = load_model()

                result = classifier(
                    review.strip(),
                    truncation=True,
                    max_length=512
                )[0]

            label = result["label"].upper()
            confidence = float(result["score"])

            if label == "POSITIVE":

                st.success("😊 Sentiment: Positive")

            elif label == "NEGATIVE":

                st.error("😞 Sentiment: Negative")

            else:

                st.error(f"Unexpected model label: {label}")

            if label in {"POSITIVE", "NEGATIVE"}:

                st.metric(
                    "Model Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.progress(confidence)

                st.caption(
                    "Confidence is the model's score for its "
                    "predicted class, not a guarantee of correctness."
                )

        except Exception:

            st.error(
                "The model could not analyze this review. "
                "Please try again."
            )

st.divider()

st.caption(
    "Movie Review Sentiment Analysis | "
    "Powered by Hugging Face Transformers"
)
