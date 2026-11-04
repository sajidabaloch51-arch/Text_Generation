
# BERT Sentiment Analysis

## Project Overview

This project uses a pretrained DistilBERT model to classify
movie reviews as positive or negative.

## Features

- Movie review input
- Sentiment prediction
- Confidence score
- Interactive Streamlit interface

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Streamlit

## Model

distilbert-base-uncased-finetuned-sst-2-english

## Run Locally

Install the required packages:

pip install -r requirements.txt

Start the application:

streamlit run app.py

## Limitations

Predictions can be incorrect. Confidence scores do not
guarantee that a prediction is correct.
