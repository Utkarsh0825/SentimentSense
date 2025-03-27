import gdown  # <-- Add this at the top

@st.cache_resource
def download_and_load_models():
    if not os.path.exists("bert_sentiment.pt"):
        st.info("📥 Downloading pre-trained models...")
        gdown.download("https://drive.google.com/uc?id=1M_3HzvbPzFOgGXHj8XM0zRbH01tYxLCX", MODEL_ZIP_PATH, quiet=False)

        st.info("🗃️ Extracting models...")
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall()

        os.remove(MODEL_ZIP_PATH)
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import zipfile
import os
import json
import requests

# UI setup
st.set_page_config(page_title="SentimentSense", layout="centered")
st.markdown("<h1 style='text-align: center;'>💬 SentimentSense</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>An AI-powered Emotion and Sentiment Analyzer</h4>", unsafe_allow_html=True)

# 📦 Google Drive zip download config
MODEL_ZIP_URL = "https://drive.google.com/uc?export=download&id=1M_3HzvbPzFOgGXHj8XM0zRbH01tYxLCX"
MODEL_ZIP_PATH = "Archive.zip"

@st.cache_resource
def download_and_load_models():
    if not os.path.exists("bert_sentiment.pt"):
        st.info("📥 Downloading pre-trained models...")
        with open(MODEL_ZIP_PATH, "wb") as f:
            response = requests.get(MODEL_ZIP_URL)
            f.write(response.content)

        st.info("🗃️ Extracting models...")
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall()

        os.remove(MODEL_ZIP_PATH)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    sentiment_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    sentiment_model.load_state_dict(torch.load("bert_sentiment.pt", map_location=torch.device("cpu")))
    sentiment_model.eval()

    emotion_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28)
    emotion_model.load_state_dict(torch.load("bert_emotion.pt", map_location=torch.device("cpu")))
    emotion_model.eval()

    with open("emotion_labels.json", "r") as f:
        emotion_labels = json.load(f)

    return tokenizer, sentiment_model, emotion_model, emotion_labels

# Load models safely
tokenizer, sentiment_model, emotion_model, emotion_labels = download_and_load_models()

# Input
text_input = st.text_area("Enter a sentence to analyze:", height=150)

if st.button("Analyze") and text_input.strip():
    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)

    # Sentiment
    with torch.no_grad():
        sent_outputs = sentiment_model(**inputs)
        sent_logits = sent_outputs.logits
        sent_probs = F.softmax(sent_logits, dim=1)
        sent_label = torch.argmax(sent_probs, dim=1).item()
        sent_conf = float(torch.max(sent_probs).item())

    sentiment_classes = ["Negative", "Neutral", "Very Positive"]
    sentiment_emojis = ["😠", "😐", "😍"]
    sentiment = sentiment_classes[sent_label]
    emoji = sentiment_emojis[sent_label]

    st.success("✅ Analysis Complete")
    st.markdown(f"**Sentiment:** {sentiment} {emoji}")
    st.progress(min(float(sent_conf), 1.0))

    # Emotion
    with torch.no_grad():
        emo_outputs = emotion_model(**inputs)
        emo_logits = emo_outputs.logits
        emo_probs = F.softmax(emo_logits, dim=1)
        top_emo_idx = torch.argmax(emo_probs, dim=1).item()
        emo_label = emotion_labels[str(top_emo_idx)]
        emo_conf = float(torch.max(emo_probs).item())

    st.markdown(f"**Top Emotion:** `{emo_label}`")
    st.progress(min(float(emo_conf), 1.0))
