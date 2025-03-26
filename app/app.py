import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import zipfile
import os
import requests

# Title and subtitle
st.set_page_config(page_title="SentimentSense", layout="centered", initial_sidebar_state="auto")
st.markdown("<h1 style='text-align: center;'>💬 SentimentSense</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>An AI-powered Emotion and Sentiment Analyzer</h4>", unsafe_allow_html=True)

# === DOWNLOAD MODELS ZIP IF NOT PRESENT ===
MODEL_ZIP_URL = "https://drive.google.com/uc?export=download&id=1M_3HzvbPzFOgGXHj8XM0zRbH01tYxLCX"
MODEL_ZIP_PATH = "models.zip"
MODEL_DIR = "models"

def download_and_extract_models():
    if not os.path.exists(MODEL_DIR):
        st.info("📦 Downloading pre-trained models...")
        with open(MODEL_ZIP_PATH, "wb") as f:
            response = requests.get(MODEL_ZIP_URL)
            f.write(response.content)

        st.info("🗃️ Extracting models...")
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)

        os.remove(MODEL_ZIP_PATH)
        st.success("✅ Models ready!")

@st.cache_resource
def load_models():
    download_and_extract_models()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    sentiment_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
    sentiment_model.load_state_dict(torch.load("models/bert_sentiment.pt", map_location=torch.device("cpu")))
    sentiment_model.eval()

    emotion_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28)
    emotion_model.load_state_dict(torch.load("models/bert_emotion.pt", map_location=torch.device("cpu")))
    emotion_model.eval()

    # Labels
    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
        'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
        'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]

    return tokenizer, sentiment_model, emotion_model, emotion_labels

tokenizer, sentiment_model, emotion_model, emotion_labels = load_models()

# === Analyze Text ===
text_input = st.text_area("Enter a sentence to analyze:", height=150)

if st.button("Analyze"):
    if text_input.strip():
        st.success("✅ Analysis Complete")

        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)

        # Sentiment Prediction
        with torch.no_grad():
            sentiment_outputs = sentiment_model(**inputs)
            sentiment_probs = F.softmax(sentiment_outputs.logits, dim=1)[0]
            sentiment_score = sentiment_probs.argmax().item()
            sentiment_conf = float(sentiment_probs[sentiment_score])

        sentiment_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
        sentiment_emojis = ["😠", "🙁", "😐", "🙂", "😍"]

        st.markdown(f"### Sentiment: {sentiment_labels[sentiment_score]} {sentiment_emojis[sentiment_score]}")
        st.progress(min(float(sentiment_conf), 1.0))

        # Emotion Prediction
        with torch.no_grad():
            emotion_outputs = emotion_model(**inputs)
            emotion_probs = F.softmax(emotion_outputs.logits, dim=1)[0]
            top_emotion = torch.argmax(emotion_probs).item()
            top_emotion_conf = float(emotion_probs[top_emotion])

        st.markdown(f"### Emotion: {emotion_labels[top_emotion].capitalize()} 🎭")
        st.progress(min(float(top_emotion_conf), 1.0))
    else:
        st.warning("⚠️ Please enter some text.")
