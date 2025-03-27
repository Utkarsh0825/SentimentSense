import streamlit as st
import gdown
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import os
import zipfile

# UI setup
st.set_page_config(page_title="SentimentSense", layout="centered")
st.markdown("<h1 style='text-align: center;'>💬 SentimentSense</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>An AI-powered Emotion and Sentiment Analyzer</h4>", unsafe_allow_html=True)

# Constants
MODEL_ZIP_PATH = "model.zip"
MODEL_DIR = "models"
GOOGLE_DRIVE_FILE_ID = "1M_3HzvbPzFOgGXHj8XM0zRbH01tYxLCX"  # 👈 replace with your ID

@st.cache_resource
def download_and_load_models():
    if not os.path.exists(MODEL_DIR):
        st.info("📥 Downloading pre-trained models...")
        gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=MODEL_ZIP_PATH, quiet=False)

        st.info("🗃️ Extracting models...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)

        os.remove(MODEL_ZIP_PATH)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    sentiment_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
    sentiment_model.load_state_dict(torch.load(f"{MODEL_DIR}/bert_sentiment.pt", map_location=torch.device("cpu")))
    sentiment_model.eval()

    emotion_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28)
    emotion_model.load_state_dict(torch.load(f"{MODEL_DIR}/bert_emotion.pt", map_location=torch.device("cpu")))
    emotion_model.eval()

    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
        'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
        'remorse', 'sadness', 'surprise', 'neutral'
    ]

    return tokenizer, sentiment_model, emotion_model, emotion_labels


# Load models
tokenizer, sentiment_model, emotion_model, emotion_labels = download_and_load_models()

# Text Input
text_input = st.text_area("Enter a sentence to analyze:", height=150)

if st.button("Analyze") and text_input.strip():
    st.success("✅ Analysis Complete")

    inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
    
    # Sentiment
    sent_outputs = sentiment_model(**inputs)
    sent_scores = torch.nn.functional.softmax(sent_outputs.logits, dim=1)
    sent_label = torch.argmax(sent_scores).item()
    sent_conf = float(sent_scores[0][sent_label])

    sentiment_map = {
        0: ("Very Negative", "😠"),
        1: ("Negative", "🙁"),
        2: ("Neutral", "😐"),
        3: ("Positive", "🙂"),
        4: ("Very Positive", "😍"),
    }

    label, emoji = sentiment_map[sent_label]
    st.markdown(f"### Sentiment: **{label} {emoji}**")
    st.progress(min(float(sent_conf), 1.0))

    # Emotion
    emo_outputs = emotion_model(**inputs)
    emo_scores = torch.nn.functional.softmax(emo_outputs.logits, dim=1)
    emo_label = torch.argmax(emo_scores).item()
    emo_conf = float(emo_scores[0][emo_label])

    st.markdown(f"### Emotion: **{emotion_labels[emo_label]}**")
    st.progress(min(float(emo_conf), 1.0))
