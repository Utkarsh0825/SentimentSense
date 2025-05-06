import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import gdown
import os

# -----------------------------
# CONFIG
# -----------------------------
SENTIMENT_ID = "10p14bu-kCjJzha5NkSl3tyzBRDPk85Qm"
EMOTION_ID = "1TkUrX-E16zF5lXm19khJbB9rgQ8PmsZq"

SENTIMENT_MODEL_PATH = "bert_sentiment.pt"
EMOTION_MODEL_PATH = "bert_emotion.pt"

# -----------------------------
# UI Setup
# -----------------------------
st.set_page_config(page_title="SentimentSense", layout="centered")
st.markdown("<h1 style='text-align: center;'>💬 SentimentSense</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>An AI-powered Emotion and Sentiment Analyzer</h4>", unsafe_allow_html=True)

# -----------------------------
# Model Loader
# -----------------------------
@st.cache_resource
def download_and_load_models():
    try:
        if not os.path.exists(SENTIMENT_MODEL_PATH):
            st.info("📥 Downloading Sentiment Model...")
            gdown.download(id=SENTIMENT_ID, output=SENTIMENT_MODEL_PATH, quiet=False, use_cookies=True)

        if not os.path.exists(EMOTION_MODEL_PATH):
            st.info("📥 Downloading Emotion Model...")
            gdown.download(id=EMOTION_ID, output=EMOTION_MODEL_PATH, quiet=False, use_cookies=True)
    except Exception as e:
        st.error("❌ Failed to download model files from Google Drive. "
                 "Please ensure they are shared publicly and accessible without login.")
        st.stop()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    sentiment_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    sentiment_model.load_state_dict(torch.load(SENTIMENT_MODEL_PATH, map_location=torch.device("cpu")))
    sentiment_model.eval()

    emotion_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28)
    emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=torch.device("cpu")))
    emotion_model.eval()

    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
        'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
        'remorse', 'sadness', 'surprise', 'neutral'
    ]

    return tokenizer, sentiment_model, emotion_model, emotion_labels

# -----------------------------
# Load Models
# -----------------------------
tokenizer, sentiment_model, emotion_model, emotion_labels = download_and_load_models()

# -----------------------------
# Input
# -----------------------------
text_input = st.text_area("Enter a sentence to analyze:", height=150)

if st.button("Analyze") and text_input.strip() != "":
    st.success("✅ Analysis Complete")

    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        sent_outputs = sentiment_model(**inputs)
        emotion_outputs = emotion_model(**inputs)

    sent_probs = F.softmax(sent_outputs.logits, dim=-1)[0]
    sent_conf, sent_label_idx = torch.max(sent_probs, dim=0)

    sentiment_labels = ["Negative", "Neutral", "Positive"]
    sentiment_result = sentiment_labels[sent_label_idx.item()]
    sent_confidence = round(sent_conf.item() * 100, 2)

    st.markdown(f"### Sentiment: {sentiment_result} ({sent_confidence}%)")
    st.progress(float(min(sent_conf.item(), 1.0)))

    emotion_probs = torch.sigmoid(emotion_outputs.logits)[0]
    top_emotions = torch.topk(emotion_probs, k=3)

    st.markdown("### Top Emotions:")
    for idx, score in zip(top_emotions.indices, top_emotions.values):
        label = emotion_labels[idx.item()]
        conf = round(score.item() * 100, 2)
        st.write(f"- {label.capitalize()} ({conf}%)")
