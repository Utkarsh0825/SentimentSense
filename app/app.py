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
# Model Download Logic
# -----------------------------
@st.cache_resource
def download_and_load_models():
    # Download Sentiment Model
    if not os.path.exists(SENTIMENT_MODEL_PATH):
        st.info("📥 Downloading Sentiment Model...")
        gdown.download(f"https://drive.google.com/uc?id={SENTIMENT_ID}", SENTIMENT_MODEL_PATH, quiet=False)

    # Download Emotion Model
    if not os.path.exists(EMOTION_MODEL_PATH):
        st.info("📥 Downloading Emotion Model...")
        gdown.download(f"https://drive.google.com/uc?id={EMOTION_ID}", EMOTION_MODEL_PATH, quiet=False)

    # Load Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load Sentiment Model (trained with 5 labels)
    sentiment_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
    sentiment_model.load_state_dict(torch.load(SENTIMENT_MODEL_PATH, map_location=torch.device("cpu")))
    sentiment_model.eval()

    # Load Emotion Model (trained with 28 labels)
    emotion_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28)
    emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=torch.device("cpu")))
    emotion_model.eval()

    # Emotion Labels
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
# Input UI
# -----------------------------
text_input = st.text_area("Enter a sentence to analyze:", height=150)

if st.button("Analyze") and text_input.strip() != "":
    st.success("✅ Analysis Complete")

    # Tokenize input
    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)

    # Predict Sentiment
    with torch.no_grad():
        sent_outputs = sentiment_model(**inputs)
        emotion_outputs = emotion_model(**inputs)

    # Sentiment
    sent_probs = torch.nn.functional.softmax(sent_outputs.logits, dim=-1)[0]
    sent_conf, sent_label_idx = torch.max(sent_probs, dim=0)

    sentiment_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    sentiment_result = sentiment_labels[sent_label_idx.item()]
    sent_confidence = round(sent_conf.item() * 100, 2)

    st.markdown(f"### Sentiment: {sentiment_result} {'🔥' if sent_label_idx.item() == 4 else '🙂' if sent_label_idx.item() == 3 else '😐' if sent_label_idx.item() == 2 else '☹️'}")
    st.progress(float(min(sent_conf.item(), 1.0)))

    # Emotion
    emotion_probs = torch.sigmoid(emotion_outputs.logits)[0]
    top_emotions = torch.topk(emotion_probs, k=3)

    st.markdown("### Top Emotions:")
    for idx, score in zip(top_emotions.indices, top_emotions.values):
        label = emotion_labels[idx.item()]
        conf = round(score.item() * 100, 2)
        st.write(f"- {label.capitalize()} ({conf}%)")
