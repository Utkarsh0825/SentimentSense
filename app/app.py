import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import os
import gdown

# UI setup
st.set_page_config(page_title="SentimentSense", layout="centered")
st.markdown("<h1 style='text-align: center;'>💬 SentimentSense</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>An AI-powered Emotion and Sentiment Analyzer</h4>", unsafe_allow_html=True)

# Google Drive file IDs
SENTIMENT_FILE_ID = "10p14bu-kCjJzha5NkSl3tyzBRDPk85Qm"
EMOTION_FILE_ID = "1TkUrX-E16zF5lXm19khJbB9rgQ8PmsZq"

MODEL_DIR = "models"

@st.cache_resource
def load_models():
    download_and_load_models()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load Sentiment model (trained with 5 classes)
    sentiment_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
    sentiment_model.load_state_dict(torch.load("bert_sentiment.pt", map_location=torch.device("cpu")))
    sentiment_model.eval()

    # Load Emotion model (trained with 28 emotion classes)
    emotion_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28)
    emotion_model.load_state_dict(torch.load("bert_emotion.pt", map_location=torch.device("cpu")))
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

# Input box
text_input = st.text_area("Enter a sentence to analyze:", height=150)

if st.button("Analyze") and text_input.strip():
    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        sentiment_output = sentiment_model(**inputs).logits
        emotion_output = emotion_model(**inputs).logits

    sentiment_probs = F.softmax(sentiment_output, dim=1)[0]
    sentiment_label = torch.argmax(sentiment_probs).item()
    sent_conf = float(torch.max(sentiment_probs).item())  # Fix float32

    sentiment_classes = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    sentiment_emojis = ["😠", "🙁", "😐", "🙂", "😍"]
    sentiment_result = sentiment_classes[sentiment_label] + " " + sentiment_emojis[sentiment_label]

    emotion_probs = torch.sigmoid(emotion_output)[0]
    top_emotions = torch.topk(emotion_probs, k=3)
    emotion_results = [(emotion_labels[i], float(emotion_probs[i])) for i in top_emotions.indices]

    st.success("✅ Analysis Complete")
    st.markdown(f"**Sentiment:** {sentiment_result}")
    st.progress(min(sent_conf, 1.0))  # Prevent Streamlit progress error

    st.markdown("**Top Emotions:**")
    for emotion, score in emotion_results:
        st.write(f"- {emotion.capitalize()} ({round(score * 100)}%)")
