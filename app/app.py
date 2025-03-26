import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# Load models and tokenizer
@st.cache_resource
def load_models():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    sentiment_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    sentiment_model.load_state_dict(torch.load("models/bert_sentiment.pt", map_location=torch.device("cpu")))
    sentiment_model.eval()

    emotion_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28)
    emotion_model.load_state_dict(torch.load("models/bert_emotion.pt", map_location=torch.device("cpu")))
    emotion_model.eval()

    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
        'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
        'remorse', 'sadness', 'surprise', 'neutral'
    ]

    return tokenizer, sentiment_model, emotion_model, emotion_labels

tokenizer, sentiment_model, emotion_model, emotion_labels = load_models()

# Theme toggle
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")
if dark_mode:
    st.markdown("""
        <style>
        body { background-color: #121212; color: white; }
        .stApp { background: linear-gradient(to right, #2c3e50, #4ca1af); }
        .stTextInput input { background-color: #333; color: white; }
        .stButton>button { background-color: #1e88e5; color: white; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(to right, #e3f2fd, #bbdefb); }
        </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("## 💬 SentimentSense")
st.markdown("#### An AI-powered Emotion and Sentiment Analyzer")

# Input text
user_input = st.text_area("Enter a sentence to analyze:", height=140)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

            # Sentiment
            sent_outputs = sentiment_model(**inputs)
            sent_logits = sent_outputs.logits
            sent_probs = torch.nn.functional.softmax(sent_logits, dim=1).detach().numpy()[0]
            sent_label = np.argmax(sent_probs)
            sent_conf = float(sent_probs[sent_label])

            sentiment_map = {
                0: "Negative 😔",
                1: "Neutral 😐",
                2: "Very Positive 😍"
            }

            # Emotion
            emo_outputs = emotion_model(**inputs)
            emo_logits = emo_outputs.logits
            emo_probs = torch.nn.functional.softmax(emo_logits, dim=1).detach().numpy()[0]
            emo_label = np.argmax(emo_probs)
            emo_conf = float(emo_probs[emo_label])
            emo_name = emotion_labels[emo_label]

        st.success("✅ Analysis Complete")

        st.subheader(f"Sentiment: {sentiment_map[sent_label]}")
        st.progress(float(min(sent_conf, 1.0)))

        st.subheader(f"Emotion: {emo_name.title()}")
        st.progress(float(min(emo_conf, 1.0)))
