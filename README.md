# 🧠 SentimentSense

**SentimentSense** is an AI-powered web application that analyzes both **sentiment** and **emotions** from user-provided text. Built using cutting-edge **BERT-based models**, it supports 28 distinct emotions, 5-level sentiment grading, and includes advanced features like **voice input** and a simulated **geo-sentiment tracker**.

<p align="center">
  <img src="https://github.com/Utkarsh0825/SentimentSense/blob/main/assets/sentimentsense-banner.png" width="100%" alt="SentimentSense Banner"/>
</p>

---

## 🌐 Live Demo

👉 Try it out here: [https://sentimentsense-utkarsh.streamlit.app](https://sentimentsense-utkarsh.streamlit.app)

---

## 🧠 Features

- ✅ **28-Class Emotion Detection** (GoEmotions dataset powered by BERT)
- ✅ **5-Level Sentiment Analysis** (Very Negative to Very Positive)
- ✅ **Voice Input Support** for seamless text capture
- ✅ **Mobile-Responsive UI** with gradient themes and dark mode
- ✅ **Simulated Geo-Sentiment Tracker** with interactive world map
- ✅ Clean, minimal UI inspired by big tech design standards

---

## ⚙️ Tech Stack

- **Frontend & App Framework**: Streamlit
- **NLP Models**: BERT (HuggingFace Transformers)
- **Model Training**: PyTorch
- **Speech-to-Text**: `speech_recognition`, `streamlit-mic-recorder`
- **Geo Mapping**: `pydeck`, `plotly`, `pandas`
- **Hosting**: Streamlit Cloud

---

## 🖼️ Screenshots

> 📸 Add screenshots to the `/assets/` folder and link them like below:

<p align="center">
  <img src="https://github.com/Utkarsh0825/SentimentSense/blob/main/assets/emotion-tab.png" width="80%" />
  <img src="https://github.com/Utkarsh0825/SentimentSense/blob/main/assets/geo-tracker.png" width="80%" />
</p>

---

## 🧪 How to Run Locally

> *Only needed for local development. Live version is fully web-based.*

### 1. Clone the Repository

```bash
git clone https://github.com/Utkarsh0825/SentimentSense.git
cd SentimentSense
### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run the app 
streamlit run app.py

🤝 Contributing
Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes:

Fork the repository

Create a new branch

Submit a pull request

🙋‍♂️ Creator
Utkarsh Yadav

⭐ If you found this project helpful or inspiring, don’t forget to star the repository!

