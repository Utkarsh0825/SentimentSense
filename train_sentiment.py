import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

# ------------------ Sentiment Mapping ------------------ #
POSITIVE = ['admiration', 'amusement', 'approval', 'caring', 'desire', 'excitement',
            'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief']
NEGATIVE = ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
            'fear', 'grief', 'nervousness', 'remorse', 'sadness']
NEUTRAL = ['curiosity', 'realization', 'surprise', 'confusion', 'neutral']

# ------------------ Load Dataset ------------------ #
df = pd.read_csv("data/sentiment/emotions.csv")

def map_sentiment(row):
    pos = row[POSITIVE].sum()
    neg = row[NEGATIVE].sum()
    neu = row[NEUTRAL].sum()
    if max(pos, neg, neu) == pos:
        return 2  # Positive
    elif max(pos, neg, neu) == neg:
        return 0  # Negative
    else:
        return 1  # Neutral

df['label'] = df.apply(map_sentiment, axis=1)
df = df[['text', 'label']].dropna()

# ------------------ Reduce Dataset ------------------ #
df = df.sample(frac=1).reset_index(drop=True)  # shuffle
df = df[:4800]  # Max 300 batches × batch_size(16)

# ------------------ Tokenization ------------------ #
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} | {'labels': torch.tensor(self.labels[idx])}

    def __len__(self):
        return len(self.labels)

train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2)
train_dataset = SentimentDataset(train_texts.tolist(), train_labels.tolist())
val_dataset = SentimentDataset(val_texts.tolist(), val_labels.tolist())

# ------------------ Model Setup ------------------ #
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# ------------------ Train for 3 Epochs, Max 300 Batches ------------------ #
max_batches = 300
for epoch in range(3):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for i, batch in enumerate(loop):
        if i >= max_batches:
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

# ------------------ Save Model ------------------ #
model.save_pretrained("models/bert_sentiment")
tokenizer.save_pretrained("models/bert_sentiment")
torch.save(model.state_dict(), "models/bert_sentiment.pt")

print("✅ Sentiment model trained and saved successfully!")
