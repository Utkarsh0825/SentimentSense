import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

# ------------------ Load Dataset ------------------ #
df = pd.read_csv("data/emotion/go_emotions_dataset.csv")
emotion_labels = df.columns[3:].tolist()  # All emotion columns
df["label"] = df[emotion_labels].idxmax(axis=1)  # Pick dominant emotion column name

# Remove rows with no emotion
df = df[df[emotion_labels].sum(axis=1) > 0][["text", "label"]].dropna()

# Encode emotion labels as numbers
label2id = {label: idx for idx, label in enumerate(sorted(df["label"].unique()))}
id2label = {v: k for k, v in label2id.items()}
df["label"] = df["label"].map(label2id)

# Reduce data size for faster training
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
df = df[:4800]  # Max 300 batches × batch_size(16)

# ------------------ Tokenization ------------------ #
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class EmotionDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} | {'labels': torch.tensor(self.labels[idx])}

    def __len__(self):
        return len(self.labels)

train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2)
train_dataset = EmotionDataset(train_texts.tolist(), train_labels.tolist())
val_dataset = EmotionDataset(val_texts.tolist(), val_labels.tolist())

# ------------------ Model Setup ------------------ #
num_labels = len(label2id)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# ------------------ Train Loop ------------------ #
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

# ------------------ Save Model & Labels ------------------ #
model.save_pretrained("models/bert_emotion")
tokenizer.save_pretrained("models/bert_emotion")
torch.save(model.state_dict(), "models/bert_emotion.pt")

# Save label map for future use
import json
with open("models/emotion_labels.json", "w") as f:
    json.dump(id2label, f)

print("✅ Emotion model trained and saved successfully!")
