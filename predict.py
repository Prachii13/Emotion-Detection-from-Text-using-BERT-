from transformers import BertTokenizer, BertForSequenceClassification
import torch
import sys

model = BertForSequenceClassification.from_pretrained("emotion_model")
tokenizer = BertTokenizer.from_pretrained("emotion_model")

text = sys.argv[1]
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
label = torch.argmax(outputs.logits, dim=1).item()

labels = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
print("🧠 Emotion:", labels[label])
