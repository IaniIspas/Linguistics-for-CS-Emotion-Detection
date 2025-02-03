import os
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    pipeline
)
from datasets import Dataset, ClassLabel
import shap
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

df = pd.read_csv("/kaggle/input/emotions/text.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df = df[["text", "label"]]

emotion_classes = ["sadness", "joy", "love", "anger", "fear", "surprise"]
class_label = ClassLabel(num_classes=6, names=emotion_classes)

dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("label", class_label)
dataset = dataset.shuffle(seed=42)
train_test = dataset.train_test_split(test_size=0.2)
train_dataset = train_test["train"]
test_dataset = train_test["test"]

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = text.strip()

    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]
    return " ".join(tokens)

def tokenize_function(examples):
    cleaned_texts = [clean_text(t) for t in examples["text"]]
    return tokenizer(cleaned_texts, truncation=True, padding=False)

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format("torch")
test_dataset.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

training_args = TrainingArguments(
    output_dir="./roberta",
    eval_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    load_best_model_at_end=False,
    save_total_limit=1,
    report_to="none"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./roberta")
tokenizer.save_pretrained("./roberta")
