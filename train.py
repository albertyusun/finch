#!/usr/bin/env python
"""
Task: Finetune a transformer model for task and status classification.

Input: Path to a CSV file (e.g., data/train.csv) with columns:
       note_id, text, task, completed (0 or 1)

Output: Saves a trained model to a specified directory (e.g., ./models/)

Usage: python train.py data/train.csv
"""

import sys

import pandas as pd

from multitask_model import MultiTaskModel

from utils import TASK_LABELS, STATUS_LABELS, clean
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os

from transformers import AutoTokenizer, AutoModel, AdamW, get_scheduler

from sklearn.metrics import accuracy_score
from tqdm import tqdm

import time

class NotesDataset(Dataset):
    def __init__(self, tokenizer, texts, task_labels, status_labels):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=64,
            return_tensors="pt"
        )
        self.task_labels = torch.tensor(task_labels.tolist())
        self.status_labels = torch.tensor(status_labels.tolist())

    def __len__(self):
        return len(self.task_labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['task_label'] = self.task_labels[idx]
        item['status_label'] = self.status_labels[idx]
        return item



# --- Training Logic ---
def main(csv_path):

    start_time = time.time()

    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
    except FileNotFoundError:
        print(f"Error: Input CSV not found at {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        sys.exit(1)

    # 2. Preprocess Data
    # Clean and encode labels
    df['text'] = df['text'].apply(clean)

    # Convert task string -> task label index
    task_to_id = {label: i for i, label in enumerate(TASK_LABELS)}
    df['task_label'] = df['task'].map(task_to_id)

    # Convert completed column -> status label index (0 or 1)
    df["status_label"] = df["completed"].astype(int)

    # 3. Split Data (Optional but Recommended)
    # Implement train/validation split
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    train_dataset = NotesDataset(tokenizer, train_df['text'], train_df['task_label'], train_df['status_label'])
    val_dataset = NotesDataset(tokenizer, val_df['text'], val_df['task_label'], val_df['status_label'])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # 4. Define Model
    # Instantiate your model architecture
    model = MultiTaskModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch = next(iter(train_loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    task_logits, status_logits = model(input_ids, attention_mask)
    print("Task logits:", task_logits.shape)     # (batch_size, 6)
    print("Status logits:", status_logits.shape) # (batch_size, 3)

    # 5. Define Optimizer, Scheduler, Loss
    optimizer = AdamW(model.parameters(), lr=2e-5)

    task_loss_fn = nn.CrossEntropyLoss()
    status_loss_fn = nn.CrossEntropyLoss()

    num_epochs = 2
    num_training_steps = num_epochs * len(train_loader)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # 6. Training Loop
    # Implement the main training loop (epochs, batches, forward/backward pass)
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            task_labels = batch['task_label'].to(device)
            status_labels = batch['status_label'].to(device)

            optimizer.zero_grad()

            task_logits, status_logits = model(input_ids, attention_mask)

            loss_task = task_loss_fn(task_logits, task_labels)
            loss_status = status_loss_fn(status_logits, status_labels)
            loss = loss_task + loss_status

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()

        print(f"  Training Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        task_preds, status_preds = [], []
        task_true, status_true = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                task_labels = batch['task_label'].to(device)
                status_labels = batch['status_label'].to(device)

                task_logits, status_logits = model(input_ids, attention_mask)

                task_preds += task_logits.argmax(dim=1).cpu().tolist()
                status_preds += status_logits.argmax(dim=1).cpu().tolist()
                task_true += task_labels.cpu().tolist()
                status_true += status_labels.cpu().tolist()

        task_acc = accuracy_score(task_true, task_preds)
        status_acc = accuracy_score(status_true, [0 if x == 2 else x for x in status_preds])

        print(f"  Val Task Accuracy: {task_acc:.4f}")
        print(f"  Val Status Accuracy (2->0): {status_acc:.4f}")

    # 7. Save Model
    # Save the trained model artifacts (weights, tokenizer config, etc.)
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model.encoder.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save task and status heads
    torch.save(model.state_dict(), os.path.join(output_dir, "multitask_heads.pt"))
    print(f"Placeholder: Training finished. Implement model saving to ./{output_dir}/")

    print(f"Time elapsed: {time.time() - start_time}")

if __name__ == "__main__":
    if len(sys.argv)!=2:
        print(f"Usage: python {sys.argv[0]} <path_to_train_csv>")
        sys.exit(1)
    main(sys.argv[1])
