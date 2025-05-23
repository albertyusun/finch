#!/usr/bin/env python
"""
Task: Predicts task & status using a trained model.

Input: Path to a CSV file (e.g., data/test.csv) with columns:
       note_id, text, task, completed
       (Assumes no header row in the input CSV)

Output: Writes a CSV file (e.g., preds.csv) with columns:
        note_id, text, task (predicted), completed (predicted: 0, 1, or 2)

Usage: python predict_csv.py data/test.csv preds.csv
"""
import sys

import pandas as pd

import torch.nn.functional as F

import torch
import os
from transformers import AutoTokenizer, AutoModel
from utils import TASK_LABELS, STATUS_LABELS, clean, IMPUTE_UNCLEAR

from multitask_model import MultiTaskModel

# --- Load Model and Tokenizer ---
model_dir = "models"
# Implement loading your trained model and tokenizer from model_dir
tokenizer = AutoTokenizer.from_pretrained(model_dir)

if IMPUTE_UNCLEAR:
    num_status = 3
else:
    num_status = 2

model = MultiTaskModel(num_status_labels=num_status)
model.encoder = AutoModel.from_pretrained(model_dir)

# Load heads
model.load_state_dict(torch.load(os.path.join(model_dir, "multitask_heads.pt"), map_location="cpu"))
model.eval()

# --- Prediction Logic ---
def predict(texts, threshold=0.9):
    """Predicts task (string) and status (0, 1, or 2 using confidence thresholding)."""
    cleaned = [clean(t) for t in texts]
    encodings = tokenizer(cleaned, return_tensors="pt", padding=True, truncation=True, max_length=64)

    with torch.no_grad():
        task_logits, status_logits = model(encodings['input_ids'], encodings['attention_mask'])

    # Task predictions
    task_preds = task_logits.argmax(dim=1).tolist()
    task_labels = [TASK_LABELS[i] for i in task_preds]

    # Status predictions using confidence threshold
    probs = F.softmax(status_logits, dim=1)
    confidence, prediction = probs.max(dim=1)
    entropy = -(probs * probs.log()).sum(dim=1)  # shape: (batch_size,)

    status_preds = [
        2 if ent.item() > threshold else pred.item()
        for ent, pred in zip(entropy, prediction)
    ]

    return task_labels, status_preds

def main(src, dst):
    # 1. Load Data
    try:
        # Read CSV assuming no header row, and provide column names
        df = pd.read_csv(src, header=None, names=['note_id', 'text', 'task', 'completed'])
        print(f"Loaded {len(df)} rows from {src}")
    except FileNotFoundError:
        print(f"Error: Input CSV not found at {src}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {src}: {e}")
        sys.exit(1)

    # 2. Get Predictions
    task_predictions, status_predictions = predict(df['text'].tolist())

    # 3. Create Output DataFrame
    output_df = pd.DataFrame({
        'note_id': df['note_id'],
        'text': df['text'],
        'task': task_predictions,
        'completed': status_predictions
    })

    # 4. Save Output
    try:
        output_df.to_csv(dst, index=False)
        print(f"💾  wrote {dst} with columns: {output_df.columns.tolist()}")
    except Exception as e:
        print(f"Error writing predictions to {dst}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv)!=3:
        print(f"Usage: python {sys.argv[0]} <input_csv_path> <output_csv_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
