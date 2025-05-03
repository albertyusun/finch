import re

from transformers import AutoTokenizer

TASK_LABELS = [
    "Intake Call",
    "Sign Engagement Letter",
    "Request Medical Records",
    "Client Check-in",
    "Create Demand",
    "none",
]
STATUS_LABELS = ["complete", "not_completed", "not_sure"]

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

def clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()

def encode(texts, max_len=64):
    return tokenizer(
        [clean(t) for t in texts],
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
