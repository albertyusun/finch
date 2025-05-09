import torch.nn as nn
from transformers import AutoModel

class MultiTaskModel(nn.Module):
    def __init__(self, encoder_name="distilbert-base-uncased", num_task_labels=6, num_status_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        self.task_head = nn.Linear(hidden_size, num_task_labels)
        self.status_head = nn.Linear(hidden_size, num_status_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.task_head(pooled_output), self.status_head(pooled_output)
