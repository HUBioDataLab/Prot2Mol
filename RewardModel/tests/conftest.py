import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class DummyTokenizer:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
        self.calls = []

    def batch_encode_plus(
        self,
        texts,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=8,
        return_tensors="pt",
        **kwargs,
    ):
        self.calls.append(
            {
                "texts": list(texts),
                "max_length": max_length,
                "padding": padding,
                "truncation": truncation,
            }
        )
        rows = []
        masks = []
        for idx, text in enumerate(texts):
            text_len = min(max(1, len(str(text))), max_length)
            row = [idx + 1] * text_len
            mask = [1] * text_len
            if padding == "max_length":
                pad_len = max_length - text_len
                row += [self.pad_token_id] * pad_len
                mask += [0] * pad_len
            rows.append(row)
            masks.append(mask)
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class DummyEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        base = torch.arange(self.config.hidden_size, dtype=torch.float32, device=input_ids.device)
        hidden = base.view(1, 1, -1).repeat(batch_size, seq_len, 1)
        hidden = hidden + input_ids.unsqueeze(-1).float()
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1).float()
        return SimpleNamespace(last_hidden_state=self.proj(hidden))
