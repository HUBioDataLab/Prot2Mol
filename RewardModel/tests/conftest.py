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

    def __call__(self, texts, **kwargs):
        return self.batch_encode_plus(texts, **kwargs)

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
        lengths = []
        for text in texts:
            text_len = max(1, len(str(text)))
            if truncation and max_length is not None:
                text_len = min(text_len, max_length)
            lengths.append(text_len)

        if padding == "max_length":
            if max_length is None:
                raise ValueError("DummyTokenizer requires max_length when padding='max_length'")
            pad_to = max_length
        elif padding == "longest":
            pad_to = max(lengths, default=0)
        elif padding in (False, None, "do_not_pad"):
            pad_to = None
        else:
            raise ValueError(f"Unsupported padding mode for DummyTokenizer: {padding}")

        rows = []
        masks = []
        for idx, text_len in enumerate(lengths):
            row = [idx + 1] * text_len
            mask = [1] * text_len
            if pad_to is not None:
                pad_len = pad_to - text_len
                row += [self.pad_token_id] * pad_len
                mask += [0] * pad_len
            rows.append(row)
            masks.append(mask)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {
            "input_ids": rows,
            "attention_mask": masks,
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
