import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class DummyBatchTokenizer:
    def __init__(self, pad_token_id=0, bos_token_id=1, eos_token_id=2):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.added_tokens_decoder = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "C"}
        self.calls = []

    def batch_encode_plus(
        self,
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=8,
        padding="max_length",
        return_tensors="pt",
        **kwargs,
    ):
        self.calls.append(list(texts))
        rows = []
        mask = []
        for idx, text in enumerate(texts):
            base_len = min(max(1, len(str(text)) % max_length), max_length)
            token_row = [min(3 + idx, 9)] * base_len
            attn_row = [1] * base_len
            if padding == "max_length":
                pad_len = max_length - base_len
                token_row += [self.pad_token_id] * pad_len
                attn_row += [0] * pad_len
            rows.append(token_row)
            mask.append(attn_row)
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

    def add_tokens(self, tokens):
        start = len(self.added_tokens_decoder)
        for i, tok in enumerate(tokens):
            self.added_tokens_decoder[start + i] = tok

    def decode(self, token_ids, skip_special_tokens=True):
        return "[C]"


class DummyEncoderModel(torch.nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
        batch, seq = input_ids.shape
        hidden = torch.zeros(batch, seq, self.hidden_size, dtype=torch.float32, device=input_ids.device)
        if output_hidden_states:
            return SimpleNamespace(hidden_states=[hidden, hidden])
        return SimpleNamespace(last_hidden_state=hidden)


class DummyEncoderObj:
    def __init__(self, hidden_size=8):
        self.model = DummyEncoderModel(hidden_size=hidden_size)


class DummyDecoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = torch.nn.Linear(8, 8)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
    ):
        batch, seq = input_ids.shape
        hidden = torch.zeros(batch, seq, 8, dtype=torch.float32, device=input_ids.device)
        logits = torch.zeros(batch, seq, self.config.vocab_size, dtype=torch.float32, device=input_ids.device)
        loss = None
        if labels is not None:
            loss = torch.tensor(0.5, dtype=torch.float32, device=input_ids.device)

        cross_attn = None
        if output_attentions:
            enc_len = encoder_hidden_states.shape[1]
            cross_attn = (
                torch.full((batch, self.config.n_head, seq, enc_len), 1.0 / max(enc_len, 1), device=input_ids.device),
            )

        out = SimpleNamespace(
            loss=loss,
            logits=logits,
            hidden_states=[hidden, hidden],
            attentions=None,
            cross_attentions=cross_attn,
        )
        return out

    def generate(self, encoder_hidden_states=None, encoder_attention_mask=None, **kwargs):
        batch = encoder_hidden_states.shape[0]
        return torch.ones((batch, 4), dtype=torch.long, device=encoder_hidden_states.device)
