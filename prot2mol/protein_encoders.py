import torch
from transformers import (T5Tokenizer, T5EncoderModel, 
                        AutoTokenizer, EsmModel,
                        EsmTokenizer, EsmForMaskedLM)
from typing import Optional, Tuple, Dict, Any
import re
import numpy as np

def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

class ProteinEncoder:
    """Base class for protein encoders"""
    def __init__(self, max_length: int = 1000, device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 freeze: bool = True):
        self.device = device
        self.tokenizer = None
        self.model = None
        self.max_length = max_length
        
        # Set model training mode based on freeze parameter
        if self.model is not None:
            self.model.train(mode=not freeze)
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False

    def encode(self, sequences: list, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class ProtT5Encoder(ProteinEncoder):
    def __init__(self, model_name: str = "Rostlab/prot_t5_xl_uniref50", max_length: int = 1000,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 active: bool = True):
        super().__init__(max_length, device, active)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, legacy=True)
        self.model = T5EncoderModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

        if active is True:
            self.model.train()
        elif active is False:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def check_model_trainability(self):
        """Check trainability status of both encoder and main model"""
        # Check encoder model
        encoder_trainable_params = count_trainable_parameters(self.model)
        
        print(f"Encoder Model Status:")
        print(f"- Trainable parameters: {encoder_trainable_params}")

    def encode(self, sequences: list, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.check_model_trainability()
        outputs = self.model(input_ids=sequences, attention_mask=attention_mask)
        return outputs.last_hidden_state

class ESM2Encoder(ProteinEncoder):
    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D", max_length: int = 1000,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 freeze: bool = True):
        super().__init__(max_length, device, freeze)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        
        # Set model training mode and freeze parameters after initialization
        self.model.train(mode=not freeze)
        if freeze:
            self.model.eval()

    def encode(self, sequences: list, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        encoded = self.tokenizer(
            sequences,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        
        outputs = self.model(**encoded)
        return outputs.last_hidden_state

class SaProtEncoder(ProteinEncoder):
    def __init__(self, model_name: str = "westlake-repl/SaProt_650M_AF2", max_length: int = 1000,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 freeze: bool = True):
        super().__init__(max_length, device, freeze)
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmForMaskedLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        
        # Set model training mode and freeze parameters after initialization
        self.model.train(mode=not freeze)
        if freeze:
            self.model.eval()

    def encode(self, sequences: list, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        encoded = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        
        outputs = self.model(**encoded, output_hidden_states=True)
        # Return the last hidden state for consistency with other encoders
        return outputs.hidden_states[-1]

def get_protein_encoder(model_name: str, max_length: int = 1000, 
                       active: bool = True) -> ProteinEncoder:
    """Factory function to get the appropriate protein encoder
    
    Args:
        model_name: Name of the encoder model to use
        max_length: Maximum sequence length
        active: Whether to activate model parameters (also sets training mode accordingly)
    """
    encoders = {
        "prot_t5": ProtT5Encoder,
        "esm2": ESM2Encoder,
        "saprot": SaProtEncoder,
    }
    
    if model_name not in encoders:
        raise ValueError(f"Unsupported protein encoder model: {model_name}. Available models: {list(encoders.keys())}")
    
    return encoders[model_name](max_length=max_length, active=active)
