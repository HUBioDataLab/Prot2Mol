import logging
import re
from typing import Iterable, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForMaskedLM, EsmModel, T5EncoderModel, T5Tokenizer

from .hf_utils import resolve_model_path

logger = logging.getLogger(__name__)
_NON_STD_AA = re.compile(r"[UZOB]")
def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

class ProteinEncoder:
    """Base class for protein encoders"""
    def __init__(self, max_length: int = 1000, active: bool = False):
        self.model = None
        self.max_length = max_length
        
        # Set model training mode based on freeze parameter
        if self.model is not None:
            self.model.train(mode=active)
            if not active:
                for param in self.model.parameters():
                    param.requires_grad = False

    def encode(self, sequences: list, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class ProtT5Encoder(ProteinEncoder):
    def __init__(self, model_name: str = "Rostlab/prot_t5_xl_uniref50", max_length: int = 1000,
                 active: bool = True):
        super().__init__(max_length, active)
        # Use local path if it contains our local models directory structure
        if "Rostlab" in model_name and not model_name.startswith("/"):
            model_path = resolve_model_path("Rostlab--prot_t5_xl_uniref50")
        else:
            model_path = model_name
        self.model = T5EncoderModel.from_pretrained(model_path)
        if active is True:
            self.model.train()
            logger.info("ProtT5 encoder set to training mode")
        elif active is False:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("ProtT5 encoder frozen (eval mode)")
        self.check_model_trainability()
    def check_model_trainability(self):
        """Check trainability status of both encoder and main model"""
        # Check encoder model
        encoder_trainable_params = count_trainable_parameters(self.model)
        logger.info("Encoder trainable parameters: %s", f"{encoder_trainable_params:,}")

    def encode(self, sequences: list, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        outputs = self.model(input_ids=sequences, attention_mask=attention_mask)

        return outputs.last_hidden_state

class ESM2Encoder(ProteinEncoder):
    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D", max_length: int = 1000,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 active: bool = True):
        super().__init__(max_length, active)
        # Use local path if it contains our local models directory structure
        if "facebook" in model_name and not model_name.startswith("/"):
            model_path = resolve_model_path("facebook--esm2_t33_650M_UR50D")
        else:
            model_path = model_name
        self.model = EsmModel.from_pretrained(model_path)


        # Set model training mode and freeze parameters after initialization
        if active is False:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("ESM2 encoder frozen (eval mode)")
        elif active is True:
            self.model.train()
            logger.info("ESM2 encoder set to training mode")
        self.check_model_trainability()  
    def check_model_trainability(self):
        """Check trainability status of both encoder and main model"""
        # Check encoder model
        encoder_trainable_params = count_trainable_parameters(self.model)
        logger.info("Encoder trainable parameters: %s", f"{encoder_trainable_params:,}")

    def encode(self, sequences: list, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        input_ids = sequences

        

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return outputs.last_hidden_state

class SaProtEncoder(ProteinEncoder):
    def __init__(self, model_name: str = "westlake-repl/SaProt_650M_AF2", max_length: int = 1000,
                 active: bool = True):
        super().__init__(max_length, active)
        # Use local path if it contains our local models directory structure
        if "westlake-repl" in model_name and not model_name.startswith("/"):
            model_path = resolve_model_path("westlake-repl--SaProt_1.3B_AF2")
        else:
            model_path = model_name
        self.model = EsmForMaskedLM.from_pretrained(model_path)
        
        # Set model training mode and freeze parameters after initialization

        if active is False:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        elif active is True:
            self.model.train()
        self.check_model_trainability() 
    def check_model_trainability(self):
        """Check trainability status of both encoder and main model"""
        # Check encoder model
        encoder_trainable_params = count_trainable_parameters(self.model)
        logger.info("Encoder trainable parameters: %s", f"{encoder_trainable_params:,}")

    def encode(self, sequences: list, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        input_ids = sequences
        
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

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

def format_protein_sequence(sequence: str, model_name: str) -> str:
    """Normalize non-standard amino acids and format for tokenizer."""
    cleaned = _NON_STD_AA.sub("X", sequence)
    if model_name == "prot_t5":
        return " ".join(list(cleaned))
    return cleaned


def format_protein_sequences(sequences: Iterable[str], model_name: str) -> List[str]:
    """Vectorized wrapper for format_protein_sequence."""
    return [format_protein_sequence(seq, model_name) for seq in sequences]

def get_protein_tokenizer(model_name: str):
    tokenizers = {
        "prot_t5": T5Tokenizer.from_pretrained(
            resolve_model_path("Rostlab--prot_t5_xl_uniref50"),
            do_lower_case=False, 
            legacy=True, 
            clean_up_tokenization_spaces=True
        ),
        "esm2": AutoTokenizer.from_pretrained(resolve_model_path("facebook--esm2_t36_3B_UR50D")),
        "saprot": AutoTokenizer.from_pretrained(resolve_model_path("westlake-repl--SaProt_1.3B_AF2")),
    }
    return tokenizers[model_name]

def get_encoder_size(model_name: str):
    ENCODER_DIMS = {
    "prot_t5": 1024,
    "esm2": 1280,
    "saprot": 1280
    }
    return ENCODER_DIMS[model_name]
