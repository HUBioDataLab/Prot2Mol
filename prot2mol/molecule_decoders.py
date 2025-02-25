import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from typing import Optional, List, Dict, Any
import re

class MoleculeDecoder:
    """Base class for molecule decoders"""
    def __init__(self, max_length: int = 128, device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 freeze: bool = True):
        self.device = device
        self.tokenizer = None
        self.model = None
        self.max_length = max_length
        
        if self.model is not None:
            self.model.train(mode=not freeze)
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class GPTDecoder(MoleculeDecoder):
    """GPT-based decoder initialized from scratch"""
    def __init__(self, vocab_size: int = 50257, n_layer: int = 12, n_head: int = 12, 
                 n_embd: int = 768, max_length: int = 128,
                 freeze: bool = False):
        super().__init__(max_length, freeze)
        
        # Initialize configuration
        config = GPT2Config(
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            max_length=max_length,
            add_cross_attention=True,
            bos_token_id=50256,
            eos_token_id=50256,
            pad_token_id=50256
        )
        
        # Initialize tokenizer and model from scratch
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Initializing GPT decoder from scratch")
        self.model = GPT2LMHeadModel(config=config)

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            labels=labels,
        )
        return outputs

    def generate(self, encoder_hidden_states: torch.Tensor, 
                num_return_sequences: int = 1) -> List[str]:
        outputs = self.model.generate(
            inputs_embeds=None,
            encoder_hidden_states=encoder_hidden_states,
            max_length=self.max_length,
            num_return_sequences=num_return_sequences,
            num_beams=num_return_sequences,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        molecules = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return molecules


def get_molecule_decoder(model_type: str, model_config: Dict[str, Any]) -> MoleculeDecoder:
    """Factory function to get the appropriate molecule decoder
    
    Args:
        model_type: Type of decoder model ('gpt' or 'chemgpt')
        model_config: Configuration dictionary containing model parameters
    """
    decoders = {
        "gpt": GPTDecoder
    }
    
    if model_type not in decoders:
        raise ValueError(f"Unsupported molecule decoder model: {model_type}. Available models: {list(decoders.keys())}")
    
    return decoders[model_type](**model_config)
