import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel
from .protein_encoders import get_protein_encoder, get_encoder_size
import logging
from dataclasses import dataclass
from typing import Optional



class ProteinEncoderWrapper(nn.Module):
    """
    Wrapper to make protein encoders proper PyTorch modules.
    This ensures proper device handling and parameter registration.
    """
    def __init__(self, encoder_obj):
        super().__init__()
        self.encoder_obj = encoder_obj
        # Register the encoder's model as a submodule
        self.encoder_model = encoder_obj.model
        
    def encode(self, sequences, attention_mask=None):
        """Forward the encode call using the registered submodule for proper device handling."""
        # Use the registered submodule instead of the original encoder object
        # to ensure proper device placement with DataParallel
        from transformers import EsmForMaskedLM, T5EncoderModel, EsmModel
        
        if isinstance(self.encoder_model, EsmForMaskedLM):
            # For SaProt that uses EsmForMaskedLM
            outputs = self.encoder_model(
                input_ids=sequences, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            )
            return outputs.hidden_states[-1]
        elif isinstance(self.encoder_model, T5EncoderModel):
            # For ProtT5 models
            outputs = self.encoder_model(input_ids=sequences, attention_mask=attention_mask)
            return outputs.last_hidden_state
        elif isinstance(self.encoder_model, EsmModel):
            # For ESM2 models
            outputs = self.encoder_model(input_ids=sequences, attention_mask=attention_mask)
            return outputs.last_hidden_state
        else:
            # Fallback - try to detect model type by available attributes
            outputs = self.encoder_model(input_ids=sequences, attention_mask=attention_mask)
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                return outputs.hidden_states[-1]
            elif hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state
            else:
                raise ValueError(f"Unknown model output format for {type(self.encoder_model)}")
    
    def forward(self, sequences, attention_mask=None):
        """Standard forward method for PyTorch modules."""
        return self.encode(sequences, attention_mask)

class Prot2MolModel(nn.Module):
    """
    Unified model that combines protein encoder and molecule decoder (GPT2).
    
    This model takes protein sequences as input, encodes them using a protein encoder,
    and generates molecule sequences using a GPT2 decoder with cross-attention.
    """
    
    def __init__(self, config):
        """
        Initialize the Prot2Mol model.
        
        Args:
            config: Dictionary containing model configuration parameters
                - prot_emb_model: Name of the protein encoder model
                - n_layer: Number of transformer layers
                - n_head: Number of attention heads
                - max_mol_len: Maximum molecule sequence length
                - prot_max_length: Maximum protein sequence length
                - train_encoder_model: Whether to train the encoder
                - mol_tokenizer: Molecule tokenizer (for vocab size)
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Store configuration (keep original for internal use)
        self._config = config
        
        # Add attributes expected by Transformers Trainer for checkpoint loading
        self._keys_to_ignore_on_save = None
        self._keys_to_ignore_on_load_missing = None
        self._keys_to_ignore_on_load_unexpected = None
        self._tied_weights_keys = None
        self.base_model_prefix = "prot2mol"
        
        # Initialize protein encoder
        self.logger.info(f"Initializing protein encoder: {self._config['prot_emb_model']}, active: {self._config['train_encoder_model']}")
        encoder_obj = get_protein_encoder(
            model_name=self._config['prot_emb_model'],
            max_length=self._config['prot_max_length'],
            active=self._config['train_encoder_model']
        )
        # Wrap the encoder to make it a proper PyTorch module
        self.protein_encoder = ProteinEncoderWrapper(encoder_obj)
        
        # Get encoder dimension
        encoder_dim = get_encoder_size(self._config['prot_emb_model'])
        
        # Initialize GPT2 configuration
        self.logger.info("Initializing GPT2 decoder configuration")
        self.gpt_config = GPT2Config(
            add_cross_attention=True,
            is_decoder=True,
            n_embd=encoder_dim,
            n_head=self._config['n_head'],
            vocab_size=len(self._config['mol_tokenizer'].added_tokens_decoder),
            pad_token_id=self._config['mol_tokenizer'].pad_token_id,
            n_positions=self._config['max_mol_len'],
            n_layer=self._config['n_layer'],
            bos_token_id=self._config['mol_tokenizer'].bos_token_id,
            eos_token_id=self._config['mol_tokenizer'].eos_token_id
        )
        self.gpt_config.use_cache = False
        # Initialize GPT2 model
        self.logger.info("Initializing GPT2 decoder model")
        self.molecule_decoder = GPT2LMHeadModel(self.gpt_config)
        
        # Initialize auxiliary pChEMBL prediction head
        self.logger.info("Initializing auxiliary pChEMBL prediction head")
        hidden_size = encoder_dim  # Use same dimension as decoder hidden states
        self.readout = AttnPool(hidden_size, n_heads=min(4, self._config['n_head']))
        self.pchembl_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Learnable loss weighting parameters
        self.lm_weight = nn.Parameter(torch.tensor(1.0))
        self.pchembl_weight = nn.Parameter(torch.tensor(1.0))
        
        # Log parameter counts separately
        encoder_params = sum(p.numel() for p in self.protein_encoder.parameters())
        decoder_params = sum(p.numel() for p in self.molecule_decoder.parameters())
        readout_params = sum(p.numel() for p in self.readout.parameters())
        pchembl_params = sum(p.numel() for p in self.pchembl_head.parameters())
        total_params = self.num_parameters()
        
        self.logger.info(f"Protein encoder parameters: {encoder_params:,}")
        self.logger.info(f"Molecule decoder parameters: {decoder_params:,}")
        self.logger.info(f"Readout (AttnPool) parameters: {readout_params:,}")
        self.logger.info(f"pChEMBL prediction head parameters: {pchembl_params:,}")
        self.logger.info(f"Total model parameters: {total_params:,}")
        self.logger.info(f"Learnable loss weights initialized: LM={self.lm_weight.item():.3f}, pChEMBL={self.pchembl_weight.item():.3f}")

        
    @property
    def config(self):
        """Return the GPT2 configuration for compatibility with Transformers library."""
        return self.gpt_config
        
    def forward(self, mol_input_ids, prot_input_ids, prot_attention_mask, labels=None, pchembl_values=None, train_lm=True, pchembl_only_mode=False):
        """
        Forward pass of the model with both language modeling and pChEMBL prediction.
        
        Important behavior:
        - LM loss is computed ONLY for positive samples (train_lm=True, pChEMBL >= threshold)
        - pChEMBL predictions are ALWAYS computed for ALL samples (positive and negative)
        - This allows separate optimization of molecule generation and binding affinity prediction
        
        Args:
            mol_input_ids: Tokenized molecule sequences
            prot_input_ids: Tokenized protein sequences  
            prot_attention_mask: Attention mask for protein sequences
            labels: Target labels for language modeling (optional)
            pchembl_values: Target pChEMBL values for regression (optional)
            train_lm: Whether to compute language modeling loss (boolean or per-sample tensor)
                     - If True: compute LM loss for all samples
                     - If False: don't compute LM loss 
                     - If tensor: per-sample flags (True for positive, False for negative samples)
            pchembl_only_mode: If True, only compute pChEMBL loss (for warm-up phase)
            
        Returns:
            Dict with both language modeling and pChEMBL prediction results
        """
        # Encode protein sequences
        with torch.set_grad_enabled(self.protein_encoder.encoder_model.training):
            protein_embeddings = self.protein_encoder.encode(
                sequences=prot_input_ids,
                attention_mask=prot_attention_mask
            )
        
        # Handle train_lm as either boolean or per-sample tensor
        # If train_lm is a tensor, we need to mask labels for samples where train_lm is False
        labels_for_lm = None
        if labels is not None and not pchembl_only_mode:
            if isinstance(train_lm, torch.Tensor):
                # Per-sample train_lm: mask labels for samples where train_lm is False
                labels_for_lm = labels.clone()
                # For samples where train_lm is False, set all labels to -100
                labels_for_lm[~train_lm] = -100
                
                # If ALL labels are -100 (all negative samples), don't compute LM loss
                if (labels_for_lm == -100).all():
                    labels_for_lm = None
            elif train_lm:  # train_lm is a boolean True
                labels_for_lm = labels
            # else: train_lm is False, labels_for_lm stays None
        
        # Forward pass through GPT2 decoder with cross-attention
        with torch.set_grad_enabled(self.molecule_decoder.training):
            decoder_outputs = self.molecule_decoder(
                input_ids=mol_input_ids,
                attention_mask=(mol_input_ids != self._config['mol_tokenizer'].pad_token_id).float(),
                encoder_hidden_states=protein_embeddings.detach() if pchembl_only_mode else protein_embeddings,
                encoder_attention_mask=prot_attention_mask,
                labels=labels_for_lm,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Extract final hidden states for pChEMBL prediction
        # Use mean pooling over sequence length (excluding padding tokens)
        hidden_states = decoder_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # Create attention mask for molecule tokens to exclude padding
        mol_attention_mask = (mol_input_ids != self._config['mol_tokenizer'].pad_token_id).float()
        
        # Mean pooling over sequence dimension
        masked_hidden_states = hidden_states * mol_attention_mask.unsqueeze(-1)
        # Compute pooled representation using attention pooling
        # In pchembl_only_mode, we detach the input hidden_states to prevent gradients to encoder/decoder
        # but allow gradients to flow through readout and pchembl_head
        input_to_readout = hidden_states.detach() if pchembl_only_mode else hidden_states
        pooled_hidden_states = self.readout(input_to_readout, mol_attention_mask)
        
        # Predict pChEMBL values for ALL samples (positive and negative)
        # Only compute if the head is active/trainable
        pchembl_predictions = None
        should_run_pchembl = self._config.get('train_pchembl_head', True)
        
        if should_run_pchembl:
            # Note: pChEMBL predictions are always computed, regardless of train_lm flag
            pchembl_predictions = self.pchembl_head(pooled_hidden_states).squeeze(-1)  # [batch_size]
        
        # Compute losses based on training mode
        lm_loss = None
        pchembl_loss = None
        total_loss = None
        corr_loss = None
        
        if (not pchembl_only_mode) and (not should_run_pchembl):
            # 1. LM Only Mode (Encoder-Decoder only)
            lm_loss = decoder_outputs.loss
            if lm_loss is not None:
                total_loss = self.lm_weight * lm_loss
                
        elif pchembl_only_mode:
            # 2. pChEMBL Only Mode (Stage 1)
            # Implicit assumption: pchembl_only_mode implies train_pchembl_head is True
            if pchembl_values is not None and pchembl_predictions is not None:
                pchembl_loss = F.mse_loss(pchembl_predictions, pchembl_values) 
                corr_loss = self.corr_loss_calculation(pchembl_predictions, pchembl_values)
                total_loss = pchembl_loss #+ corr_loss * 0.2
                
        else:
            # 3. Full Training Mode (Stage 2 with pChEMBL head active)
            # Note: train_lm is already handled in labels_for_lm above
            lm_loss = decoder_outputs.loss
            
            if pchembl_values is not None and pchembl_predictions is not None:
                pchembl_loss = F.mse_loss(pchembl_predictions, pchembl_values)
                corr_loss = self.corr_loss_calculation(pchembl_predictions, pchembl_values)
                
                # Combine losses with learnable weights
                if lm_loss is not None:
                    total_loss = self.lm_weight * lm_loss + self.pchembl_weight * pchembl_loss
                else:
                    total_loss = self.pchembl_weight * pchembl_loss
            elif lm_loss is not None:
                # Fallback if pchembl values are missing but head is active
                total_loss = self.lm_weight * lm_loss

        outputs = {
            "loss": total_loss,
            "lm_loss": lm_loss,
            "pchembl_loss": pchembl_loss,
            #"corr_loss": corr_loss,
            "logits": decoder_outputs.logits,
            "pchembl_predictions": pchembl_predictions,
            "hidden_states": decoder_outputs.hidden_states,
            "attentions": decoder_outputs.attentions,
            "cross_attentions": decoder_outputs.cross_attentions,
        }
        return {k: v for k, v in outputs.items() if v is not None}
    
    def generate(self, prot_input_ids, prot_attention_mask, **generation_kwargs):
        """
        Generate molecule sequences given protein sequences.
        
        Args:
            prot_input_ids: Tokenized protein sequences
            prot_attention_mask: Attention mask for protein sequences
            **generation_kwargs: Additional arguments for generation
            
        Returns:
            Generated molecule token sequences
        """
        # Encode protein sequences
        protein_embeddings = self.protein_encoder.encode(
            sequences=prot_input_ids,
            attention_mask=prot_attention_mask
        )
        
        # Generate using GPT2
        return self.molecule_decoder.generate(
            encoder_hidden_states=protein_embeddings,
            encoder_attention_mask=prot_attention_mask,
            **generation_kwargs
        )

    def corr_loss_calculation(self, pchembl_predictions, pchembl_values, eps=1e-8):
        x = pchembl_predictions - pchembl_predictions.mean(); v = pchembl_values - pchembl_values.mean()
        return 1 - (x*v).mean() / (x.pow(2).mean().sqrt()*v.pow(2).mean().sqrt()+eps)
    
    def get_encoder_hidden_states(self, prot_input_ids, prot_attention_mask):
        """
        Get the hidden states from the protein encoder.
        """
        return self.protein_encoder.encode(
            sequences=prot_input_ids,
            attention_mask=prot_attention_mask)

    def num_parameters(self):
        """Return the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def num_trainable_parameters(self):
        """Return the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_encoder(self):
        """Return the protein encoder."""
        return self.protein_encoder
    
    def get_decoder(self):
        """Return the molecule decoder."""
        return self.molecule_decoder

    def _set_requires_grad(self, module, requires_grad):
        for param in module.parameters():
            param.requires_grad = requires_grad

    def update_trainable_components(self, trainable_encoder: bool, trainable_decoder: bool, trainable_pchembl_head: bool):
        """
        Granularly freeze or unfreeze model components.

        Args:
            trainable_encoder: If True, the protein encoder will be trainable.
            trainable_decoder: If True, the molecule decoder will be trainable.
            trainable_pchembl_head: If True, the pChEMBL prediction head will be trainable.
        """
        self.logger.info(f"Updating trainable components: Encoder={trainable_encoder}, Decoder={trainable_decoder}, pChEMBL Head={trainable_pchembl_head}")
        
        self._set_requires_grad(self.protein_encoder, trainable_encoder)
        self._set_requires_grad(self.molecule_decoder, trainable_decoder)
        # Train both readout and pchembl_head together
        self._set_requires_grad(self.readout, trainable_pchembl_head)
        self._set_requires_grad(self.pchembl_head, trainable_pchembl_head)
        
        # The learnable loss weights should always be trainable
        self.lm_weight.requires_grad = True
        self.pchembl_weight.requires_grad = True
        
        encoder_trainable = sum(p.numel() for p in self.protein_encoder.parameters() if p.requires_grad)
        decoder_trainable = sum(p.numel() for p in self.molecule_decoder.parameters() if p.requires_grad)
        readout_trainable = sum(p.numel() for p in self.readout.parameters() if p.requires_grad)
        pchembl_head_trainable = sum(p.numel() for p in self.pchembl_head.parameters() if p.requires_grad)
        lm_weight_trainable = self.lm_weight.numel() if self.lm_weight.requires_grad else 0
        pchembl_weight_trainable = self.pchembl_weight.numel() if self.pchembl_weight.requires_grad else 0
        total_trainable = self.num_trainable_parameters()
        self.logger.info(
            f"Trainable parameters per module: "
            f"Encoder={encoder_trainable:,}, "
            f"Decoder={decoder_trainable:,}, "
            f"Readout={readout_trainable:,}, "
            f"pChEMBL Head={pchembl_head_trainable:,}, "
            f"LM Weight={lm_weight_trainable}, "
            f"pChEMBL Weight={pchembl_weight_trainable}, "
            f"Total={total_trainable:,}"
        )


def create_prot2mol_model(config):
    """
    Factory function to create a Prot2MolModel.
    
    Args:
        config: Dictionary containing model configuration
        
    Returns:
        Prot2MolModel instance
    """
    return Prot2MolModel(config)

class AttnPool(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln    = nn.LayerNorm(d_model)

    def forward(self, H, mask):
        # H: [B, T, d], mask: [B, T] with 1 for tokens, 0 for pads
        B = H.size(0)
        q = self.query.expand(B, -1, -1)                 # [B,1,d]
        # key_padding_mask: True for pads
        kpm = ~(mask.bool())                              # [B,T]
        pooled, _ = self.attn(q, H, H, key_padding_mask=kpm, need_weights=False)
        return self.ln(pooled.squeeze(1))                 # [B,d]