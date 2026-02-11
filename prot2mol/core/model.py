import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from .protein_encoders import get_protein_encoder, get_encoder_size
import logging



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
        pair_dim = self._config.get("affinity_pair_dim", min(256, hidden_size))
        pair_heads = self._config.get("affinity_pair_heads", min(8, self._config['n_head']))
        pair_layers = self._config.get("affinity_pair_layers", 2)
        max_prot_tokens = self._config.get("affinity_max_prot_tokens", 256)
        attn_bins = self._config.get("affinity_attn_bins", 16)
        self.pchembl_head = AffinityHead(
            d_model=hidden_size,
            d_pair=pair_dim,
            n_heads=pair_heads,
            n_layers=pair_layers,
            max_prot_tokens=max_prot_tokens,
            attn_bins=attn_bins,
            dropout=self._config.get("affinity_dropout", 0.1)
        )
        self.logger.info(
            "pChEMBL gradient backprop to encoder/decoder: %s",
            "disabled" if self._config.get("stop_pchembl_gradients", True) else "enabled",
        )
        
        # Learnable loss weighting parameters
        self.lm_weight = nn.Parameter(torch.tensor(1.0))
        self.pchembl_weight = nn.Parameter(torch.tensor(1.0))

        # Track per-module trainability so global train()/eval() calls can respect freezing.
        self._trainable_encoder = bool(self._config.get("train_encoder_model", True))
        self._trainable_decoder = bool(self._config.get("train_decoder_model", True))
        self._trainable_pchembl_head = bool(self._config.get("train_pchembl_head", True))
        
        # Log parameter counts separately
        encoder_params = sum(p.numel() for p in self.protein_encoder.parameters())
        decoder_params = sum(p.numel() for p in self.molecule_decoder.parameters())
        pchembl_params = sum(p.numel() for p in self.pchembl_head.parameters())
        total_params = self.num_parameters()
        
        self.logger.info(f"Protein encoder parameters: {encoder_params:,}")
        self.logger.info(f"Molecule decoder parameters: {decoder_params:,}")
        self.logger.info(f"pChEMBL prediction head parameters: {pchembl_params:,}")
        self.logger.info(f"Total model parameters: {total_params:,}")
        self.logger.info(f"Learnable loss weights initialized: LM={self.lm_weight.item():.3f}, pChEMBL={self.pchembl_weight.item():.3f}")

        
    @property
    def config(self):
        """Return the GPT2 configuration for compatibility with Transformers library."""
        return self.gpt_config
        
    def forward(self, mol_input_ids, prot_input_ids, prot_attention_mask, labels=None, pchembl_values=None, train_lm=True, pchembl_only_mode=False):
        """
        Forward pass for language modeling and pChEMBL prediction.
        
        Important behavior:
        - LM labels can be masked per-sample via `train_lm`.
        - pChEMBL predictions are computed for all samples when the head is enabled.
        - Loss composition is handled in the custom trainer to keep responsibilities clear.
        
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
            Dict with logits, optional lm_loss, and optional pchembl_predictions
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
        
        # Determine if we need cross-attention weights for the pChEMBL head
        should_run_pchembl = self._config.get('train_pchembl_head', True)
        output_attentions = bool(should_run_pchembl)
        
        # Forward pass through GPT2 decoder with cross-attention
        with torch.set_grad_enabled(self.molecule_decoder.training):
            decoder_outputs = self.molecule_decoder(
                input_ids=mol_input_ids,
                attention_mask=(mol_input_ids != self._config['mol_tokenizer'].pad_token_id).float(),
                encoder_hidden_states=protein_embeddings.detach() if pchembl_only_mode else protein_embeddings,
                encoder_attention_mask=prot_attention_mask,
                labels=labels_for_lm,
                output_hidden_states=True,
                output_attentions=output_attentions,
                return_dict=True
            )
        
        # Extract final hidden states for pChEMBL prediction
        # Use mean pooling over sequence length (excluding padding tokens)
        hidden_states = decoder_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # Create attention mask for molecule tokens to exclude padding
        mol_attention_mask = (mol_input_ids != self._config['mol_tokenizer'].pad_token_id).float()
        
        # Predict pChEMBL values for ALL samples (positive and negative)
        # Only compute if the head is active/trainable
        pchembl_predictions = None
        if should_run_pchembl:
            # Note: pChEMBL predictions are always computed, regardless of train_lm flag
            stop_pchembl_gradients = self._config.get("stop_pchembl_gradients", True)
            block_backprop = pchembl_only_mode or stop_pchembl_gradients

            head_prot = protein_embeddings.detach() if block_backprop else protein_embeddings
            head_mol = hidden_states.detach() if block_backprop else hidden_states
            cross_attn = None
            if decoder_outputs.cross_attentions is not None:
                cross_attn = self._reduce_cross_attn(decoder_outputs.cross_attentions)
                if block_backprop:
                    cross_attn = cross_attn.detach()
            pchembl_predictions = self.pchembl_head(
                head_prot,
                head_mol,
                cross_attn,
                prot_attention_mask,
                mol_attention_mask
            )

        lm_loss = decoder_outputs.loss

        outputs = {
            "lm_loss": lm_loss,
            "logits": decoder_outputs.logits,
            "pchembl_predictions": pchembl_predictions,
            "hidden_states": decoder_outputs.hidden_states,
            "attentions": decoder_outputs.attentions,
            "cross_attentions": decoder_outputs.cross_attentions,
        }
        if lm_loss is not None:
            # Backward-compatible field for consumers expecting "loss" from the model output.
            outputs["loss"] = lm_loss
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

    def _reduce_cross_attn(self, cross_attentions):
        """Average cross-attention over layers and heads to get [B, Lm, Lp]."""
        if cross_attentions is None:
            return None
        if isinstance(cross_attentions, (list, tuple)):
            attn = torch.stack(cross_attentions, dim=0)
        else:
            attn = cross_attentions
        if attn.dim() == 5:
            # [layers, B, heads, Lm, Lp]
            attn = attn.mean(dim=0).mean(dim=1)
        elif attn.dim() == 4:
            # [B, heads, Lm, Lp]
            attn = attn.mean(dim=1)
        return attn
    
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

        self._trainable_encoder = bool(trainable_encoder)
        self._trainable_decoder = bool(trainable_decoder)
        self._trainable_pchembl_head = bool(trainable_pchembl_head)

        self._set_requires_grad(self.protein_encoder, trainable_encoder)
        self._set_requires_grad(self.molecule_decoder, trainable_decoder)
        self._set_requires_grad(self.pchembl_head, trainable_pchembl_head)

        self.protein_encoder.train(mode=trainable_encoder)
        self.molecule_decoder.train(mode=trainable_decoder)
        self.pchembl_head.train(mode=trainable_pchembl_head)
        
        # The learnable loss weights should always be trainable
        self.lm_weight.requires_grad = True
        self.pchembl_weight.requires_grad = True
        
        encoder_trainable = sum(p.numel() for p in self.protein_encoder.parameters() if p.requires_grad)
        decoder_trainable = sum(p.numel() for p in self.molecule_decoder.parameters() if p.requires_grad)
        pchembl_head_trainable = sum(p.numel() for p in self.pchembl_head.parameters() if p.requires_grad)
        lm_weight_trainable = self.lm_weight.numel() if self.lm_weight.requires_grad else 0
        pchembl_weight_trainable = self.pchembl_weight.numel() if self.pchembl_weight.requires_grad else 0
        total_trainable = self.num_trainable_parameters()
        self.logger.info(
            f"Trainable parameters per module: "
            f"Encoder={encoder_trainable:,}, "
            f"Decoder={decoder_trainable:,}, "
            f"pChEMBL Head={pchembl_head_trainable:,}, "
            f"LM Weight={lm_weight_trainable}, "
            f"pChEMBL Weight={pchembl_weight_trainable}, "
            f"Total={total_trainable:,}"
        )

    def train(self, mode: bool = True):
        """
        Keep frozen components in eval mode even when parent code calls model.train().
        """
        super().train(mode)
        if mode:
            if not self._trainable_encoder:
                self.protein_encoder.eval()
            if not self._trainable_decoder:
                self.molecule_decoder.eval()
            if not self._trainable_pchembl_head:
                self.pchembl_head.eval()
        return self


def create_prot2mol_model(config):
    """
    Factory function to create a Prot2MolModel.
    
    Args:
        config: Dictionary containing model configuration
        
    Returns:
        Prot2MolModel instance
    """
    return Prot2MolModel(config)


class PairFormerLiteBlock(nn.Module):
    def __init__(self, d_pair, n_heads=4, dropout=0.1):
        super().__init__()
        self.row_attn = nn.MultiheadAttention(d_pair, n_heads, dropout=dropout, batch_first=True)
        self.col_attn = nn.MultiheadAttention(d_pair, n_heads, dropout=dropout, batch_first=True)
        self.ln_row = nn.LayerNorm(d_pair)
        self.ln_col = nn.LayerNorm(d_pair)
        self.ffn = nn.Sequential(
            nn.Linear(d_pair, d_pair * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_pair * 2, d_pair)
        )
        self.ln_ffn = nn.LayerNorm(d_pair)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, prot_mask, mol_mask):
        # z: [B, Lm, Lp, d]
        B, Lm, Lp, d = z.shape

        # Row attention (ligand token attends over protein tokens)
        z_row = z.reshape(B * Lm, Lp, d)
        kpm = None
        if prot_mask is not None:
            kpm = ~(prot_mask.bool())
            kpm = kpm.unsqueeze(1).expand(B, Lm, Lp).reshape(B * Lm, Lp)
        row_out, _ = self.row_attn(z_row, z_row, z_row, key_padding_mask=kpm, need_weights=False)
        z_row = self.ln_row(z_row + self.dropout(row_out))
        z = z_row.reshape(B, Lm, Lp, d)

        # Column attention (protein token attends over ligand tokens)
        z_col = z.permute(0, 2, 1, 3).reshape(B * Lp, Lm, d)
        kpm = None
        if mol_mask is not None:
            kpm = ~(mol_mask.bool())
            kpm = kpm.unsqueeze(1).expand(B, Lp, Lm).reshape(B * Lp, Lm)
        col_out, _ = self.col_attn(z_col, z_col, z_col, key_padding_mask=kpm, need_weights=False)
        z_col = self.ln_col(z_col + self.dropout(col_out))
        z = z_col.reshape(B, Lp, Lm, d).permute(0, 2, 1, 3)

        # Feed-forward
        z = self.ln_ffn(z + self.dropout(self.ffn(z)))
        return z


class AffinityHead(nn.Module):
    def __init__(
        self,
        d_model,
        d_pair=256,
        n_heads=4,
        n_layers=2,
        max_prot_tokens=256,
        attn_bins=16,
        dropout=0.1
    ):
        super().__init__()
        self.d_pair = d_pair
        self.max_prot_tokens = max_prot_tokens
        self.attn_bins = attn_bins
        self.attn_eps = 1e-6
        self.attn_dmax = 8.0

        self.proj_p = nn.Linear(d_model, d_pair)
        self.proj_m = nn.Linear(d_model, d_pair)
        self.ln_p = nn.LayerNorm(d_pair)
        self.ln_m = nn.LayerNorm(d_pair)

        self.bias_p = nn.Linear(d_pair, d_pair, bias=False)
        self.bias_m = nn.Linear(d_pair, d_pair, bias=False)

        self.pair_mlp = nn.Sequential(
            nn.Linear(4 * d_pair, d_pair),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_pair, d_pair)
        )

        self.bin_embed = nn.Embedding(attn_bins, d_pair)

        self.blocks = nn.ModuleList([
            PairFormerLiteBlock(d_pair, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.out = nn.Sequential(
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, d_pair),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_pair, 1)
        )

    def _crop_protein(self, E, A, prot_mask):
        if A is None or self.max_prot_tokens is None:
            return E, A, prot_mask
        B, Lp, d = E.shape
        if Lp <= self.max_prot_tokens:
            return E, A, prot_mask

        if prot_mask is None:
            prot_mask = torch.ones(B, Lp, device=E.device, dtype=torch.float)

        # Importance from attention mass
        importance = A.sum(dim=1)  # [B, Lp]
        importance = importance.masked_fill(~prot_mask.bool(), float("-inf"))
        k = min(self.max_prot_tokens, Lp)
        topk = torch.topk(importance, k=k, dim=1).indices  # [B, k]

        E = torch.gather(E, dim=1, index=topk.unsqueeze(-1).expand(B, k, d))
        prot_mask = torch.gather(prot_mask, dim=1, index=topk)
        A = torch.gather(A, dim=2, index=topk.unsqueeze(1).expand(B, A.size(1), k))
        return E, A, prot_mask

    def forward(self, E, D, A, prot_mask, mol_mask):
        # E: [B, Lp, d_model], D: [B, Lm, d_model], A: [B, Lm, Lp]
        if prot_mask is None:
            prot_mask = torch.ones(E.size(0), E.size(1), device=E.device, dtype=torch.float)
        if mol_mask is None:
            mol_mask = torch.ones(D.size(0), D.size(1), device=D.device, dtype=torch.float)

        E, A, prot_mask = self._crop_protein(E, A, prot_mask)

        Ep = self.ln_p(self.proj_p(E))
        Dm = self.ln_m(self.proj_m(D))

        B, Lm, _ = Dm.shape
        Lp = Ep.shape[1]

        Dm_i = Dm.unsqueeze(2).expand(B, Lm, Lp, self.d_pair)
        Ep_j = Ep.unsqueeze(1).expand(B, Lm, Lp, self.d_pair)
        pair_input = torch.cat(
            [Dm_i, Ep_j, Dm_i * Ep_j, (Dm_i - Ep_j).abs()],
            dim=-1
        )

        z = self.pair_mlp(pair_input)
        z = z + self.bias_m(Dm).unsqueeze(2) + self.bias_p(Ep).unsqueeze(1)

        if A is not None:
            A_clamped = A.clamp(min=self.attn_eps)
            dtilde = (-torch.log(A_clamped)).clamp(max=self.attn_dmax)
            bin_width = self.attn_dmax / self.attn_bins
            bin_idx = torch.clamp((dtilde / bin_width).long(), max=self.attn_bins - 1)
            z = z + self.bin_embed(bin_idx)

        for blk in self.blocks:
            z = blk(z, prot_mask, mol_mask)

        # Attention-weighted pooling
        if A is None:
            w = mol_mask.unsqueeze(2) * prot_mask.unsqueeze(1)
        else:
            w = A * mol_mask.unsqueeze(2) * prot_mask.unsqueeze(1)
        w_sum = w.sum(dim=(1, 2)).clamp(min=1e-6).unsqueeze(-1)
        g = (z * w.unsqueeze(-1)).sum(dim=(1, 2)) / w_sum

        return self.out(g).squeeze(-1)
