import logging
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from .protein_encoders import get_encoder_size, get_protein_encoder



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
        self.logger.info("Initializing FusionDTI-style pChEMBL prediction head")
        hidden_size = encoder_dim
        self.pchembl_head = FusionDTIPChemblHead(
            d_model=hidden_size,
            hidden_dim=self._config.get("pchembl_tf_hidden_dim", 768),
            num_heads=self._config.get("pchembl_tf_num_heads", 8),
            group_size=self._config.get("pchembl_tf_group_size", 1),
            agg_mode=self._config.get("pchembl_tf_agg_mode", "mean"),
            dropout=self._config.get("pchembl_tf_dropout", 0.1),
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

    def encode_protein(self, prot_input_ids, prot_attention_mask):
        """Encode protein tokens once so callers can reuse embeddings across batches."""
        with torch.set_grad_enabled(self.protein_encoder.encoder_model.training):
            return self.protein_encoder.encode(
                sequences=prot_input_ids,
                attention_mask=prot_attention_mask,
            )

    def _prepare_lm_labels(self, labels=None, train_lm=True, pchembl_only_mode=False):
        labels_for_lm = None
        if labels is not None and not pchembl_only_mode:
            if isinstance(train_lm, torch.Tensor):
                labels_for_lm = labels.clone()
                labels_for_lm[~train_lm] = -100
                if (labels_for_lm == -100).all():
                    labels_for_lm = None
            elif train_lm:
                labels_for_lm = labels
        return labels_for_lm

    def _compute_lm_loss(self, logits, labels):
        if labels is None:
            return None
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    def _pairwise_huber_loss(self, y_pred, y_true, group_ids, delta=1.0):
        if group_ids is None:
            return None

        valid_mask = group_ids >= 0
        if not torch.any(valid_mask):
            return None

        y_pred = y_pred[valid_mask]
        y_true = y_true[valid_mask]
        group_ids = group_ids[valid_mask]

        losses = []
        for group_id in torch.unique(group_ids):
            idx = (group_ids == group_id).nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue
            perm = idx[torch.randperm(idx.numel(), device=idx.device)]
            half = perm.numel() // 2
            if half == 0:
                continue
            a = perm[:half]
            b = perm[half:half * 2]
            dy_pred = y_pred[a] - y_pred[b]
            dy_true = y_true[a] - y_true[b]
            losses.append(F.smooth_l1_loss(dy_pred, dy_true, beta=delta))

        if not losses:
            return None
        return torch.stack(losses).mean()

    def _compose_total_loss(self, lm_loss, pchembl_predictions, pchembl_values, group_ids=None, pchembl_pair_weight=1.0):
        pchembl_loss = None
        pair_loss = None
        total_loss = None

        if lm_loss is not None:
            total_loss = self.lm_weight * lm_loss

        if pchembl_predictions is not None and pchembl_values is not None:
            delta = self._config.get("pchembl_huber_delta", 1.0)
            pchembl_loss = F.smooth_l1_loss(pchembl_predictions, pchembl_values, beta=delta)
            pair_loss = self._pairwise_huber_loss(
                pchembl_predictions,
                pchembl_values,
                group_ids,
                delta=delta,
            )
            pair_term = pchembl_pair_weight * (pair_loss if pair_loss is not None else 0.0)
            pchembl_term = self.pchembl_weight * (pchembl_loss + pair_term)
            total_loss = pchembl_term if total_loss is None else total_loss + pchembl_term

        return total_loss, pchembl_loss, pair_loss

    def _decode_from_protein_embeddings(
        self,
        mol_input_ids,
        protein_embeddings,
        prot_attention_mask,
        labels=None,
        pchembl_values=None,
        group_ids=None,
        train_lm=True,
        pchembl_only_mode=False,
        compute_pchembl=True,
        pchembl_pair_weight=1.0,
    ):
        labels_for_lm = self._prepare_lm_labels(
            labels=labels,
            train_lm=train_lm,
            pchembl_only_mode=pchembl_only_mode,
        )

        if hasattr(self.molecule_decoder, "transformer") and hasattr(self.molecule_decoder, "lm_head"):
            with torch.set_grad_enabled(self.molecule_decoder.training):
                decoder_outputs = self.molecule_decoder.transformer(
                    input_ids=mol_input_ids,
                    attention_mask=(mol_input_ids != self._config["mol_tokenizer"].pad_token_id).float(),
                    encoder_hidden_states=protein_embeddings.detach() if pchembl_only_mode else protein_embeddings,
                    encoder_attention_mask=prot_attention_mask,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )

            hidden_states = decoder_outputs.last_hidden_state
            logits = self.molecule_decoder.lm_head(hidden_states)
            lm_loss = self._compute_lm_loss(logits, labels_for_lm)
        else:
            with torch.set_grad_enabled(self.molecule_decoder.training):
                decoder_outputs = self.molecule_decoder(
                    input_ids=mol_input_ids,
                    attention_mask=(mol_input_ids != self._config["mol_tokenizer"].pad_token_id).float(),
                    encoder_hidden_states=protein_embeddings.detach() if pchembl_only_mode else protein_embeddings,
                    encoder_attention_mask=prot_attention_mask,
                    labels=labels_for_lm,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True,
                )
            hidden_states = decoder_outputs.hidden_states[-1]
            logits = decoder_outputs.logits
            lm_loss = decoder_outputs.loss

        mol_attention_mask = (mol_input_ids != self._config["mol_tokenizer"].pad_token_id).float()
        pchembl_predictions = None
        should_run_pchembl = compute_pchembl and self._config.get("train_pchembl_head", True)
        if should_run_pchembl:
            stop_pchembl_gradients = self._config.get("stop_pchembl_gradients", True)
            block_backprop = pchembl_only_mode or stop_pchembl_gradients

            head_prot = protein_embeddings.detach() if block_backprop else protein_embeddings
            head_mol = hidden_states.detach() if block_backprop else hidden_states
            pchembl_predictions = self.pchembl_head(
                head_prot,
                head_mol,
                prot_attention_mask,
                mol_attention_mask,
            )

        total_loss, pchembl_loss, pair_loss = self._compose_total_loss(
            lm_loss=lm_loss,
            pchembl_predictions=pchembl_predictions,
            pchembl_values=pchembl_values,
            group_ids=group_ids,
            pchembl_pair_weight=pchembl_pair_weight,
        )

        outputs = {
            "lm_loss": lm_loss,
            "logits": logits,
            "pchembl_predictions": pchembl_predictions,
            "pchembl_loss": pchembl_loss,
            "pchembl_pair_loss": pair_loss,
            "last_hidden_state": hidden_states,
            "loss": total_loss,
        }
        return {key: value for key, value in outputs.items() if value is not None}

    def forward(
        self,
        mol_input_ids,
        prot_input_ids,
        prot_attention_mask,
        labels=None,
        pchembl_values=None,
        group_ids=None,
        train_lm=True,
        pchembl_only_mode=False,
        compute_pchembl=True,
        pchembl_pair_weight=1.0,
    ):
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
        protein_embeddings = self.encode_protein(prot_input_ids, prot_attention_mask)
        return self._decode_from_protein_embeddings(
            mol_input_ids=mol_input_ids,
            protein_embeddings=protein_embeddings,
            prot_attention_mask=prot_attention_mask,
            labels=labels,
            pchembl_values=pchembl_values,
            group_ids=group_ids,
            train_lm=train_lm,
            pchembl_only_mode=pchembl_only_mode,
            compute_pchembl=compute_pchembl,
            pchembl_pair_weight=pchembl_pair_weight,
        )
    
    def predict_pchembl_from_protein_embeddings(self, mol_input_ids, protein_embeddings, prot_attention_mask):
        outputs = self._decode_from_protein_embeddings(
            mol_input_ids=mol_input_ids,
            protein_embeddings=protein_embeddings,
            prot_attention_mask=prot_attention_mask,
            labels=None,
            train_lm=False,
            pchembl_only_mode=False,
            compute_pchembl=True,
        )
        return outputs["pchembl_predictions"]

    def generate_from_protein_embeddings(self, protein_embeddings, prot_attention_mask, **generation_kwargs):
        """Generate molecules from cached protein embeddings."""
        return self.molecule_decoder.generate(
            encoder_hidden_states=protein_embeddings,
            encoder_attention_mask=prot_attention_mask,
            **generation_kwargs,
        )

    def generate(self, prot_input_ids, prot_attention_mask, protein_embeddings=None, **generation_kwargs):
        """
        Generate molecule sequences given protein sequences.
        
        Args:
            prot_input_ids: Tokenized protein sequences
            prot_attention_mask: Attention mask for protein sequences
            **generation_kwargs: Additional arguments for generation
            
        Returns:
            Generated molecule token sequences
        """
        if protein_embeddings is None:
            protein_embeddings = self.encode_protein(prot_input_ids, prot_attention_mask)
        return self.generate_from_protein_embeddings(
            protein_embeddings=protein_embeddings,
            prot_attention_mask=prot_attention_mask,
            **generation_kwargs,
        )

    def corr_loss_calculation(self, pchembl_predictions, pchembl_values, eps=1e-8):
        x = pchembl_predictions - pchembl_predictions.mean(); v = pchembl_values - pchembl_values.mean()
        return 1 - (x*v).mean() / (x.pow(2).mean().sqrt()*v.pow(2).mean().sqrt()+eps)
    
    def get_encoder_hidden_states(self, prot_input_ids, prot_attention_mask):
        """
        Get the hidden states from the protein encoder.
        """
        return self.encode_protein(prot_input_ids, prot_attention_mask)

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


class FusionDTITokenFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        self.query_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_p = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.query_m = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_m = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_m = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _apply_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_size)

    def _masked_softmax(self, logits: torch.Tensor, row_mask: torch.Tensor, col_mask: torch.Tensor) -> torch.Tensor:
        valid_pairs = row_mask.unsqueeze(2).unsqueeze(-1) & col_mask.unsqueeze(1).unsqueeze(-1)
        mask_fill_value = torch.finfo(logits.dtype).min
        masked_logits = torch.where(valid_pairs, logits, torch.full_like(logits, mask_fill_value))
        alpha = torch.softmax(masked_logits, dim=2)
        return torch.where(valid_pairs, alpha, torch.zeros_like(alpha))

    def forward(
        self,
        protein_tokens: torch.Tensor,
        molecule_tokens: torch.Tensor,
        protein_mask: torch.Tensor,
        molecule_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        protein_q = self._apply_heads(self.query_p(protein_tokens))
        protein_k = self._apply_heads(self.key_p(protein_tokens))
        protein_v = self._apply_heads(self.value_p(protein_tokens))

        molecule_q = self._apply_heads(self.query_m(molecule_tokens))
        molecule_k = self._apply_heads(self.key_m(molecule_tokens))
        molecule_v = self._apply_heads(self.value_m(molecule_tokens))

        logits_pp = torch.einsum("blhd,bkhd->blkh", protein_q, protein_k) / math.sqrt(self.head_size)
        logits_pm = torch.einsum("blhd,bkhd->blkh", protein_q, molecule_k) / math.sqrt(self.head_size)
        logits_mp = torch.einsum("blhd,bkhd->blkh", molecule_q, protein_k) / math.sqrt(self.head_size)
        logits_mm = torch.einsum("blhd,bkhd->blkh", molecule_q, molecule_k) / math.sqrt(self.head_size)

        alpha_pp = self._masked_softmax(logits_pp, protein_mask, protein_mask)
        alpha_pm = self._masked_softmax(logits_pm, protein_mask, molecule_mask)
        alpha_mp = self._masked_softmax(logits_mp, molecule_mask, protein_mask)
        alpha_mm = self._masked_softmax(logits_mm, molecule_mask, molecule_mask)

        fused_protein = (
            torch.einsum("blkh,bkhd->blhd", alpha_pp, protein_v).flatten(-2)
            + torch.einsum("blkh,bkhd->blhd", alpha_pm, molecule_v).flatten(-2)
        ) / 2.0
        fused_molecule = (
            torch.einsum("blkh,bkhd->blhd", alpha_mp, protein_v).flatten(-2)
            + torch.einsum("blkh,bkhd->blhd", alpha_mm, molecule_v).flatten(-2)
        ) / 2.0

        fused_protein = fused_protein * protein_mask.unsqueeze(-1)
        fused_molecule = fused_molecule * molecule_mask.unsqueeze(-1)
        return fused_protein, fused_molecule


class FusionDTIRegressionMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float):
        super().__init__()
        hidden_mid = max(input_dim // 2, 1)
        hidden_low = max(input_dim // 4, 1)
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.ln1 = nn.LayerNorm(input_dim)
        self.fc2 = nn.Linear(input_dim, hidden_mid)
        self.ln2 = nn.LayerNorm(hidden_mid)
        self.fc3 = nn.Linear(hidden_mid, hidden_low)
        self.ln3 = nn.LayerNorm(hidden_low)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_low, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.ln1(F.gelu(self.fc1(x))))
        x = self.dropout(self.ln2(F.gelu(self.fc2(x))))
        x = self.dropout(self.ln3(F.gelu(self.fc3(x))))
        return self.output(x).squeeze(-1)


class FusionDTIPChemblHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 768,
        num_heads: int = 8,
        group_size: int = 1,
        agg_mode: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        if agg_mode not in {"cls", "mean", "mean_all_tok"}:
            raise ValueError(f"Unsupported agg_mode: {agg_mode}")

        self.group_size = group_size
        self.agg_mode = agg_mode
        self.proj_p = nn.Linear(d_model, hidden_dim)
        self.proj_m = nn.Linear(d_model, hidden_dim)
        self.ln_p = nn.LayerNorm(hidden_dim)
        self.ln_m = nn.LayerNorm(hidden_dim)
        self.fusion = FusionDTITokenFusion(hidden_dim=hidden_dim, num_heads=num_heads)
        self.regression = FusionDTIRegressionMLP(input_dim=hidden_dim * 2, dropout=dropout)

    def _normalize_mask(self, mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return torch.ones(x.size(0), x.size(1), device=x.device, dtype=torch.bool)
        return mask.to(device=x.device).bool()

    def _group_embeddings(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.group_size == 1:
            return x, mask

        batch_size, seq_len, hidden_dim = x.shape
        pad_len = (-seq_len) % self.group_size
        if pad_len:
            x = torch.cat([x, torch.zeros(batch_size, pad_len, hidden_dim, device=x.device, dtype=x.dtype)], dim=1)
            mask = torch.cat([mask, torch.zeros(batch_size, pad_len, device=mask.device, dtype=mask.dtype)], dim=1)

        grouped_len = x.size(1) // self.group_size
        x_grouped = x.reshape(batch_size, grouped_len, self.group_size, hidden_dim)
        mask_grouped = mask.reshape(batch_size, grouped_len, self.group_size)

        counts = mask_grouped.sum(dim=2, keepdim=True).clamp(min=1)
        grouped_embeddings = (x_grouped * mask_grouped.unsqueeze(-1)).sum(dim=2) / counts
        grouped_mask = mask_grouped.any(dim=2)
        return grouped_embeddings, grouped_mask

    def _aggregate(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.agg_mode == "cls":
            return tokens[:, 0]
        if self.agg_mode == "mean_all_tok":
            return tokens.mean(dim=1)

        weights = mask.unsqueeze(-1).to(tokens.dtype)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (tokens * weights).sum(dim=1) / denom

    def forward(
        self,
        protein_embeddings: torch.Tensor,
        molecule_embeddings: torch.Tensor,
        prot_mask: torch.Tensor,
        mol_mask: torch.Tensor,
    ) -> torch.Tensor:
        prot_mask = self._normalize_mask(prot_mask, protein_embeddings)
        mol_mask = self._normalize_mask(mol_mask, molecule_embeddings)

        protein_tokens = self.ln_p(self.proj_p(protein_embeddings))
        molecule_tokens = self.ln_m(self.proj_m(molecule_embeddings))

        protein_tokens, prot_mask = self._group_embeddings(protein_tokens, prot_mask)
        molecule_tokens, mol_mask = self._group_embeddings(molecule_tokens, mol_mask)

        fused_protein, fused_molecule = self.fusion(
            protein_tokens=protein_tokens,
            molecule_tokens=molecule_tokens,
            protein_mask=prot_mask,
            molecule_mask=mol_mask,
        )

        pooled_protein = self._aggregate(fused_protein, prot_mask)
        pooled_molecule = self._aggregate(fused_molecule, mol_mask)
        joint_embedding = torch.cat([pooled_protein, pooled_molecule], dim=-1)
        return self.regression(joint_embedding)
