from transformers import Trainer
import torch
class GPT2_w_crs_attn_Trainer(Trainer):
    def __init__(self, encoder_model=None, train_encoder_model=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_encoder_model = train_encoder_model
        self.encoder_model = encoder_model
        
    def compute_loss(self, model, inputs, return_outputs=False):
        input_sequence = inputs["mol_input_ids"]
        
        # Use the encoder's encode method
        last_hidden_state = self.encoder_model.encode(
            sequences=inputs["prot_input_ids"], 
            attention_mask=inputs["prot_attention_mask"]
        )

        outputs = model(
            input_ids=input_sequence, 
            encoder_hidden_states=last_hidden_state, 
            labels=input_sequence
        )  

        return (outputs.loss, outputs) if return_outputs else outputs.loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model_inputs = {
            "mol_input_ids": inputs["mol_input_ids"],
            "prot_input_ids": inputs["prot_input_ids"],
            "prot_attention_mask": inputs["prot_attention_mask"],
            "labels": inputs["mol_input_ids"] if "labels" not in inputs else inputs["labels"]
        }

        return super().prediction_step(
            model, 
            model_inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys
        )