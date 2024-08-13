from transformers import Trainer

class GPT2_w_crs_attn_Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        
        input_sequence = inputs["input_ids"]
        last_hidden_state = inputs["encoder_hidden_states"]
        outputs = model(input_ids=input_sequence, encoder_hidden_states=last_hidden_state, labels=input_sequence)
        print(input_sequence,last_hidden_state )
        return (outputs.loss, outputs) if return_outputs else outputs.loss