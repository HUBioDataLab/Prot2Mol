import os 
import sys
import math
import argparse
import numpy as np
import selfies as sf
from utils import metrics_calculation
from transformers import TrainingArguments
from transformers import BartTokenizer, GPT2Config, GPT2LMHeadModel
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing import train_val_test
import torch.distributed
from gpt2_trainer import GPT2_w_crs_attn_Trainer
from transformers import T5Tokenizer, T5EncoderModel
import re
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "false"

class TrainingScript(object):
    
    """(Pre)Trainer for training and testing Prot2Mol."""

    def __init__(self, config,
                       selfies_path, 
                       pretrain_save_to,
                       dataset_name,
                       run_name):
        
        self.selfies_path = selfies_path
        self.pretrain_save_to = pretrain_save_to
        self.prot_emb_model = config.prot_emb_model
        self.run_name = run_name
        
        self.TRAIN_BATCH_SIZE = config.train_batch_size
        self.VALID_BATCH_SIZE = config.valid_batch_size
        self.TRAIN_EPOCHS = config.epoch
        self.LEARNING_RATE = config.learning_rate
        self.WEIGHT_DECAY = config.weight_decay
        self.N_LAYER = config.n_layer
        self.N_HEAD = config.n_head
        self.max_mol_len = config.max_mol_len
        self.N_EMB = config.n_emb
        self.train_encoder_model = config.train_encoder_model
        self.prot_max_length = config.prot_max_length
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        train_vecs_path = os.path.join(data_dir, "train_vecs.npy")
        
        if not os.path.exists(train_vecs_path):
            print(f"Warning: Training vectors file not found at {train_vecs_path}")
            print("You need to generate train_vecs.npy to calculate training similarity")

            # OR
            self.training_vec = None  
        else:
            self.training_vec = np.load(train_vecs_path)

        self.mol_tokenizer = BartTokenizer.from_pretrained("zjunlp/MolGen-large", padding_side="left")
        self.prot_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False, legacy=True, clean_up_tokenization_spaces=True)
        
        print("Load GPT2 model...\n")
        self.configuration = GPT2Config(add_cross_attention=True, is_decoder = True,
                                n_embd=self.N_EMB, n_head=self.N_HEAD, vocab_size=len(self.mol_tokenizer.added_tokens_decoder), 
                                n_positions=256, n_layer=self.N_LAYER, bos_token_id=self.mol_tokenizer.bos_token_id,
                                eos_token_id=self.mol_tokenizer.eos_token_id)
        self.model = GPT2LMHeadModel(self.configuration)
        self.encoder_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", torch_dtype=torch.float16)

        
        print("Model parameter count:", self.model.num_parameters())

    def tokenize_prot_function(self, batch):
        sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in batch["Target_FASTA"]]
        ids = self.prot_tokenizer.batch_encode_plus(
            sequence_examples, 
            add_special_tokens=True, 
            truncation=True,
            max_length=self.prot_max_length,
            padding="max_length"

        )
        return {
            'prot_input_ids': ids['input_ids'],
            'prot_attention_mask': ids['attention_mask']
        }

    def tokenize_mol_function(self, batch):
        ids = self.mol_tokenizer.batch_encode_plus(
            batch["Compound_SELFIES"], 
            add_special_tokens=True, 
            truncation=True,
            max_length=self.max_mol_len,
            padding="max_length"
        )
        return {
            'mol_input_ids': ids['input_ids'],
            'mol_attention_mask': ids['attention_mask']
        }

    def compute_metrics(self, eval_pred):
        #if torch.distributed.get_rank() == 0:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        labels = np.where(labels != -100, labels, self.mol_tokenizer.pad_token_id)
        decoded_preds = self.mol_tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.mol_tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        return metrics_calculation(predictions=decoded_preds, references=decoded_labels, train_data = self.train_data, train_vec = self.training_vec)
        #else: 
            #return {}

    def model_training(self):
        
        dataset = load_dataset("csv", data_files=self.selfies_path)
        dataset = dataset.map(
                                self.tokenize_prot_function,
                                batched=True,
                                num_proc=1,
                                batch_size=100,
                                desc="Tokenizing protein sequences"
                            )
        dataset = dataset.map(
                                self.tokenize_mol_function,
                                batched=True,
                                num_proc=1,
                                batch_size=100,
                                desc="Tokenizing molecule sequences"
                            )
        dataset = dataset["train"].train_test_split(test_size=0.1)
        self.train_data = dataset["train"]
        self.test_data = dataset["test"]
        self.model.train()
        if self.train_encoder_model:
            self.encoder_model.train()
        else:
            self.encoder_model.eval()
        
        training_args = TrainingArguments(
            run_name=self.run_name,
            output_dir=self.pretrain_save_to,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=self.TRAIN_EPOCHS,
            learning_rate=self.LEARNING_RATE,
            weight_decay=self.WEIGHT_DECAY,
            per_device_train_batch_size=self.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=self.VALID_BATCH_SIZE,
            save_total_limit=1,
            disable_tqdm=True,
            logging_steps=10,
            dataloader_num_workers=10,
            fp16=True,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False)
        self.encoder_model.to(training_args.device)
        trainer = GPT2_w_crs_attn_Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.test_data,
            compute_metrics=self.compute_metrics,
            encoder_model=self.encoder_model,
            train_encoder_model=self.train_encoder_model
        )
        
        trainer.args._n_gpu = 1
        
        print("build pretrain trainer with on device:", training_args.device, "with n gpus:", training_args.n_gpu)
        trainer.train()
        print("training finished.")

        eval_results = trainer.evaluate()
        print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        trainer.save_model(self.pretrain_save_to) 

def main(config):
    # Convert relative path to absolute path
    if config.selfies_path.startswith("../"):
        config.selfies_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            config.selfies_path[3:]  # Remove "../" prefix
        )
    
    # Verify file exists
    if not os.path.exists(config.selfies_path):
        raise FileNotFoundError(f"Data file not found at: {config.selfies_path}")

    dataset_name = os.path.splitext(os.path.basename(config.selfies_path))[0]
    
    if config.full_set:
        run_name =  f"""lr_{str(config.learning_rate)}_bs_{str(config.train_batch_size)}_ep_{str(config.epoch)}_wd_{str(config.weight_decay)}_nlayer_{str(config.n_layer)}_nhead_{str(config.n_head)}_prot_{config.prot_emb_model}_dataset_{dataset_name}_fp16"""
    else:
        run_name =  f"""lr_{str(config.learning_rate)}_bs_{str(config.train_batch_size)}_ep_{str(config.epoch)}_wd_{str(config.weight_decay)}_nlayer_{str(config.n_layer)}_nhead_{str(config.n_head)}_prot_{config.prot_emb_model}_dataset_{dataset_name}_testID_{config.prot_ID}"""
    
    trainingscript = TrainingScript(config=config, 
                                    selfies_path=config.selfies_path, 
                                    pretrain_save_to=f"../saved_models/{run_name}",
                                    dataset_name = dataset_name,
                                    run_name=run_name)
    
    trainingscript.model_training()
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Update default path to use absolute path
    default_data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data/papyrus/papyrus_100data.csv"
    )
    
    parser.add_argument("--selfies_path", required=False, default=default_data_path, help="Path of the SELFIES dataset.")
    parser.add_argument("--prot_emb_model", required=False, default="prot_t5", help="Which protein embedding model to use", choices=["prot_t5", "esm2", "esm3", "af2_single", "af2_struct", "af2_combined"])
    parser.add_argument("--prot_ID", required=False, default="CHEMBL4282")
    parser.add_argument("--full_set", required=False, default=True, help="Use full dataset.")
    parser.add_argument("--train_encoder_model", required=False, default=False, help="Train encoder model.")
    # Model parameters
    parser.add_argument("--learning_rate", default=1.0e-5)
    parser.add_argument("--prot_max_length", default=1000)
    parser.add_argument("--max_mol_len", default=200)
    parser.add_argument("--train_batch_size", default=2)
    parser.add_argument("--valid_batch_size", default=2)
    parser.add_argument("--epoch", default=50)
    parser.add_argument("--weight_decay", default=0.0005)
    parser.add_argument("--max_positional_emb", default=202)
    parser.add_argument("--n_layer", default=4)
    parser.add_argument("--n_head", default=16)
    parser.add_argument("--n_emb", default=1024) # prot_t5=1024, esm2=1280, esm3=1024, af2_single=384, af2_struct=384, af2_combined=768
    
    
    config = parser.parse_args()
    main(config)


              