from transformers import BartTokenizer, GPT2Config, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
import os 
import math
from utils import metrics_calculation
import numpy as np
from transformers import DataCollatorForLanguageModeling
import selfies as sf
import pandas as pd
import argparse
import torch 
import torch.distributed
import sys
sys.path.insert(1, '../data_processing')
import train_val_test
from data_loader import CustomDataset
from gpt2_trainer import GPT2_w_crs_attn_Trainer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "false"

class FinetuneScript(object):
    
    """(FineTune)Trainer for training and testing Prot2Mol."""

    def __init__(self, config,
                       selfies_path, 
                       finetune_save_to,
                       model_name,
                       dataset_name,
                       run_name):
        
        self.selfies_path = selfies_path
        self.finetune_save_to = finetune_save_to
        self.model_name = model_name
        self.prot_emb_model = config.prot_emb_model
        self.run_name = run_name

        self.TRAIN_BATCH_SIZE = config.train_batch_size
        self.VALID_BATCH_SIZE = config.valid_batch_size
        self.TRAIN_EPOCHS = config.epoch
        self.LEARNING_RATE = config.learning_rate
        self.WEIGHT_DECAY = config.weight_decay
        self.N_LAYER = config.n_layer

        self.training_vec = np.load("../data/train_vecs.npy")
        
        self.train_data, self.eval_data, self.test_data = train_val_test.train_val_test_split(config.selfies_path, config.prot_ID)

        if "af2" in self.prot_emb_model:
            self.prot_emb_model_path = f"../data/prot_embed/{self.prot_emb_model}/FoldedPapyrus_4581_v01/embeddings"
        else:
            prot_emb_model_path = f"../data/prot_embed/{self.prot_emb_model}/{dataset_name}/embeddings.npz"
        
        self.target_data = np.load(prot_emb_model_path, mmap_mode='r')
        self.N_EMBED = np.array(self.target_data["encoder_hidden_states"][0]).shape[-1]
        
        self.tokenizer = BartTokenizer.from_pretrained("zjunlp/MolGen-large", padding_side="left")    
        alphabet =  list(sf.get_alphabet_from_selfies(list(self.train_data.Compound_SELFIES)))
        self.tokenizer.add_tokens(alphabet)
        alphabet =  list(sf.get_alphabet_from_selfies(list(self.eval_data.Compound_SELFIES)))
        self.tokenizer.add_tokens(alphabet)
        del alphabet

        print("Loading model from:", self.model_name)
            
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        print("Model parameter count:", self.model.num_parameters())

    def compute_metrics(self, eval_pred):
        if torch.distributed.get_rank() == 0:
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            return metrics_calculation(predictions=decoded_preds, references=decoded_labels, train_data = self.train_data, train_vec = self.training_vec)
        else: 
            return {}
        
    def finetune_with_target(self, target_id):
        
        finetune_data = self.test_data
        self.alphabet =  list(sf.get_alphabet_from_selfies(list(finetune_data.Compound_SELFIES)))
        self.tokenizer.add_tokens(self.alphabet)
        
        finetune_dataset = CustomDataset(ligand_data=finetune_data, target_data=self.target_data, tokenizer=self.tokenizer, max_length=200)
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        
        self.model.train()
        
        finetune_args = TrainingArguments(
            run_name=self.run_name,
            output_dir=self.finetune_save_to,
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
            dataloader_num_workers=4,
            fp16=True,
            ddp_find_unused_parameters=False)

        finetune_trainer = GPT2_w_crs_attn_Trainer(
            model=self.model,
            args=finetune_args,
            train_dataset=finetune_dataset,
            eval_dataset=finetune_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
            # prediction_loss_only=True,
        )
        finetune_trainer.args._n_gpu = 1
        
        print("build finetuning with on device:", finetune_args.device, "with n gpus:", finetune_args.n_gpu)
        
        finetune_trainer.train()
        print("training finished.")

        eval_results = finetune_trainer.evaluate()
        print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        finetune_trainer.save_model(self.finetune_save_to)      
 
def main(config):
    run_name = f"""TARGET_{str(config.target_id)}_lr_{str(config.learning_rate)}_bs_{str(config.train_batch_size)}_ep_{str(config.epoch)}_wd_{str(config.weight_decay)}_nlayer_{str(config.n_layer)}"""
    dataset_name = config.selfies_path.split("/")[-1].split(".")[0]
    
    trainingscript = FinetuneScript(config=config, 
                                    selfies_path=config.selfies_path, 
                                    finetune_save_to=f"./finetuned_models/{run_name}",
                                    model_name = config.pretrained_model_path,
                                    dataset_name=dataset_name,
                                    run_name=run_name)
    
    trainingscript.finetune_with_target(config.target_id)
    
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Dataset parameters
    
    parser.add_argument("--selfies_path", required=False, default="../data/fasta_to_selfies_500.csv", help="Path of the SELFIES dataset.")
    parser.add_argument("--target_id", default="CHEMBL4282", help="Target ID (ChEMBL_ID) to finetune on.")
    parser.add_argument("--prot_emb_model", required=False, default="prot_t5", help="Which protein embedding model to use", choices=["prot_t5", "esm2", "esm3", "af2_single", "af2_struct", "af2_combined"])
    # Model parameters
    
    parser.add_argument("--pretrained_model_path", default="./saved_models/set_100_saved_model/checkpoint-31628")
    
    parser.add_argument("--learning_rate", default=1.0e-5)
    parser.add_argument("--max_mol_len", default=200)
    parser.add_argument("--train_batch_size", default=64)
    parser.add_argument("--valid_batch_size", default=64)
    parser.add_argument("--epoch", default=50)
    parser.add_argument("--weight_decay", default=0.0005)
    parser.add_argument("--max_positional_emb", default=202)
    parser.add_argument("--n_layer", default=4)

    
    config = parser.parse_args()
    main(config)

