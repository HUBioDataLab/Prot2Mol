import os
import json
import tqdm
import torch
import argparse
import warnings
import pandas as pd
from utils import *
import selfies as sf
from rdkit import RDLogger    
RDLogger.DisableLog('rdApp.*')  
warnings.filterwarnings("ignore")
from datasets import load_from_disk
from transformers import GenerationConfig
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers.utils import logging
logging.set_verbosity_error() 
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_dataset(path, prot_emb_model, prot_id):
    
    print("Loading required data")
    train_data = pd.read_csv(f"{path}/train.csv")
    alphabet =  list(sf.get_alphabet_from_selfies(list(train_data.Compound_SELFIES)))
    tokenizer.add_tokens(alphabet)
    train_vec = np.load(f"{path}/train_vecs.npy")

    eval_data = pd.read_csv(f"{path}/eval.csv")
    alphabet =  list(sf.get_alphabet_from_selfies(list(eval_data.Compound_SELFIES)))
    tokenizer.add_tokens(alphabet)  
    del eval_data
    
    test_data = pd.read_csv(f"{path}/test_{prot_id}.csv")
    alphabet =  list(sf.get_alphabet_from_selfies(list(test_data.Compound_SELFIES)))
    tokenizer.add_tokens(alphabet)
    
    
    target_data = load_from_disk(prot_emb_model)
    selected_target = test_data[test_data["Target_CHEMBL_ID"].isin([args.prot_id])].reset_index(drop=True)
    
    return train_data, train_vec, target_data, selected_target


def get_target(target, target_id):

    enc_state = target[target["Target_CHEMBL_ID"].index(target_id)]["encoder_hidden_states"]
    sample = {"encoder_hidden_states": enc_state, "target_chembl_id": target_id}
    return sample 

def generate_molecules(data):
    generated_tokens = model.generate(encoder_hidden_states=data,
                            num_return_sequences=args.bs,
                            do_sample=True,
                            max_length=200,
                            pad_token_id=1,
                            output_attentions = True if args.attn_output else False)
    
    generated_selfies = [tokenizer.decode(x, skip_special_tokens=True) for x in generated_tokens]
    return generated_selfies

print("Starting to generate molecules.")

def generation_loop(target_data, num_samples, bs):

    gen_mols = []
    sample = get_target(target_data, args.prot_id)["encoder_hidden_states"].view(1,-1,1024).to("cuda:0")

    for _ in tqdm.tqdm(range(int(num_samples/bs))):
        gen_mols.extend(generate_molecules(sample))
        
    gen_mols_df = pd.DataFrame(gen_mols, columns=["Generated_SELFIES"])
    
    return gen_mols_df

print("Metrics are being calculated.")

def calc_metrics(dataset, prot_emb_model, prot_id, num_samples, bs, generated_mol_file):

    train_data, train_vec, target_data, selected_target = load_dataset(dataset, prot_emb_model=prot_emb_model, prot_id=prot_id)
    
    gen_mols_df = generation_loop(target_data, num_samples, bs)
    
    metrics, generated_smiles = metrics_calculation(predictions=gen_mols_df["Generated_SELFIES"], 
                                references=selected_target["Compound_SELFIES"], 
                                train_data = train_data, 
                                train_vec = train_vec,
                                training=False)
    print(metrics)

    gen_mols_df["smiles"] = generated_smiles
    gen_mols_df.to_csv(generated_mol_file, index=False)
    with open(generated_mol_file.replace(".csv", "_metrics.json"), "w") as f:
        json.dump(metrics, f)
    print("Molecules and metrics are saved.")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", default="./finetuned_models/set_100_finetuned_model/checkpoint-3100", help="Path of the pretrained model file.")
    parser.add_argument("--prot_emb_model", default="./data/prot_embed/prot_t5/prot_comp_set_pchembl_None_protlen_None/embeddings", help="Path of the pretrained model file.")
    parser.add_argument("--generated_mol_file", default="./saved_mols/_kt_finetune_mols.csv", help="Path of the output embeddings file.")
    parser.add_argument("--selfies_path", default='./data/papyrus/prot_comp_set_pchembl_None_protlen_500_human_False', help="Path of the input SEFLIES dataset.")
    parser.add_argument("--attn_output", default=False, help="Path of the output embeddings file.")
    parser.add_argument("--prot_id", default="CHEMBL4282", help="Target Protein ID.")
    parser.add_argument("--num_samples", default=10000, help="Sample number.")
    parser.add_argument("--bs", default=100, help="Batch size.")
    args = parser.parse_args()
    
    genearted_mol_file_path = f"""./saved_mols/
                                    {args.selfies_path.split("/")[2]}_
                                    {args.prot_id}_
                                    {args.model_file.split("/")[2]}_
                                    {args.prot_emb_model.split("/")[3]}/
                                    {args.num_samples}_mols.csv"""
    # Load tokenizer and the model
    print("Loading model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("zjunlp/MolGen-large", padding_side="left") # we can convert this to our own tokenizer later.
    model_name = args.model_file
    model = GPT2LMHeadModel.from_pretrained(model_name).to("cuda:0")
    generation_config = GenerationConfig.from_pretrained(model_name)
    model.config.bos_token_id = 1
    
    calc_metrics(args.selfies_path, args.prot_emb_model, args.prot_id, args.num_samples, args.bs, genearted_mol_file_path)