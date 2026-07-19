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
from transformers import GenerationConfig
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers.utils import logging
import re
import numpy as np
import multiprocessing as mp
from protein_encoders import get_protein_encoder, get_protein_tokenizer
from utils_fps import generate_morgan_fingerprints_parallel

logging.set_verbosity_error() 
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def get_fasta(dataset, prot_id):
    test_df = pd.read_csv(dataset)
    fasta = test_df.loc[test_df.Target_CHEMBL_ID == prot_id, "Target_FASTA"].iloc[0]
    fasta = re.sub(r"[UZOB]", "X", fasta)

    return fasta
    
def prepare_protein_train_vecs(selfies_csv, prot_id):
    df = pd.read_csv(selfies_csv)
    df = df[df["Target_CHEMBL_ID"] == prot_id].reset_index(drop=True)

    selfies_list = df["Compound_SELFIES"].tolist()
    smiles_list = [sf.decoder(s) for s in selfies_list]

    fps = generate_morgan_fingerprints_parallel(
                smiles_list,
                radius=2,
                nBits=1024,
                n_jobs= min(mp.cpu_count()-1, 10)
            )
    np.save("vec_for_prod.npy", fps)
    print("vector file saved successfully")
    return df, fps

def generate_molecules(data):
    generated_tokens = model.generate(generation_config=generation_config,
                            encoder_hidden_states=data,
                            num_return_sequences=args.bs,
                            do_sample=True,
                            max_length=200,
                            pad_token_id=tokenizer.pad_token_id,
                            output_attentions = True if args.attn_output else False)
    
    generated_selfies = [tokenizer.decode(x, skip_special_tokens=True) for x in generated_tokens]
    return generated_selfies

print("Starting to generate molecules.")

def generation_loop(encoder_hidden, num_samples, bs):

    gen_mols = []

    for _ in tqdm.tqdm(range(int(num_samples/bs))):
        gen_mols.extend(generate_molecules(encoder_hidden))
        
    gen_mols_df = pd.DataFrame(gen_mols, columns=["Generated_SELFIES"])
    
    return gen_mols_df

print("Metrics are being calculated.")

def calc_metrics(train_df, train_vec, selected_target_df, gen_mols_df, generated_mol_file):
    #train df tüm dataset olcak
    #gen_mols_df = generation_loop(target_data, num_samples, bs)
    
    metrics, results_df = metrics_calculation(predictions=gen_mols_df["Generated_SELFIES"], 
                                references=selected_target_df["Compound_SELFIES"], 
                                train_data = train_df, 
                                train_vec = train_vec,
                                training=False)
    print(metrics)

    gen_mols_df["smiles"] = results_df["smiles"]
    gen_mols_df.to_csv(generated_mol_file, index=False)
    
    results_df.to_csv(generated_mol_file.replace(".csv", "_per_sample_results.csv"), index = False)

    with open(generated_mol_file.replace(".csv", "_metrics.json"), "w") as f:
        json.dump(metrics, f)
    print("Molecules and metrics are saved.")

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", default="./finetuned_models/set_100_finetuned_model/checkpoint-3100", help="Path of the pretrained model file.")
    parser.add_argument("--prot_emb_model", default="saprot", help="Encoder selection: prot_t5, esm2, saprot")
    parser.add_argument("--generated_mol_file", default="./saved_mols/_kt_finetune_mols.csv", help="Path of the output embeddings file.")
    parser.add_argument("--selfies_path", default='./data/papyrus/prot_comp_set_pchembl_None_protlen_500_human_False', help="Path of the input SEFLIES dataset.")
    parser.add_argument("--attn_output", default=False, help="Path of the output embeddings file.")
    parser.add_argument("--prot_id", default="CHEMBL4282", help="Target Protein ID.")
    parser.add_argument("--num_samples", type = int, default=10000, help="Sample number.")
    parser.add_argument("--bs", type = int, default=100, help="Batch size.")
    args = parser.parse_args()
                     
    
    
    # Load tokenizer and the model
    print("Loading model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("zjunlp/MolGen-large", padding_side="left") # we can convert this to our own tokenizer later.
    model_name = args.model_file
    model = GPT2LMHeadModel.from_pretrained(model_name).to("cuda:0")
    generation_config = GenerationConfig.from_pretrained(model_name)
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    encoder = get_protein_encoder(args.prot_emb_model, active = False)
    prot_tokenizer = get_protein_tokenizer(args.prot_emb_model)

    fasta = get_fasta(args.selfies_path, args.prot_id)

    enc_inputs = prot_tokenizer(fasta, return_tensors="pt", truncation = True,
                                 padding = "max_length", max_length = encoder.max_length)
    with torch.no_grad():
        encoder_hidden = encoder.encode(sequences=enc_inputs["input_ids"],
                                         attention_mask = enc_inputs["attention_mask"]).to(model.device)

    prot_df, train_vec = prepare_protein_train_vecs(args.selfies_path, args.prot_id)

    gen_mols_df = generation_loop(encoder_hidden, args.num_samples, args.bs)
    
    calc_metrics(train_df = pd.read_csv(args.selfies_path), train_vec = train_vec, selected_target_df=prot_df,
                 gen_mols_df=gen_mols_df, generated_mol_file=args.generated_mol_file)
