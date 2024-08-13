import os
import torch
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, EsmModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load ESM-2 model
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
model.eval()

def produce_esm2_embedding(data, prot_len):

    inputs = tokenizer(data, return_tensors="pt", padding="max_length", max_length=prot_len, truncation=True).to(device)
    outputs = model(**inputs)

    sequence_out = outputs.last_hidden_state
    return sequence_out

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument("--dataset", required=False, default="../data/papyrus/prot_comp_set_pchembl_None_protlen_500_human_False.csv", help="Path of the SELFIES dataset.")
    parser.add_argument("--prot_len", default=500, help="Max length of the protein sequence.")
    config = parser.parse_args()
    
    dataset_name = config.dataset.split("/")[-1].split(".")[0]
    data = pd.read_csv(config.dataset)
    unique_target  = data.drop_duplicates(subset=['Target_CHEMBL_ID'])[["Target_CHEMBL_ID", "Target_FASTA"]]
    unique_target["len"] = unique_target["Target_FASTA"].apply(lambda x: len(x))
    unique_target = unique_target[unique_target["len"] < config.prot_len].reset_index(drop=True)
    path  = "../data/prot_embed/esm2/" + dataset_name + "/unique_target.csv"
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
    unique_target.to_csv(path, index=False)

    token_rep = np.array([produce_esm2_embedding(seq, config.prot_len).detach().cpu().numpy() for seq in unique_target["Target_FASTA"]]).squeeze()
    
    print(token_rep.shape)
    
    prot_path= f"../data/prot_embed/esm2/{dataset_name}/embeddings.npz"
    np.savez(prot_path, Target_CHEMBL_ID=unique_target["Target_CHEMBL_ID"], encoder_hidden_states=token_rep)  # Save the embeddings
    

