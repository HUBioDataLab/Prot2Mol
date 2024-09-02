import os
import numpy as np
from datasets import Dataset
from conversion import ChemblUniprotConverter
import argparse
import requests
import zipfile

converter = ChemblUniprotConverter()

def download_af2_emb(url, save_path, target_dir):
    
    if not os.path.exists(save_path):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            
            os.makedirs(target_dir, exist_ok=True)
            zip_ref.extractall(target_dir)
        os.remove(save_path)
    
def get_af2_embeddings(path, file_name, max_length):
    folder_names = []
    all_structures = []
    for root, _, files in os.walk(path):
        current_directory_path = os.path.abspath(root)
        folder_name = os.path.basename(root)
        for f in files:
            if f == file_name:
                current_structure_path = os.path.join(current_directory_path, f)
                current_structure = np.load(current_structure_path)
                
                # Pad the array if necessary
                max_length = int(max_length)
                if current_structure.shape[0] < max_length:
                    padding = max_length - current_structure.shape[0]
                    current_structure = np.pad(current_structure, ((0, padding), (0, 0)), 'constant')
                    all_structures.append(current_structure)
                    folder_names.append(folder_name)
                elif current_structure.shape[0] == max_length:
                    all_structures.append(current_structure)
                    folder_names.append(folder_name)

    # Concatenate all the structures into a single array
    if all_structures:
        all_structures = np.stack(all_structures, axis=0)

    return all_structures, folder_names

def create_combined_hf_dataset(path, max_length, save_path):
    all_structures, folder_names = get_af2_embeddings(path, "structure.npy", max_length)
    all_single, _ = get_af2_embeddings(path, "single.npy", max_length)
    all_combined = np.concatenate((all_structures, all_single), axis=-1)
    prot_names = [converter.convert_2_chembl_id(folder_name) for folder_name in folder_names]

    print("combined data:", all_combined.shape)
    np.savez(save_path, Target_CHEMBL_ID=prot_names, encoder_hidden_states=all_combined)

def create_hf_dataset(path, file_name, max_length, save_path):
    all_structures, folder_names = get_af2_embeddings(path, file_name, max_length)
    prot_names = [converter.convert_2_chembl_id(folder_name) for folder_name in folder_names]
    print(file_name, ":", all_structures.shape)
    np.savez(save_path, Target_CHEMBL_ID=prot_names, encoder_hidden_states=all_structures)  # Save the embeddings

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument("--max_len", required=False, default=500, help="Maximum protein length")
    config = parser.parse_args()        
    
    save_path = "../data/FoldedPapyrus_4581_v01.zip"
    af2_url = "https://zenodo.org/records/10671261/files/FoldedPapyrus_4581_v01.zip"
    target_dir = "../data/af2/"
    
    if not os.path.exists(target_dir):
        print("\nDownloading AlphaFold2 embeddings...\n")
        download_af2_emb(af2_url, save_path, target_dir)
    print("Alphafold2 embeddings have been found skipping download.\n")
    
    max_length = config.max_len  # Change this to the length you want to pad to
    
    data_name = save_path.split("/")[-1].split(".")[0]
    path = f"../data/af2/proteins"
    
    print("Processing AlphaFold2 embeddings...\n")
    
    
    if not os.path.exists(f"../data/prot_embed/af2_struct/{data_name}"):
        os.makedirs(f"../data/prot_embed/af2_struct/{data_name}") 
    create_hf_dataset(path, "structure.npy", config.max_len, f"../data/prot_embed/af2_struct/{data_name}/embeddings.npz")
    
    if not os.path.exists(f"../data/prot_embed/af2_single/{data_name}"):
        os.makedirs(f"../data/prot_embed/af2_single/{data_name}")
    create_hf_dataset(path, "single.npy", config.max_len, f"../data/prot_embed/af2_single/{data_name}/embeddings.npz")
    
    
    if not os.path.exists(f"../data/prot_embed/af2_combined/{data_name}"):
        os.makedirs(f"../data/prot_embed/af2_combined/{data_name}")
    create_combined_hf_dataset(path, config.max_len, f"../data/prot_embed/af2_combined/{data_name}/embeddings.npz")
    
    print("Residual files are being removed...\n")
    
    os.system(f"rm -r {target_dir}")
    
    print("\nAlphaFold2 embeddings have been successfully downloaded and processed.\n")
