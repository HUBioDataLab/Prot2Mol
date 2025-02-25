import os
import lzma
import argparse
import requests
import pandas as pd
from conversion import ChemblUniprotConverter, process_in_parallel

# Define the URLs and the output directory
molecule_url = "https://zenodo.org/records/7019874/files/05.5++_combined_set_without_stereochemistry.tsv.xz"
protein_url = "https://zenodo.org/records/7019874/files/05.5_combined_set_protein_targets.tsv.xz"
output_directory = "../data/papyrus/"

def download_and_decompress(url, output_dir):
    # Get the filename from the URL
    filename = url.split('/')[-1]
    compressed_file_path = os.path.join(output_dir, filename)
    decompressed_file_path = os.path.join(output_dir, filename.replace('.xz', ''))

    if os.path.exists(decompressed_file_path):
        print(f"\nFile {decompressed_file_path} already exists. Skipping download.\n")
        return decompressed_file_path
    else:
        print(f"Downloading and decompressing {url}...")
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful
        with open(compressed_file_path, 'wb') as compressed_file:
            for chunk in response.iter_content(chunk_size=8192):
                compressed_file.write(chunk)

        # Decompress the file
        with lzma.open(compressed_file_path, 'rb') as compressed_file:
            with open(decompressed_file_path, 'wb') as decompressed_file:
                decompressed_file.write(compressed_file.read())

        # Optionally remove the compressed file
        os.remove(compressed_file_path)

        return decompressed_file_path

def prepare_papyrus(molecule_url, protein_url, output_directory, pchembl_threshold=None, prot_len=None, only_human=None, max_cores=10):
    
    converter = ChemblUniprotConverter()
    molecule_file = download_and_decompress(molecule_url, output_directory)
    protein_file = download_and_decompress(protein_url, output_directory)
    
    mol_data = pd.read_csv(molecule_file, sep='\t')
    prot_data = pd.read_csv(protein_file, sep='\t')
    if only_human:
        prot_data = prot_data[prot_data["Organism"] == "Homo sapiens (Human)"].reset_index(drop=True)
    
    prot_comp_set = (
            pd.merge(mol_data[["SMILES","accession", "pchembl_value_Mean","target_id"]], prot_data[["target_id","Sequence"]], on="target_id")
            .assign(Target_CHEMBL_ID=lambda df: df['accession'].apply(converter.convert_2_chembl_id))
            .query('Target_CHEMBL_ID.str.startswith("CHEMBL")')
            .rename(columns={"SMILES": "Compound_SMILES", "accession": "Target_Accession", "target_id": "Target_ID", "Sequence": "Target_FASTA"})
            .assign(Protein_Length=lambda df: df["Target_FASTA"].apply(len))
            .assign(Compound_SELFIES=lambda df: process_in_parallel(df["Compound_SMILES"], max_cores))
            .dropna())

    print(len(prot_comp_set))
    if pchembl_threshold: 
        prot_comp_set = prot_comp_set.query("pchembl_value_Mean >= @pchembl_threshold")
        print(len(prot_comp_set))

    if prot_len:
        prot_comp_set = prot_comp_set.query("Protein_Length < @prot_len")
        print(len(prot_comp_set))
        
    prot_comp_set[["Target_FASTA", "Target_CHEMBL_ID", "Compound_SELFIES"]].to_csv(
        f"../data/papyrus/prot_comp_set_pchembl_{pchembl_threshold}_protlen_{prot_len}_human_{only_human}.csv", 
        index=False)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pchembl_threshold",help="pchembl threshold, can be None", type=int, default=6)
    parser.add_argument("--prot_len", help="Maximum protein length, can be None", type=int, default=1000)
    parser.add_argument("--human_only", help="Only human proteins", type=bool, default=None)
    parser.add_argument("--max_cores", help="Maximum number of cores to use", type=int, default=10)
    config = parser.parse_args()    
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Download and decompress the files
    prepare_papyrus(molecule_url, protein_url, output_directory, 
                    pchembl_threshold=config.pchembl_threshold, 
                    prot_len=config.prot_len, only_human=config.human_only,
                    max_cores=config.max_cores)
    
    print("Papyrus data preparation complete.\n")