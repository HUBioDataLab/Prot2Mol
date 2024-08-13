import selfies as sf
import requests
import pandas as pd
import os 
from multiprocessing import Pool

def to_selfies(smiles):
    try:
        selfi = sf.encoder(smiles,strict=False)
    except:
        return None
    return selfi

def process_in_parallel(smiles_list, num_processes):
    with Pool(num_processes) as p:
        selfies_list = p.map(to_selfies, smiles_list)
    return selfies_list

class ChemblUniprotConverter:
    def __init__(self):
        self.url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_uniprot_mapping.txt"
        self.save_path = "../data/chembl_uniprot_mapping.txt"
        self.uniprot_to_chembl = self.load_mapping()

    def load_mapping(self):
        if not os.path.exists(self.save_path):
            self.download_text_file(self.url, self.save_path)
        data = pd.read_table(self.save_path,  names=["ChEMBL_ID", "UniProt_ID", "Name", "Type"]).drop(axis=0, labels=0)
        data = data[data["Type"] == "SINGLE PROTEIN"]
        return {item["ChEMBL_ID"]: item["UniProt_ID"] for item in data.to_dict('records')}

    def convert_2_chembl_id(self, input_id):
        return self.uniprot_to_chembl.get(input_id, input_id)

    def download_text_file(self, url, save_path):
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)
            print(f"File downloaded and saved to {save_path}")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")





