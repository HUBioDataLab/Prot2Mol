from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, ligand_data, target_data, tokenizer, max_length=200):
        self.ligand_data = ligand_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_data_dict = self.preprocess_target_data(target_data)
        self.length = len(self.ligand_data)

    def preprocess_target_data(self, target_data):
        return {target_id: enc_state for target_id, enc_state in zip(target_data["Target_CHEMBL_ID"], target_data["encoder_hidden_states"])}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return self._get_batch(indices)
        elif isinstance(idx, list):
            return self._get_batch(idx)
        else:
            return self._get_single(idx)

    def _get_single(self, idx):
        ligand_row = self.ligand_data.iloc[idx]
        target_id = ligand_row["Target_CHEMBL_ID"]
        enc_state = self.target_data_dict[target_id]
        compound_selfies = ligand_row["Compound_SELFIES"]
        input_ids = self.get_input_ids(compound_selfies)
        return {"input_ids": input_ids, "encoder_hidden_states": enc_state}

    def _get_batch(self, indices):
        input_ids_list = []
        encoder_hidden_states_list = []
        
        for idx in indices:
            sample = self._get_single(idx)
            input_ids_list.append(sample["input_ids"])
            encoder_hidden_states_list.append(sample["encoder_hidden_states"])
        
        input_ids_batch = torch.stack(input_ids_list)
        encoder_hidden_states_batch = torch.stack(encoder_hidden_states_list)
        
        return {"input_ids": input_ids_batch, "encoder_hidden_states": encoder_hidden_states_batch}

    def get_input_ids(self, compound_selfies):
        return self.tokenizer(compound_selfies, 
                              max_length=self.max_length, 
                              padding="max_length", 
                              truncation=True, 
                              return_tensors="pt")["input_ids"].squeeze()