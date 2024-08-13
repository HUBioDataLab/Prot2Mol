import pandas as pd
import argparse
import os

def train_val_test_split(dataset, test_ids, train_val_ratio=0.9):
    
    data = pd.read_csv(dataset)
    path = dataset.split(".csv")[0]
    if not os.path.exists(path):
        os.makedirs(path)    

    train_val_data = data[~data["Target_CHEMBL_ID"].isin([test_ids])]
    test_data = data[data["Target_CHEMBL_ID"].isin([test_ids])]
    train_val_data = train_val_data.sample(frac=1, random_state=42)
    train_data = train_val_data[:int(train_val_data.shape[0] * train_val_ratio)]
    val_data = train_val_data[int(train_val_data.shape[0] * train_val_ratio):]
    
    return train_data.reset_index(drop=True), val_data.reset_index(drop=True), test_data.reset_index(drop=True)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument("--dataset", required=False, default="../data/papyrus/prot_comp_set_pchembl_8_protlen_150_human_False.csv", help="Path of the SELFIES dataset.")
    parser.add_argument("--test_ids", required=False, default="CHEMBL4296327", help="Protein ids for test set.")
    config = parser.parse_args()

    train_data, val_data, test_data = train_val_test_split(config.dataset, config.test_ids)
    
    path = config.dataset.split(".csv")[0]
    if not os.path.exists(path):
        os.makedirs(path)
    
    train_data.to_csv(f"{path}/train.csv", index=False)
    val_data.to_csv(f"{path}/val.csv", index=False)
    test_data.to_csv(f"{path}/test_{config.test_ids}.csv", index=False)