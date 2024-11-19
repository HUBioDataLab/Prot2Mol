import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from rdkit.Chem import QED
import warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger    
RDLogger.DisableLog('rdApp.*')  
from multiprocessing import Pool
import torch
import numpy as np
import wandb

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map


def remove_invalid(gen, canonize=True, n_jobs=1):
    """
    Removes invalid molecules from the dataset
    """
    if not canonize:
        mols = mapper(n_jobs)(get_mol, gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in mapper(n_jobs)(canonic_smiles, gen) if
            x is not None]


def fraction_valid(gen, n_jobs=1):
    """
    Computes a number of valid molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
    """
    gen = mapper(n_jobs)(get_mol, gen)
    return 1 - gen.count(None) / len(gen)


def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def fraction_unique(gen, k=None, n_jobs=1, check_validity=True):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        k: compute unique@k
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    if k is not None:
        if len(gen) < k:
            warnings.warn(
                "Can't compute unique@{}.".format(k) +
                "gen contains only {} molecules".format(len(gen))
            )
        gen = gen[:k]
    canonic = set(mapper(n_jobs)(canonic_smiles, gen))
    if None in canonic and check_validity:
        canonic = [i for i in canonic if i is not None]
        #raise ValueError("Invalid molecule passed to unique@k")
    return 0 if len(gen) == 0 else len(canonic) / len(gen)


def novelty(gen, train, n_jobs=1):
    gen_smiles = mapper(n_jobs)(canonic_smiles, gen)
    gen_smiles_set = set(gen_smiles) - {None}
    train_set = set(train)
    return 0 if len(gen_smiles_set) == 0 else len(gen_smiles_set - train_set) / len(gen_smiles_set)


def average_agg_tanimoto(stock_vecs, gen_vecs,
                         batch_size=5000, agg='max',
                         device='cpu', p=1, no_list=True):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return np.mean(agg_tanimoto) if no_list else np.mean(agg_tanimoto), agg_tanimoto

def generate_vecs(mols):
    zero_vec = np.zeros(1024)
    return np.array([AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) if mol is not None else zero_vec for mol in mols])

def to_mol(smiles_list):
    return [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

def sascorer_calculation(mols):
    return [sascorer.calculateScore(mol) if mol is not None else None for mol in mols]

def qed_calculation(mols):
    return [QED.qed(mol) if mol is not None else None for mol in mols]

def logp_calculation(mols):
    return [Chem.Crippen.MolLogP(mol) if mol is not None else None for mol in mols]

def metrics_calculation(predictions, references, train_data, train_vec=None,training=True):

    predictions = [x.replace(" ", "") for x in predictions]
    references = [x.replace(" ", "") for x in references]

    prediction_smiles = pd.DataFrame([sf.decoder(x) for x in predictions], columns=["smiles"])
    
    prediction_validity_ratio = fraction_valid(list(prediction_smiles["smiles"]))
    
    if prediction_validity_ratio != 0:
        
        prediction_mols = to_mol(list(prediction_smiles["smiles"]))
    
        training_data_smiles = [sf.decoder(x) for x in train_data["Compound_SELFIES"]]
        reference_smiles = [sf.decoder(x) for x in references] 
        
        prediction_uniqueness_ratio = fraction_unique(prediction_smiles["smiles"])
        
        prediction_smiles_novelty_against_training_samples = novelty(list(prediction_smiles["smiles"]), training_data_smiles)
        prediction_smiles_novelty_against_reference_samples = novelty(list(prediction_smiles["smiles"]), reference_smiles)
        
        prediction_vecs = generate_vecs(prediction_mols)
        reference_vec = generate_vecs([Chem.MolFromSmiles(x) for x in reference_smiles if Chem.MolFromSmiles(x) is not None])
        
        predicted_vs_reference_sim_mean, predicted_vs_reference_sim_list = average_agg_tanimoto(reference_vec,prediction_vecs, no_list=False)
        if train_vec is not None:
            predicted_vs_training_sim_mean, predicted_vs_training_sim_list = average_agg_tanimoto(train_vec,prediction_vecs, no_list=False)
        else:
            predicted_vs_training_sim_mean, predicted_vs_training_sim_list = 0, []
        
        IntDiv = 1 - average_agg_tanimoto(prediction_vecs, prediction_vecs, agg="mean", no_list=True)[0]
        
        prediction_sa_score_list = sascorer_calculation(prediction_mols)
        prediction_sa_score = np.mean(prediction_sa_score_list)
        
        prediction_qed_score_list = qed_calculation(prediction_mols)
        prediction_qed_score = np.mean(prediction_qed_score_list)
        
        prediction_logp_score_list = logp_calculation(prediction_mols)
        prediction_logp_score = np.mean(prediction_logp_score_list)
        
        metrics = {"validity": prediction_validity_ratio,
                   "uniqueness": prediction_uniqueness_ratio,
                   "novelty_against_training_samples": prediction_smiles_novelty_against_training_samples,
                   "novelty_against_reference_samples": prediction_smiles_novelty_against_reference_samples,
                   "intdiv": IntDiv,
                   "similarity_to_training_samples": predicted_vs_training_sim_mean,
                   "similarity_to_reference_samples": predicted_vs_reference_sim_mean,
                   "sa_score": prediction_sa_score,
                   "qed_score": prediction_qed_score,
                   "logp_score": prediction_logp_score}
    
    else:
        metrics = {"validity": 0,
                   "uniqueness": 0,
                   "novelty_against_training_samples": 0,
                   "novelty_against_reference_samples": 0,
                   "intdiv": 0,
                   "similarity_to_training_samples": 0,
                   "similarity_to_reference_samples": 0,
                   "sa_score": 0,
                   "qed_score": 0,
                   "logp_score": 0}
    if training: 
        wandb.log(metrics) 
    if training:
        return metrics
    else:
        
        result_dict = {"smiles": prediction_smiles["smiles"],
                       "test_sim": predicted_vs_reference_sim_list, 
                       "train_sim": predicted_vs_training_sim_list,
                       "sa_score": prediction_sa_score_list,
                       "qed_score": prediction_qed_score_list,
                       "logp_score": prediction_logp_score_list
                       }
        results = pd.DataFrame.from_dict(result_dict)
        
        return metrics, results
        


        
        
        