import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem
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
import logging

logger = logging.getLogger(__name__)

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

def decode_selfies_list(selfies_list):
    """
    Decode a collection of SELFIES strings into SMILES strings.
    Invalid SELFIES are converted to None.
    """
    decoded = []
    for selfies in selfies_list:
        if not isinstance(selfies, str):
            decoded.append(None)
            continue
        normalized = selfies.replace(" ", "")
        try:
            smiles = sf.decoder(normalized)
            decoded.append(smiles if smiles else None)
        except Exception:
            decoded.append(None)
    return decoded

def canonicalize_smiles_list(smiles_list, drop_invalid=False):
    """
    Canonicalize an iterable of SMILES strings using RDKit.
    
    Args:
        smiles_list: Iterable of SMILES strings (or values convertible to strings)
        drop_invalid: When True, invalid entries are removed. Otherwise they are kept as empty strings.
    """
    canonical = []
    for smiles in smiles_list:
        if isinstance(smiles, str):
            cleaned = smiles.strip()
        else:
            cleaned = None
        canonical_smiles = canonic_smiles(cleaned) if cleaned else None
        if canonical_smiles:
            canonical.append(canonical_smiles)
        elif not drop_invalid:
            canonical.append("")
    if drop_invalid:
        canonical = [s for s in canonical if s]
    return canonical


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
    best_stock_indices = np.zeros(len(gen_vecs), dtype=int)
    
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
                max_vals = jac.max(0)
                max_indices = jac.argmax(0)
                mask = max_vals > agg_tanimoto[i:i + y_gen.shape[1]]
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], max_vals)
                best_stock_indices[i:i + y_gen.shape[1]][mask] = j + max_indices[mask]
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    
    if no_list:
        return np.mean(agg_tanimoto)
    else:
        return np.mean(agg_tanimoto), agg_tanimoto, best_stock_indices

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
    
    # `predictions` are decoded SELFIES from the model.
    predictions = [(x or "").replace(" ", "") for x in predictions]
    decoded_predictions = decode_selfies_list(predictions)
    prediction_smiles = pd.DataFrame(
        canonicalize_smiles_list(decoded_predictions, drop_invalid=False),
        columns=["smiles"]
    )
    
    # Initialize all metrics to 0
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
    
    # Try validity calculation
    try:
        prediction_validity_ratio = fraction_valid(list(prediction_smiles["smiles"]))
        metrics["validity"] = prediction_validity_ratio
    except (ZeroDivisionError, ValueError) as e:
        logging.warning(f"Zero division at validity calculation: {e}")
        metrics["validity"] = 0
        prediction_validity_ratio = 0
    
    if prediction_validity_ratio != 0:
        
        prediction_mols = to_mol(list(prediction_smiles["smiles"]))
    
        # Handle both DataFrame and list inputs for train_data
        if isinstance(train_data, list):
            training_data_smiles = [s for s in train_data if s]
        elif hasattr(train_data, 'columns'):
            if "Compound_SMILES" in train_data.columns:
                training_data_smiles = canonicalize_smiles_list(train_data["Compound_SMILES"].tolist(), drop_invalid=True)
            elif "Compound_SELFIES" in train_data.columns:
                decoded = decode_selfies_list(train_data["Compound_SELFIES"].tolist())
                training_data_smiles = canonicalize_smiles_list(decoded, drop_invalid=True)
            else:
                training_data_smiles = []
        elif hasattr(train_data, 'column_names'):
            if "Compound_SMILES" in train_data.column_names:
                training_data_smiles = canonicalize_smiles_list(train_data["Compound_SMILES"], drop_invalid=True)
            elif "Compound_SELFIES" in train_data.column_names:
                decoded = decode_selfies_list(train_data["Compound_SELFIES"])
                training_data_smiles = canonicalize_smiles_list(decoded, drop_invalid=True)
            else:
                training_data_smiles = []
        else:
            logging.warning(f"Unexpected train_data type: {type(train_data)}, using empty list")
            training_data_smiles = []
        
        reference_smiles = canonicalize_smiles_list(references or [], drop_invalid=True)
        
        # Try uniqueness calculation
        try:
            prediction_uniqueness_ratio = fraction_unique(prediction_smiles["smiles"])
            metrics["uniqueness"] = prediction_uniqueness_ratio
        except (ZeroDivisionError, ValueError) as e:
            logging.warning(f"Zero division at uniqueness calculation: {e}")
            metrics["uniqueness"] = 0
        
        # Try novelty calculations
        try:
            prediction_smiles_novelty_against_training_samples = novelty(list(prediction_smiles["smiles"]), training_data_smiles)
            metrics["novelty_against_training_samples"] = prediction_smiles_novelty_against_training_samples
        except (ZeroDivisionError, ValueError) as e:
            logging.warning(f"Zero division at novelty_against_training_samples calculation: {e}")
            metrics["novelty_against_training_samples"] = 0
            
        try:
            prediction_smiles_novelty_against_reference_samples = novelty(list(prediction_smiles["smiles"]), reference_smiles)
            metrics["novelty_against_reference_samples"] = prediction_smiles_novelty_against_reference_samples
        except (ZeroDivisionError, ValueError) as e:
            logging.warning(f"Zero division at novelty_against_reference_samples calculation: {e}")
            metrics["novelty_against_reference_samples"] = 0
        
        # Try similarity calculations
        try:
            prediction_vecs = generate_vecs(prediction_mols)
            reference_mols = [Chem.MolFromSmiles(x) for x in reference_smiles]
            reference_mols = [mol for mol in reference_mols if mol is not None]
            
            if reference_mols:
                reference_vec = generate_vecs(reference_mols)
                predicted_vs_reference_sim_mean, predicted_vs_reference_sim_list, _ = average_agg_tanimoto(reference_vec,prediction_vecs, no_list=False)
                metrics["similarity_to_reference_samples"] = predicted_vs_reference_sim_mean
            else:
                predicted_vs_reference_sim_list = []
                metrics["similarity_to_reference_samples"] = 0
        except (ZeroDivisionError, ValueError, RuntimeError) as e:
            logging.warning(f"Zero division at similarity_to_reference_samples calculation: {e}")
            metrics["similarity_to_reference_samples"] = 0
            predicted_vs_reference_sim_list = []
            
        try:
            if train_vec is not None:
                predicted_vs_training_sim_mean, predicted_vs_training_sim_list, _ = average_agg_tanimoto(train_vec,prediction_vecs, no_list=False)
                metrics["similarity_to_training_samples"] = predicted_vs_training_sim_mean
            else:
                predicted_vs_training_sim_mean, predicted_vs_training_sim_list = 0, []
                metrics["similarity_to_training_samples"] = 0
        except (ZeroDivisionError, ValueError, RuntimeError) as e:
            logging.warning(f"Zero division at similarity_to_training_samples calculation: {e}")
            metrics["similarity_to_training_samples"] = 0
            predicted_vs_training_sim_list = []
        
        # Try internal diversity calculation
        try:
            IntDiv = 1 - average_agg_tanimoto(prediction_vecs, prediction_vecs, agg="mean", no_list=True)
            metrics["intdiv"] = IntDiv
        except (ZeroDivisionError, ValueError, RuntimeError) as e:
            logging.warning(f"Zero division at intdiv calculation: {e}")
            metrics["intdiv"] = 0
        
        # Try SA score calculation
        try:
            prediction_sa_score_list = sascorer_calculation(prediction_mols)
            # Filter out None values before calculating mean
            valid_sa_scores = [score for score in prediction_sa_score_list if score is not None]
            if valid_sa_scores:
                prediction_sa_score = np.mean(valid_sa_scores)
            else:
                prediction_sa_score = 0
            metrics["sa_score"] = prediction_sa_score
        except (ZeroDivisionError, ValueError) as e:
            logging.warning(f"Zero division at sa_score calculation: {e}")
            metrics["sa_score"] = 0
            prediction_sa_score_list = []
        
        # Try QED score calculation
        try:
            prediction_qed_score_list = qed_calculation(prediction_mols)
            # Filter out None values before calculating mean
            valid_qed_scores = [score for score in prediction_qed_score_list if score is not None]
            if valid_qed_scores:
                prediction_qed_score = np.mean(valid_qed_scores)
            else:
                prediction_qed_score = 0
            metrics["qed_score"] = prediction_qed_score
        except (ZeroDivisionError, ValueError) as e:
            logging.warning(f"Zero division at qed_score calculation: {e}")
            metrics["qed_score"] = 0
            prediction_qed_score_list = []
        
        # Try LogP score calculation
        try:
            prediction_logp_score_list = logp_calculation(prediction_mols)
            # Filter out None values before calculating mean
            valid_logp_scores = [score for score in prediction_logp_score_list if score is not None]
            if valid_logp_scores:
                prediction_logp_score = np.mean(valid_logp_scores)
            else:
                prediction_logp_score = 0
            metrics["logp_score"] = prediction_logp_score
        except (ZeroDivisionError, ValueError) as e:
            logging.warning(f"Zero division at logp_score calculation: {e}")
            metrics["logp_score"] = 0
            prediction_logp_score_list = []
    
    if training: 
        # Only log to wandb if it has been initialized
        try:
            if wandb.run is not None:
                wandb.log(metrics)
        except Exception as e:
            logging.warning(f"Failed to log metrics to wandb: {e}")
    if training:
        return metrics
    elif training == False:
        # Get the number of predictions to ensure all arrays have the same length
        num_predictions = len(prediction_smiles["smiles"])
        
        # Ensure all lists exist and have the correct length
        if 'predicted_vs_reference_sim_list' not in locals():
            predicted_vs_reference_sim_list = [None] * num_predictions
        elif len(predicted_vs_reference_sim_list) != num_predictions:
            predicted_vs_reference_sim_list = (predicted_vs_reference_sim_list + [None] * num_predictions)[:num_predictions]
            
        if 'predicted_vs_training_sim_list' not in locals():
            predicted_vs_training_sim_list = [None] * num_predictions
        elif len(predicted_vs_training_sim_list) != num_predictions:
            predicted_vs_training_sim_list = (predicted_vs_training_sim_list + [None] * num_predictions)[:num_predictions]
            
        if 'prediction_sa_score_list' not in locals():
            prediction_sa_score_list = [None] * num_predictions
        elif len(prediction_sa_score_list) != num_predictions:
            prediction_sa_score_list = (prediction_sa_score_list + [None] * num_predictions)[:num_predictions]
            
        if 'prediction_qed_score_list' not in locals():
            prediction_qed_score_list = [None] * num_predictions
        elif len(prediction_qed_score_list) != num_predictions:
            prediction_qed_score_list = (prediction_qed_score_list + [None] * num_predictions)[:num_predictions]
            
        if 'prediction_logp_score_list' not in locals():
            prediction_logp_score_list = [None] * num_predictions
        elif len(prediction_logp_score_list) != num_predictions:
            prediction_logp_score_list = (prediction_logp_score_list + [None] * num_predictions)[:num_predictions]
            
        # Verify all arrays have the same length before creating DataFrame
        arrays_info = {
            "smiles": len(prediction_smiles["smiles"]),
            "test_sim": len(predicted_vs_reference_sim_list),
            "train_sim": len(predicted_vs_training_sim_list),
            "sa_score": len(prediction_sa_score_list),
            "qed_score": len(prediction_qed_score_list),
            "logp_score": len(prediction_logp_score_list)
        }
        
        # Check if all arrays have the same length
        lengths = list(arrays_info.values())
        if len(set(lengths)) > 1:
            logging.warning(f"Array length mismatch detected: {arrays_info}")
            # Force all arrays to have the same length as smiles
            target_length = len(prediction_smiles["smiles"])
            predicted_vs_reference_sim_list = (predicted_vs_reference_sim_list + [None] * target_length)[:target_length]
            predicted_vs_training_sim_list = (predicted_vs_training_sim_list + [None] * target_length)[:target_length]
            prediction_sa_score_list = (prediction_sa_score_list + [None] * target_length)[:target_length]
            prediction_qed_score_list = (prediction_qed_score_list + [None] * target_length)[:target_length]
            prediction_logp_score_list = (prediction_logp_score_list + [None] * target_length)[:target_length]
            
        result_dict = {"smiles": prediction_smiles["smiles"],
                       "test_sim": predicted_vs_reference_sim_list, 
                       "train_sim": predicted_vs_training_sim_list,
                       "sa_score": prediction_sa_score_list,
                       "qed_score": prediction_qed_score_list,
                       "logp_score": prediction_logp_score_list
                       }
        
        try:
            results = pd.DataFrame.from_dict(result_dict)
        except ValueError as e:
            logging.error(f"DataFrame creation failed: {e}")
            # Fallback: return only SMILES
            results = pd.DataFrame({"smiles": prediction_smiles["smiles"]})
        
        return metrics, results
