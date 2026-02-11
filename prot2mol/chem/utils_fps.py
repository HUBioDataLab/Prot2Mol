from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import multiprocessing as mp
from functools import partial
import numpy as np
from typing import List, Optional

def _smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """Convert a SMILES string to an RDKit Mol object, or return None if invalid."""
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None

def _process_smiles_chunk(smiles_list: List[str], radius: int, nBits: int) -> List[np.ndarray]:
    """
    Process a small list of SMILES and return their Morgan fingerprint bit-vectors.
    Each fingerprint is converted into a numpy array of shape (nBits,).
    """
    fps = []
    for smi in smiles_list:
        mol = _smiles_to_mol(smi)
        if mol is not None:
            # Compute Morgan fingerprint
            fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            # Convert to numpy array of uint8 (0/1 bits)
            arr = np.zeros((nBits,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp_vect, arr)  # RDKit helper
            fps.append(arr)
    return fps

def generate_morgan_fingerprints_parallel(
    smiles: List[str],
    radius: int = 2,
    nBits: int = 1024,
    n_jobs: int = None,
    chunk_size: int = None
) -> np.ndarray:
    """
    Generate Morgan fingerprints for a list of SMILES in parallel, returning
    a (N, nBits) numpy array. Limits cores to max 10 and splits work into chunks.
    """
    # determine number of processes
    max_cores = mp.cpu_count() - 1
    n_jobs = min(max_cores, 10) if n_jobs is None else min(n_jobs, 10)
    # filter out empty or None
    valid = [s for s in smiles if s]
    total = len(valid)
    if total == 0:
        return np.zeros((0, nBits), dtype=np.uint8)

    # if small or only 1 core, do serial
    if n_jobs <= 1 or total < 1000:
        fps = _process_smiles_chunk(valid, radius, nBits)
    else:
        # determine chunk size
        chunk_size = chunk_size or max(1, total // (n_jobs * 10))
        chunks = [valid[i:i+chunk_size] for i in range(0, total, chunk_size)]
        func = partial(_process_smiles_chunk, radius=radius, nBits=nBits)
        with mp.Pool(n_jobs) as pool:
            results = pool.map(func, chunks)
        fps = [fp for sub in results for fp in sub]

    # stack into single numpy array
    return np.vstack(fps)