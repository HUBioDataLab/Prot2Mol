"""Autoregressive molecule-generation metrics."""

from __future__ import annotations

from ..chem.utils import metrics_calculation


def compute_generation_metrics(
    generated_token_ids,
    mol_tokenizer,
    eval_reference_smiles,
    train_smiles_list,
    training_vec,
):
    decoded = mol_tokenizer.batch_decode(
        generated_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    metrics = metrics_calculation(
        predictions=decoded,
        references=eval_reference_smiles,
        train_data=train_smiles_list,
        train_vec=training_vec,
        return_details=False,
    )
    return {f"gen_{key}": value for key, value in metrics.items()}


def compute_conditional_generation_metrics(
    generated_token_ids_by_protein,
    mol_tokenizer,
    reference_smiles_by_protein,
):
    """Macro-average reference metrics within the correct protein context."""

    per_protein = []
    for protein_sequence, token_ids in generated_token_ids_by_protein.items():
        references = reference_smiles_by_protein.get(protein_sequence, [])
        if not references:
            continue
        decoded = mol_tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        metrics = metrics_calculation(
            predictions=decoded,
            references=references,
            train_data=[],
        )
        per_protein.append(metrics)

    if not per_protein:
        return {
            "gen_conditional_protein_count": 0,
            "gen_conditional_similarity_eval_macro": 0.0,
            "gen_conditional_reference_recovery_macro": 0.0,
        }
    count = len(per_protein)
    return {
        "gen_conditional_protein_count": count,
        "gen_conditional_similarity_eval_macro": sum(
            row["similarity_eval"] for row in per_protein
        )
        / count,
        "gen_conditional_reference_recovery_macro": sum(
            1.0 - row["novelty_eval"] for row in per_protein
        )
        / count,
    }
