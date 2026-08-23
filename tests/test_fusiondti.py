from types import SimpleNamespace

import pytest
import torch

from prot2mol.rewards.fusiondti import (
    FusionDTIActivityHead,
    FusionDTIActivityScorer,
    FusionDTISelfiesTokenizer,
    download_fusiondti_artifacts,
    load_fusiondti_head,
)


class TinyProteinTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, sequences, *, max_length, **kwargs):
        self.calls.append(list(sequences))
        rows = []
        masks = []
        for sequence in sequences:
            tokens = [1]
            tokens.extend((ord(character) % 29) + 3 for character in sequence)
            tokens.append(2)
            tokens = tokens[:max_length]
            mask = [1] * len(tokens)
            padding = max_length - len(tokens)
            rows.append(tokens + [0] * padding)
            masks.append(mask + [0] * padding)
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class TinyProteinEncoder(torch.nn.Module):
    def __init__(self, output_dim=6):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, output_dim)
        self.calls = 0

    def forward(self, input_ids, attention_mask, return_dict=True):
        self.calls += 1
        return SimpleNamespace(logits=self.embedding(input_ids))


class TinyMoleculeEncoder(torch.nn.Module):
    def __init__(self, output_dim=5):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, output_dim)

    def forward(self, input_ids, attention_mask, return_dict=True):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def tiny_molecule_tokenizer():
    return FusionDTISelfiesTokenizer(
        vocab={
            "<unk>": 0,
            "<s>": 1,
            "</s>": 2,
            "<pad>": 3,
            "C": 4,
            "O": 5,
            "N": 6,
            "F": 7,
            "=O": 8,
        },
        special_tokens={
            "unk_token": "<unk>",
            "cls_token": "<s>",
            "sep_token": "</s>",
            "pad_token": "<pad>",
        },
    )


def test_fusiondti_selfies_tokenizer_matches_published_token_contract():
    tokenizer = tiny_molecule_tokenizer()

    encoded = tokenizer.batch_encode(
        ["[C][=O]", "[Xe]"],
        max_length=5,
    )

    assert encoded["input_ids"].tolist() == [
        [1, 4, 8, 2, 3],
        [1, 0, 2, 3, 3],
    ]
    assert encoded["attention_mask"].tolist() == [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
    ]
    with pytest.raises(ValueError, match="expected SELFIES"):
        tokenizer.batch_encode(["CCO"], max_length=5)


def test_fusiondti_head_has_released_bindingdb_checkpoint_shapes():
    head = FusionDTIActivityHead()
    state = head.state_dict()

    assert state["prot_reg.weight"].shape == (512, 446)
    assert state["drug_reg.weight"].shape == (512, 768)
    assert state["can_layer.query_p.weight"].shape == (512, 512)
    assert state["mlp_classifier.fc1.weight"].shape == (1024, 1024)
    assert state["mlp_classifier.output.weight"].shape == (1, 256)
    assert len(state) == 33


def test_fusiondti_checkpoint_loader_is_strict_and_frozen(tmp_path):
    checkpoint = tmp_path / "fusiondti.ckpt"
    source = FusionDTIActivityHead()
    torch.save(source.state_dict(), checkpoint)

    loaded = load_fusiondti_head(checkpoint)

    assert loaded.training is False
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert torch.equal(loaded.prot_reg.weight, source.prot_reg.weight)

    invalid_state = source.state_dict()
    invalid_state.pop("prot_reg.bias")
    torch.save(invalid_state, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_fusiondti_head(checkpoint)


def test_fusiondti_scorer_runs_structure_aware_strings_to_probabilities():
    torch.manual_seed(11)
    protein_tokenizer = TinyProteinTokenizer()
    scorer = FusionDTIActivityScorer(
        protein_encoder=TinyProteinEncoder(),
        molecule_encoder=TinyMoleculeEncoder(),
        activity_head=FusionDTIActivityHead(
            protein_dim=6,
            molecule_dim=5,
            hidden_dim=8,
            num_heads=2,
        ),
        protein_tokenizer=protein_tokenizer,
        molecule_tokenizer=tiny_molecule_tokenizer(),
        max_length=8,
        batch_size=8,
        device="cpu",
    )

    scores = scorer(
        ["MdEvLp", "MdEvLp", "AcGq"],
        ["[C]", "[O]", "[N][C]"],
    )

    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    assert scores.ge(0.0).all() and scores.le(1.0).all()
    assert scores.std(unbiased=False) > 0.0
    assert protein_tokenizer.calls == [["MdEvLp", "AcGq"]]
    assert scorer.training is False
    scorer.train()
    assert scorer.training is False
    assert all(not parameter.requires_grad for parameter in scorer.parameters())


def test_fusiondti_caches_frozen_protein_features_across_chunks_and_calls():
    protein_encoder = TinyProteinEncoder()
    scorer = FusionDTIActivityScorer(
        protein_encoder=protein_encoder,
        molecule_encoder=TinyMoleculeEncoder(),
        activity_head=FusionDTIActivityHead(
            protein_dim=6,
            molecule_dim=5,
            hidden_dim=8,
            num_heads=2,
        ),
        protein_tokenizer=TinyProteinTokenizer(),
        molecule_tokenizer=tiny_molecule_tokenizer(),
        max_length=8,
        batch_size=2,
        protein_cache_size=4,
        device="cpu",
    )

    scorer(["MdEvLp"] * 5, ["[C]", "[O]", "[N]", "[F]", "[C][O]"])
    scorer(["MdEvLp"], ["[C]"])

    assert protein_encoder.calls == 1


def test_fusiondti_rejects_proteins_that_would_be_truncated():
    scorer = FusionDTIActivityScorer(
        protein_encoder=TinyProteinEncoder(),
        molecule_encoder=TinyMoleculeEncoder(),
        activity_head=FusionDTIActivityHead(
            protein_dim=6,
            molecule_dim=5,
            hidden_dim=8,
            num_heads=2,
        ),
        protein_tokenizer=TinyProteinTokenizer(),
        molecule_tokenizer=tiny_molecule_tokenizer(),
        max_length=5,
        batch_size=1,
        protein_cache_size=1,
        device="cpu",
    )

    with pytest.raises(ValueError, match="no-truncation residue limit"):
        scorer(["AaCaDaEa"], ["[C]"])


def test_fusiondti_artifact_resolver_rejects_unknown_dataset_without_network():
    with pytest.raises(ValueError, match="unsupported FusionDTI dataset"):
        download_fusiondti_artifacts("not-a-dataset")
