# Boltz-2-Style Affinity Dataset

This repo cannot reproduce the private Boltz-2 affinity training table exactly because the authors did not publish the curated row-level dataset. What we can reproduce is the public curation contract from the paper and generate our own compatible dataset from ChEMBL/BindingDB-like exports.

## Outputs

`data_processing/prepare_boltz2_affinity_dataset.py` writes:

- `boltz2_affinity_full.csv`: full Boltz-2-style affinity table.
- `prot2mol_training.csv`: current Prot2Mol training input with `Target_FASTA`, `Compound_SELFIES`, and `pchembl_value_Median`.
- `summary.json`: filter counts, split counts, target/assay/compound counts, and config.

## Full Schema

```text
source
supervision              # values for continuous affinity
task_type                # optimization for hit-to-lead style assays
assay_id
assay_group_id           # source:assay_id:target_id
target_id
target_chembl_id
protein_sequence
protein_cluster_90
mutation_info
compound_id
standardized_smiles
compound_selfies
activity_type            # Ki/Kd/IC50/XC50/EC50/AC50
activity_value_uM
activity_qualifier       # = or >
y_log10_uM               # Boltz-2 affinity value target
pchembl_value            # 6 - y_log10_uM
binary_label
is_censored
heavy_atom_count
is_pains
assay_size
assay_exact_size
assay_activity_std
assay_activity_iqr
split
```

## Core Filters

- Keep activity types `Ki`, `Kd`, `IC50`, `XC50`, `EC50`, `AC50`.
- Convert all values to `uM`, then compute `y_log10_uM = log10(value_uM)`.
- ChEMBL rows: confidence score `9`, target type `SINGLE PROTEIN`, binding/functional/biochemical assays, no unreliable sources.
- BindingDB rows: exclude records with more than one protein chain when chain count is present.
- Remove invalid molecules, PAINS, and molecules with more than 50 heavy atoms.
- Keep assays with enough intra-assay signal:
  - at least 10 exact measurements
  - at least 10 unique exact values
  - exact-value standard deviation at least `0.25`
  - unique-value fraction at least `0.20`
- Deduplicate after ligand standardization:
  - remove exact duplicate measurements
  - remove BindingDB rows already covered by ChEMBL when protein sequence, canonical SMILES, activity type, qualifier, and rounded `log10(uM)` value match

## Leakage Control

The script can use an external `--cluster-map` with `protein_sequence,protein_cluster_90`. For exact Boltz-2-style leakage control, generate that map with MMseqs at 90% sequence identity:

```bash
mmseqs easy-cluster sequences.fasta seq90 tmp \
  --min-seq-id 0.9 --cov-mode 0 -c 0.01
```

If no map is provided, the script falls back to exact-sequence hashes. That is deterministic and useful for local development, but it is not equivalent to the paper's 90% sequence clustering.

## Example

Build from public source dumps directly:

```bash
python data_processing/build_boltz2_affinity_from_sources.py \
  --raw-dir dataset/raw/boltz2_sources \
  --output-dir dataset/processed/boltz2_affinity
```

This downloads ChEMBL v34 SQLite and BindingDB 202504 TSV by default, extracts normalized source files, then runs the final curation/deduplication builder.

If the raw dumps are already available:

```bash
python data_processing/build_boltz2_affinity_from_sources.py \
  --chembl-sqlite /path/to/chembl_34.db \
  --bindingdb-tsv /path/to/BindingDB_All_202504_tsv.zip \
  --raw-dir dataset/raw/boltz2_sources \
  --output-dir dataset/processed/boltz2_affinity \
  --skip-download
```

Run from already extracted source CSV/TSV files:

```bash
python data_processing/prepare_boltz2_affinity_dataset.py \
  --input data/raw/chembl_affinity.csv data/raw/bindingdb_affinity.tsv \
  --output-dir data/processed/boltz2_affinity \
  --cluster-map data/processed/protein_seq90_clusters.csv
```

Then preprocess for current Prot2Mol training:

```bash
python data_processing/preprocess_dataset.py \
  --selfies_path data/processed/boltz2_affinity/prot2mol_training.csv \
  --prot_emb_model saprot \
  --max_mol_len 256 \
  --prot_max_length 1024
```

## ChEMBL Export Filters

Use ChEMBL v34 if we want to match the Boltz-2 paper. Newer ChEMBL versions are usable, but they are not the same training universe.

Required ChEMBL filters:

- `assays.confidence_score = 9`
- `target_dictionary.target_type = 'SINGLE PROTEIN'`
- one protein component sequence per target
- `assays.assay_type IN ('B', 'F')`
- `activities.standard_type IN ('Ki', 'Kd', 'IC50', 'XC50', 'EC50', 'AC50')`
- `activities.standard_value IS NOT NULL AND activities.standard_value > 0`
- `activities.standard_units IN ('nM', 'uM', 'µM', 'M', 'mM', 'pM')`
- `activities.standard_relation IN ('=', '>')`; use `=` only for the current Prot2Mol pChEMBL training export
- `compound_structures.canonical_smiles IS NOT NULL`
- recommended: `activities.standard_flag = 1`
- recommended: `activities.potential_duplicate = 0` if the column exists

Minimal SQLite-style export query:

```sql
WITH single_component_targets AS (
  SELECT
    tc.tid,
    MIN(cs.accession) AS target_accession,
    MIN(cs.sequence) AS protein_sequence,
    COUNT(DISTINCT tc.component_id) AS component_count
  FROM target_components tc
  JOIN component_sequences cs
    ON cs.component_id = tc.component_id
  WHERE cs.sequence IS NOT NULL
  GROUP BY tc.tid
  HAVING COUNT(DISTINCT tc.component_id) = 1
)
SELECT
  'ChEMBL' AS source,
  ass.chembl_id AS assay_id,
  td.chembl_id AS target_id,
  td.chembl_id AS target_chembl_id,
  sct.target_accession,
  sct.protein_sequence,
  str.canonical_smiles AS smiles,
  act.standard_type,
  act.standard_value,
  act.standard_units,
  COALESCE(act.standard_relation, '=') AS standard_relation,
  ass.confidence_score,
  td.target_type,
  ass.assay_type,
  0 AS source_unreliable,
  md.chembl_id AS molecule_chembl_id,
  act.activity_id
FROM activities act
JOIN assays ass
  ON ass.assay_id = act.assay_id
JOIN target_dictionary td
  ON td.tid = ass.tid
JOIN single_component_targets sct
  ON sct.tid = td.tid
JOIN molecule_dictionary md
  ON md.molregno = act.molregno
JOIN compound_structures str
  ON str.molregno = act.molregno
WHERE ass.confidence_score = 9
  AND td.target_type = 'SINGLE PROTEIN'
  AND ass.assay_type IN ('B', 'F')
  AND act.standard_type IN ('Ki', 'Kd', 'IC50', 'XC50', 'EC50', 'AC50')
  AND act.standard_value IS NOT NULL
  AND act.standard_value > 0
  AND act.standard_units IN ('nM', 'uM', 'µM', 'M', 'mM', 'pM')
  AND COALESCE(act.standard_relation, '=') IN ('=', '>')
  AND str.canonical_smiles IS NOT NULL
  AND act.standard_flag = 1;
```

## BindingDB Export Filters

BindingDB should be used as an additional continuous-affinity source after ChEMBL. The Boltz-2 paper says BindingDB records were retained only when not already covered by ChEMBL.

Required BindingDB filters:

- one protein chain/sequence per row
- keep activity types `Ki`, `Kd`, `IC50`, `XC50`, `EC50`, `AC50`
- keep exact `=` values, and optionally `>` censored lower-bound values in the full export
- parse protein sequence from the BindingDB protein field
- use DOI as `assay_id` when available
- keep the activity qualifier
- standardize ligand SMILES downstream with the builder

BindingDB column names vary by export. The builder accepts these common logical columns:

```text
source                  # set to BindingDB
doi                     # assay id
target_id
protein_sequence
smiles
activity_type
activity_value_uM       # already converted to uM, preferred
activity_qualifier
num_protein_chains
compound_id
```

If you export raw units instead, use:

```text
standard_value
standard_units
standard_relation
standard_type
```

Run ChEMBL and BindingDB together so cross-source dedupe can happen:

```bash
python data_processing/prepare_boltz2_affinity_dataset.py \
  --input data/raw/chembl_affinity.csv data/raw/bindingdb_affinity.tsv \
  --output-dir data/processed/boltz2_affinity
```

Use `--keep-bindingdb-chembl-overlap` only for debugging. For real training data, leave overlap removal enabled.

## PubChem HTS Binary Data

`data_processing/build_binary_affinity_sources.py` builds a PubChem HTS binary source through the curated MF-PCBA binding table. This is the practical route rather than downloading the full PubChem BioAssay universe.

Default filters:

- source: `Leash-Biosciences/mf-pcba-bind`, derived from MF-PCBA
- binding assays only
- at least 100 tested compounds
- assay hit rate below 10%
- exact text-level dedupe only
- cap each assay to 50,000 protein-ligand rows
- no local RDKit standardization in this script

Run:

```bash
python data_processing/build_binary_affinity_sources.py \
  --raw-dir dataset/raw/binary_sources/pubchem_mf_pcba \
  --output-dir dataset/processed/binary_affinity/pubchem_mf_pcba \
  --cap-per-assay 50000 \
  --cap-mode keep-positives
```

Outputs:

- `pubchem_mf_pcba_binary.csv`
- `pubchem_mf_pcba_binary.parquet`
- `summary.json`

## CeMM And MIDAS Binary Data

The same builder also parses CeMM and MIDAS binary screening sources into the shared schema. All binary rows include split metadata:

```text
assay_type
screening_stage
assay_group_id
```

CeMM primary screen rules:

- input: `dataset/raw/binary_sources/cemm/extracted/finalScreen.tsv`
- ligand SMILES from CeMM fragment maps, primarily `Table-S1.csv`
- positive: `mdfClass >= 2`
- negative: `mdfClass == 0`
- drop: `mdfClass == 1`
- `assay_type = fragment_chemoproteomics`
- `screening_stage = primary_screen`
- no RDKit standardization

Run:

```bash
python data_processing/build_binary_affinity_sources.py --source cemm
```

MIDAS rules:

- inputs: Data S1 metabolites, Data S3 proteins, Data S4 measurements
- ligand SMILES from Data S1
- protein sequences from UniProt, cached in `dataset/raw/binary_sources/uniprot_sequences.json`
- positive: `q_value < 0.01`
- negative: `q_value >= 0.01`
- drop multi-accession protein rows because they are not single protein sequences
- `assay_type = metabolite_binding`
- `screening_stage = fia_ms_midas`
- no RDKit standardization

Run:

```bash
python data_processing/build_binary_affinity_sources.py --source midas
```

Merge and dedupe the binary sources:

```bash
python data_processing/build_binary_affinity_sources.py --source merge
```

Merge policy:

- exact row dedupe by `source,dataset,assay_id,protein_sequence,smiles,binary_label`
- drop conflicting `protein_sequence,smiles` pairs with both binary labels
- dedupe same-label `protein_sequence,smiles,binary_label` pairs, keeping source priority `PubChem_HTS`, then `CeMM`, then `MIDAS`

Current local merged output:

- `dataset/processed/binary_affinity/merged/binary_sources_merged.csv`
- `dataset/processed/binary_affinity/merged/binary_sources_merged.parquet`
- `dataset/processed/binary_affinity/merged/binary_label_conflicts.csv`
- `dataset/processed/binary_affinity/merged/summary.json`

## Protein-Cluster Splits

`data_processing/build_protein_cluster_splits.py` builds the Boltz-style split views:

- dedupe is expected to happen before this step
- protein sequences are clustered with MMseqs at 90% identity
- train/val/test assignment is by protein cluster, not row
- cluster split is stratified by row, positive, and negative counts
- whole assay groups are preserved when possible, but protein-cluster leakage control takes priority
- ranking pairs are generated after split from exact ChEMBL/BindingDB affinity rows

Run:

```bash
python data_processing/build_protein_cluster_splits.py \
  --output-dir dataset/processed/splits/protein_cluster_90 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --ranking-max-pairs-per-split 500000 \
  --ranking-max-pairs-per-group 200
```

Outputs:

- `protein_cluster_90.csv`: protein sequence to 90% cluster map
- `cluster_split.csv`: cluster-level train/val/test assignment
- `binary_all_source/{train,val,test}.parquet`
- `binary_screen_only/{train,val,test}.parquet`
- `binary_chembl_bindingdb_threshold/{train,val,test}.parquet`
- `ranking_affinity/{train,val,test}.parquet`
- `ranking_pairs/{train,val,test}.parquet`
- `summary.json`

Current split checks:

- `13,843` protein sequences
- `10,718` MMseqs 90% clusters
- zero train/val/test protein-cluster overlap
- ranking pairs are capped sampled pairs with `delta_pchembl >= 0.5`
