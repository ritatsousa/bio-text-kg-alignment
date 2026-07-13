# Aligning Biomedical Text and Knowledge Graphs: A Systematic Comparison of Lightweight Alignment Strategies

This repository contains the code used to build a CTD-Align dataset, generate biomedical text and KG embeddings, and run alignment experiments between evidence sentences and KG triples.

Large generated artifacts are intentionally not included in this GitHub repo. This includes downloaded data, generated embeddings, checkpoints, Optuna databases and MLflow runs. 

## Pipeline

- `dataset_generation/`: scripts to build the CTD/PubTator dataset and KG files.
- `text_embeddings/`: scripts to generate biomedical sentence embeddings.
- `kg_embeddings/`: scripts to train RDF2Vec, RotatE, and TuckER embeddings.
- `alignment/`: scripts to train and evaluate the text-KG alignment models.

## Data

LThe alignment code expects the following structure under `Data/`:

```text
Data/
├── CTD_to_PubTator/kg_data/
│   ├── evidence_aligned_ided_medcpt_dedup.tsv
│   ├── kg_full.tsv
│   └── metadata_full.tsv
├── text_embeddings/
│   ├── embeddings_biobert_mcpt.npy
│   ├── embeddings_pubmedbert_mcpt.npy
│   └── index.tsv
└── kg_embeddings/outputs/selected/
    ├── rdf2vec/
    ├── rotate/
    └── tucker/
```

Main files: 

- `evidence_aligned_ided_medcpt_dedup.tsv`: evidence rows aligned to KG triples, with `subject_uri`, `predicate_uri`, `object_uri`, `pmid`, and `text`.
- `kg_full.tsv`: extended KG triples in tab-separated subject-predicate-object format, without a header.
- `metadata_full.tsv`: metadata triples, including `rdf:type` rows for node types.
- `embeddings_*_mcpt.npy`: aligned sentence embedding matrices.
- `index.tsv`: text embedding index used to align embeddings to evidence rows.
- `entity_embeddings.csv` and `relation_embeddings.csv`: selected KG embeddings.


## Alignment Experiments

### KG Embeddings

The KG embedding code trains or evaluates three families:

- `RDF2Vec`: random-walk generation followed by Skip-gram Word2Vec training.
- `RotatE`: PyKEEN-based link-prediction model.
- `TuckER`: PyKEEN-based link-prediction model.

Selected embeddings are exported as entity and relation CSV files and consumed by the alignment pipeline.

### Text Embeddings

Evidence texts are encoded with biomedical transformer models. The generated `.npy` embedding matrices must be aligned with `index.tsv`.

### Alignment Models

The alignment code uses grouped cross-validation by PMID and stratification by predicate where possible.

The final alignment experiments search over:

- training direction: `text_to_kg`, `kg_to_text`, `bidirectional_random`;
- text model: BioBERT/MedCPT and PubMedBERT/MedCPT variants;
- KG model family/configuration: RDF2Vec, RotatE, TuckER;
- architecture: `linear`, `mlp`, `cross_attention`;
- triple combination: `concat`, `hadamard`, `l1`, `l2`;
- hard-negative strategy: `none`, `all`, and individual triple corruptions.

The alignment objective is InfoNCE with in-batch negatives, optional hard KG negatives, temperature `0.07`, AdamW optimization, batch size `256`, maximum `100` epochs, early stopping patience `15`, and gradient clipping at `1.0`.

For each training direction, text model, KG model, architecture, triple combination, hard-negative strategy, the code runs a small Optuna search over:
- hidden size choices: `[256]` or `[512]`;
- dropout: `[0.1, 0.3]`;
- AdamW learning rate: `[1e-4, 5e-3]` on a log scale;
- weight decay: `[1e-5, 1e-2]` on a log scale.
Fixed training settings include batch size `256`, InfoNCE temperature `0.07`, maximum `100` epochs, early stopping patience `15`, and gradient clipping `1.0`.


## Run

Create an environment and install the combined dependency list:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run alignment experiments:

```bash
cd alignment
python run_hpo.py
```

