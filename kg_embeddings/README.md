# KG Embeddings Generation

This folder contains scripts for training and evaluating KG embeddings.

## Contents

- `hpo_rotate.py`, `train_rotate.py`: RotatE HPO/training.
- `hpo_tucker.py`, `train_tucker.py`: TuckER HPO/training.
- `hpo_rdf2vec.py`, `train_rdf2vec.py`: RDF2Vec HPO/training.
- `evaluate_pykeen.py`: evaluate PyKEEN models.
- `evaluate_rdf2vec_hpo.py`: evaluate RDF2Vec HPO configurations.
- `extract_embeddings.py`: export entity and relation embeddings.
- `src/`: shared loading, training, and embedding I/O utilities.

## Overview

### 1. RotatE

yKEEN's RotatE **defaults to `MarginRankingLoss(margin=1.0)`** (inherited from the base `Model` class). We override to `NSSALoss`, which is the loss function proposed in the original RotatE paper.

RotatE represents embeddings as **complex numbers**. PyKEEN's `embedding_dim` parameter counts complex values, not real dimensions:
- `embedding_dim=100` → 100 complex values → **200 real dimensions** (100 real + 100 imaginary)
- Extraction: `evaluate_pykeen.py:_complex_to_real()` stacks `[real_parts | imag_parts]`


### 2. TuckER

PyKEEN's `TuckER` model correctly defaults to `BCEAfterSigmoidLoss` (set in `tucker.py:loss_default`), matching the original paper's binary cross-entropy loss. No override needed.


### 3. RDF2Vec


RDF2Vec operates in two phases:
1. **Random Walk Generation**: Walk the KG graph, generating sequences of entities and relations
2. **Word2Vec Training**: Treat walks as "sentences" and entity/relation names as "words"; train skip-gram Word2Vec.

RDF2Vec natively produces only entity embeddings (relations appear in walks but are not standalone vocabulary items in the same way). We compute relation embeddings as the **mean vector** of all entities participating in each relation type. 


## Expected Outputs

The alignment code expects selected embeddings in this structure:

```text
kg_embeddings/outputs/selected/
├── rdf2vec/<config>/entity_embeddings.csv
├── rdf2vec/<config>/relation_embeddings.csv
├── rotate/<config>/entity_embeddings.csv
├── rotate/<config>/relation_embeddings.csv
├── tucker/<config>/entity_embeddings.csv
└── tucker/<config>/relation_embeddings.csv
```


## External Resources

- **RotatE**: https://arxiv.org/abs/1902.10197
- **TuckER**: https://arxiv.org/abs/1901.09590
- **RDF2Vec**: https://link.springer.com/chapter/10.1007/978-3-319-46523-4_30
- **PyKEEN Documentation about Loss Functions**: https://pykeen.readthedocs.io/en/stable/reference/losses.html
- **PyKEEN Documentation about Negative Sampling**: https://pykeen.readthedocs.io/en/stable/reference/negative_sampling.html
