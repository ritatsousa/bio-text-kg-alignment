# Alignment Experiments

This folder contains the final alignment code used to map between biomedical
text embeddings and KG triple embeddings.

## Contents

- `run_hpo.py`: run all HPO cells sequentially.
- `run_baselines.py`: baseline evaluation.
- `data.py`: loading, split creation, and triple-combination strategies.
- `model.py`: alignment architectures.
- `trainer.py`: training loop, early stopping, and evaluation calls.
- `losses.py`: InfoNCE loss.
- `negatives.py`: hard-negative generation strategies.
- `evaluation.py`: ranking metrics.
- `config.py`: configuration schema.

