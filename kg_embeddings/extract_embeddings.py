"""
Extract entity and relation embeddings from top-K HPO trial models.

Loads Optuna study, ranks by MRR, loads saved models, extracts embeddings.

Usage:
    python extract_embeddings.py --model rotate --top_k 10
    python extract_embeddings.py --model tucker --top_k 10
"""

import sys
import json
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")

import torch
import optuna


def load_trial_model(trial_dir):
    """Load a PyKEEN model saved by hpo_pipeline's save_model_directory."""
    trial_dir = Path(trial_dir)

    # PipelineResult.save_to_directory() saves:
    # - trained_model.pkl (the model)
    # - training_triples/ (factory)
    # - metadata.json
    model_path = trial_dir / "trained_model.pkl"
    if model_path.exists():
        model = torch.load(model_path, map_location="cpu")
        model.eval()
        return model

    # Alternative: PyKEEN may save as results.pkl
    results_path = trial_dir / "results.pkl"
    if results_path.exists():
        with open(results_path, "rb") as f:
            result = pickle.load(f)
        result.model.eval()
        return result.model

    raise FileNotFoundError(f"No model found in {trial_dir}. Files: {list(trial_dir.iterdir())}")


def extract_from_model(model, training_factory):
    """Extract entity and relation embeddings from a PyKEEN model."""
    with torch.no_grad():
        entity_indices = torch.arange(training_factory.num_entities, device=model.device)
        entity_emb = model.entity_representations[0](entity_indices).cpu().numpy()

        relation_indices = torch.arange(training_factory.num_relations, device=model.device)
        relation_emb = model.relation_representations[0](relation_indices).cpu().numpy()

    # RotatE: complex → real
    if np.iscomplexobj(entity_emb):
        entity_emb = np.column_stack([entity_emb.real, entity_emb.imag]).astype(np.float32)
        logger.info(f"  Converted complex entity embeddings → {entity_emb.shape}")
    if np.iscomplexobj(relation_emb):
        relation_emb = np.column_stack([relation_emb.real, relation_emb.imag]).astype(np.float32)

    return entity_emb.astype(np.float32), relation_emb.astype(np.float32)


def save_embeddings_csv(embeddings, id_mapping, path, id_col="entity_id"):
    """Save embeddings to CSV with entity/relation IDs."""
    idx_to_id = {v: k for k, v in id_mapping.items()}
    data = {id_col: [idx_to_id[i] for i in range(len(embeddings))]}
    for i in range(embeddings.shape[1]):
        data[f"dim_{i}"] = embeddings[:, i]
    pd.DataFrame(data).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["rotate", "tucker"])
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    model_name = args.model
    top_k = args.top_k
    hpo_dir = Path(f"outputs/hpo/{model_name}")
    trials_dir = hpo_dir / "trials"
    db_path = hpo_dir / "optuna.db"

    logger.info(f"Extracting top-{top_k} {model_name.upper()} embeddings")

    # Load training factory (for entity/relation mappings)
    with open("outputs/split/training_factory.pkl", "rb") as f:
        training_factory = pickle.load(f)

    # Load Optuna study
    study_name = f"{model_name}_hpo"
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
    )

    # Rank trials by MRR (maximize)
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_trials.sort(key=lambda t: t.value, reverse=True)

    logger.info(f"Found {len(completed_trials)} completed trials. Extracting top-{top_k}.")

    all_results = []

    for rank, trial in enumerate(completed_trials[:top_k], 1):
        trial_dir = trials_dir / str(trial.number)
        out_dir = hpo_dir / f"rank{rank}"
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Rank {rank}: trial #{trial.number}, MRR={trial.value:.4f}")

        try:
            model = load_trial_model(trial_dir)
            entity_emb, relation_emb = extract_from_model(model, training_factory)

            # Save embeddings
            save_embeddings_csv(entity_emb, training_factory.entity_to_id, out_dir / "entity_embeddings.csv")
            save_embeddings_csv(relation_emb, training_factory.relation_to_id, out_dir / "relation_embeddings.csv", id_col="relation_id")
            logger.info(f"  Saved: entity={entity_emb.shape}, relation={relation_emb.shape}")

            # Save training summary
            summary = {
                "model": model_name,
                "rank": rank,
                "trial_number": trial.number,
                "mrr": trial.value,
                "hyperparameters": trial.params,
                "embedding_dim_entity": entity_emb.shape[1],
                "embedding_dim_relation": relation_emb.shape[1],
                "num_entities": entity_emb.shape[0],
                "num_relations": relation_emb.shape[0],
            }
            with open(out_dir / "training_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            all_results.append({
                "model": model_name,
                "rank": rank,
                "trial_number": trial.number,
                "mrr": trial.value,
                "entity_dim": entity_emb.shape[1],
                **trial.params,
            })

        except Exception as e:
            logger.error(f"  Failed: {e}")
            continue

    # Save combined results
    if all_results:
        results_path = hpo_dir / "results_summary.csv"
        pd.DataFrame(all_results).to_csv(results_path, index=False)
        logger.info(f"\nResults saved to {results_path}")

    logger.info("Extraction complete.")


if __name__ == "__main__":
    main()
