"""
TuckER Hyperparameter Optimization via PyKEEN hpo_pipeline()

40 Optuna trials, MRR objective, LCWA + BCEAfterSigmoidLoss, A100 GPU.
All trial models saved to disk for top-K extraction.

Usage:
    python hpo_tucker.py
"""

import sys
import pickle
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")
logger.add("logs/hpo_tucker.log", format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}", level="DEBUG", rotation="100 MB")

from pykeen.hpo import hpo_pipeline


def main():
    logger.info("=" * 60)
    logger.info("TuckER HPO Pipeline (LCWA)")
    logger.info("=" * 60)

    # Load pre-saved split
    split_dir = Path("outputs/split")
    with open(split_dir / "training_factory.pkl", "rb") as f:
        training = pickle.load(f)
    with open(split_dir / "validation_factory.pkl", "rb") as f:
        validation = pickle.load(f)

    logger.info(f"Training: {training.num_triples:,} triples, {training.num_entities:,} entities, {training.num_relations} relations")
    logger.info(f"Validation: {validation.num_triples:,} triples")

    # Create output directories
    save_dir = Path("outputs/hpo/tucker/trials")
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting HPO: 40 trials, MRR objective, LCWA + BCEAfterSigmoidLoss")
    logger.info("Search space: embedding_dim, relation_dim, dropouts, lr, label_smoothing")

    # PyKEEN hpo_pipeline requires both training AND testing.
    # Since we have no test set, pass validation as testing.
    hpo_result = hpo_pipeline(
        n_trials=40,
        training=training,
        testing=validation,
        validation=validation,

        # Model
        model="TuckER",
        model_kwargs=dict(apply_batch_normalization=True),
        model_kwargs_ranges=dict(
            embedding_dim=dict(type=int, low=100, high=300, q=100),
            relation_dim=dict(type=int, low=30, high=100, q=35),
            dropout_0=dict(type=float, low=0.1, high=0.3),
            dropout_1=dict(type=float, low=0.1, high=0.4),
            dropout_2=dict(type=float, low=0.2, high=0.5),
        ),

        # Loss — BCEAfterSigmoidLoss (TuckER default, matches paper)
        loss="BCEAfterSigmoidLoss",

        # Training loop — LCWA (paper methodology, NOT SLCWA)
        training_loop="LCWA",

        # Optimizer — Adam (paper choice)
        optimizer="Adam",
        optimizer_kwargs_ranges=dict(
            lr=dict(type=float, low=5e-4, high=5e-3, log=True),
        ),

        # Training config
        epochs=500,
        training_kwargs=dict(
            batch_size=128,
        ),
        training_kwargs_ranges=dict(
            label_smoothing=dict(type=float, low=0.0, high=0.2, q=0.1),
        ),

        # Early stopping per trial
        stopper="early",
        stopper_kwargs=dict(
            frequency=10,
            patience=5,
            relative_delta=0.01,
            metric="mean_reciprocal_rank",
        ),

        # HPO objective
        metric="mean_reciprocal_rank",

        # Save ALL trial models
        save_model_directory=str(save_dir),

        # Crash recovery
        storage="sqlite:///outputs/hpo/tucker/optuna.db",
        load_if_exists=True,
        study_name="tucker_hpo",

        # MLflow tracking
        result_tracker="mlflow",
        result_tracker_kwargs=dict(
            tracking_uri="file:./outputs/mlruns",
            experiment_name="tucker_hpo",
        ),
    )

    # Save trials dataframe
    study = hpo_result.study
    trials_df = study.trials_dataframe()
    trials_df.to_csv("outputs/hpo/tucker/hpo_trials.csv", index=False)
    logger.info(f"Saved {len(trials_df)} trial results to outputs/hpo/tucker/hpo_trials.csv")

    # Print best result
    best = study.best_trial
    logger.info("=" * 60)
    logger.info(f"Best trial: #{best.number}")
    logger.info(f"Best MRR: {best.value:.4f}")
    logger.info(f"Best params: {best.params}")
    logger.info("=" * 60)

    hpo_result.save_to_directory("outputs/hpo/tucker")
    logger.info("HPO complete. Run: python extract_embeddings.py --model tucker --top_k 10")


if __name__ == "__main__":
    main()
