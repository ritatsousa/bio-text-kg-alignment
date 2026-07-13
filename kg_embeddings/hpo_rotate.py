"""
RotatE Hyperparameter Optimization via PyKEEN hpo_pipeline()

40 Optuna trials, MRR objective, SLCWA + NSSALoss, A100 GPU.
All trial models saved to disk for top-K extraction.

Usage:
    python hpo_rotate.py
"""

import sys
import pickle
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")
logger.add("logs/hpo_rotate.log", format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}", level="DEBUG", rotation="100 MB")

from pykeen.hpo import hpo_pipeline
from pykeen.losses import NSSALoss


def main():
    logger.info("=" * 60)
    logger.info("RotatE HPO Pipeline")
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
    save_dir = Path("outputs/hpo/rotate/trials")
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting HPO: 40 trials, MRR objective, SLCWA + NSSALoss")
    logger.info("Search space: embedding_dim, margin, adversarial_temp, lr, num_negs")

    # PyKEEN hpo_pipeline requires both training AND testing.
    # Since we have no test set, pass validation as testing.
    # HPO evaluates on this set; early stopping also monitors it.
    hpo_result = hpo_pipeline(
        n_trials=40,
        training=training,
        testing=validation,
        validation=validation,

        # Model
        model="RotatE",
        model_kwargs_ranges=dict(
            embedding_dim=dict(type=int, low=50, high=250, q=50),
        ),

        # Loss — NSSALoss (RotatE paper core method)
        loss=NSSALoss,
        loss_kwargs_ranges=dict(
            margin=dict(type=int, low=3, high=12, q=3),
            adversarial_temperature=dict(type=float, low=0.5, high=1.0),
        ),

        # Training loop
        training_loop="sLCWA",

        # Optimizer — Adam (paper uses Adam, NOT PyKEEN default Adagrad)
        optimizer="Adam",
        optimizer_kwargs_ranges=dict(
            lr=dict(type=float, low=1e-5, high=5e-4, log=True),
        ),

        # Negative sampler
        negative_sampler="basic",
        negative_sampler_kwargs_ranges=dict(
            num_negs_per_pos=dict(type=int, low=50, high=500, log=True),
        ),

        # Training config
        epochs=500,
        training_kwargs=dict(
            batch_size=512,
        ),

        # Early stopping per trial
        stopper="early",
        stopper_kwargs=dict(
            frequency=10,
            patience=10,
            relative_delta=0.002,
            metric="mean_reciprocal_rank",
        ),

        # HPO objective
        metric="mean_reciprocal_rank",

        # Save ALL trial models for top-K extraction
        save_model_directory=str(save_dir),

        # Crash recovery
        storage="sqlite:///outputs/hpo/rotate/optuna.db",
        load_if_exists=True,
        study_name="rotate_hpo",

        # MLflow tracking
        result_tracker="mlflow",
        result_tracker_kwargs=dict(
            tracking_uri="file:./outputs/mlruns",
            experiment_name="rotate_hpo",
        ),
    )

    # Save trials dataframe
    study = hpo_result.study
    trials_df = study.trials_dataframe()
    trials_df.to_csv("outputs/hpo/rotate/hpo_trials.csv", index=False)
    logger.info(f"Saved {len(trials_df)} trial results to outputs/hpo/rotate/hpo_trials.csv")

    # Print best result
    best = study.best_trial
    logger.info("=" * 60)
    logger.info(f"Best trial: #{best.number}")
    logger.info(f"Best MRR: {best.value:.4f}")
    logger.info(f"Best params: {best.params}")
    logger.info("=" * 60)

    # Save study summary
    hpo_result.save_to_directory("outputs/hpo/rotate")
    logger.info("HPO complete. Run: python extract_embeddings.py --model rotate --top_k 10")


if __name__ == "__main__":
    main()
