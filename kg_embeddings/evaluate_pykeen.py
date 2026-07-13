"""
PyKEEN Evaluation Script (standalone, CPU)

Loads a trained PyKEEN model, runs rank-based evaluation on the test set,
extracts embeddings, and saves them as CSV.

Usage:
    python evaluate_pykeen.py --model rotate
    python evaluate_pykeen.py --model tucker

Expects:
    outputs/pykeen/{model}/model/
        - trained_model.pt
        - training_factory.pkl
        - data_splits.pkl

Output:
    outputs/pykeen/{model}/
        - entity_embeddings.csv
        - relation_embeddings.csv
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
from pathlib import Path

from loguru import logger

# Configure loguru
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
    colorize=True,
)

import torch
from pykeen.evaluation import RankBasedEvaluator

from src.pykeen_trainer import load_model
from src.embeddings_io import save_embeddings_csv


def _complex_to_real(embeddings: np.ndarray) -> np.ndarray:
    """Convert complex-valued embeddings to real by stacking real and imaginary parts.

    RotatE uses complex embeddings (embedding_dim/2 complex values).
    This converts them to embedding_dim real values: [real_parts | imag_parts].
    TuckER embeddings are already real and pass through unchanged.
    """
    if np.iscomplexobj(embeddings):
        real_emb = np.column_stack([embeddings.real, embeddings.imag]).astype(np.float32)
        logger.info(f"Converted complex embeddings {embeddings.shape} -> real {real_emb.shape}")
        return real_emb
    return embeddings.astype(np.float32)


def extract_embeddings_from_model(model, training_factory):
    """Extract entity and relation embeddings directly from a model."""
    with torch.no_grad():
        entity_indices = torch.arange(training_factory.num_entities, device=model.device)
        entity_representations = model.entity_representations[0]
        entity_embeddings = entity_representations(entity_indices).cpu().numpy()

        relation_indices = torch.arange(training_factory.num_relations, device=model.device)
        relation_representations = model.relation_representations[0]
        relation_embeddings = relation_representations(relation_indices).cpu().numpy()

    # RotatE uses complex embeddings — convert to real
    entity_embeddings = _complex_to_real(entity_embeddings)
    relation_embeddings = _complex_to_real(relation_embeddings)

    entity_to_idx = training_factory.entity_to_id
    relation_to_idx = training_factory.relation_to_id

    logger.info(
        f"Extracted embeddings: {entity_embeddings.shape[0]} entities "
        f"({entity_embeddings.shape[1]}-dim), "
        f"{relation_embeddings.shape[0]} relations ({relation_embeddings.shape[1]}-dim)"
    )

    return entity_embeddings, relation_embeddings, entity_to_idx, relation_to_idx


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PyKEEN model")
    parser.add_argument(
        "--model", type=str, required=True, choices=["rotate", "tucker"],
        help="Model to evaluate (rotate or tucker)"
    )
    args = parser.parse_args()

    model_name = args.model
    base_dir = Path(f"outputs/pykeen/{model_name}")
    model_dir = base_dir / "model"

    # Set CPU threading
    num_threads = int(os.environ.get("OMP_NUM_THREADS", 1))
    torch.set_num_threads(num_threads)

    # Also log to file
    logger.add(
        f"logs/evaluate_{model_name}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="100 MB",
    )

    logger.info("=" * 60)
    logger.info(f"PyKEEN Evaluation: {model_name.upper()}")
    logger.info("=" * 60)
    logger.info(f"Model dir: {model_dir}")
    logger.info(f"Torch threads: {num_threads}")

    # Load model
    logger.info("Loading trained model...")
    model, training_factory = load_model(str(model_dir), device="cpu")

    # Load data splits
    splits_path = model_dir / "data_splits.pkl"
    logger.info(f"Loading data splits from {splits_path}...")
    with open(splits_path, "rb") as f:
        splits = pickle.load(f)

    training = splits["training"]
    testing = splits["testing"]

    logger.info(f"Training: {training.num_triples:,} triples")
    logger.info(f"Testing: {testing.num_triples:,} triples")

    # Run evaluation
    logger.info("Starting rank-based evaluation (filtered)...")
    logger.info("This may take several hours on CPU with large datasets.")
    start_time = time.time()

    evaluator = RankBasedEvaluator(filtered=True)
    metric_results = evaluator.evaluate(
        model=model,
        mapped_triples=testing.mapped_triples,
        additional_filter_triples=[
            training.mapped_triples,
        ],
    )

    elapsed = time.time() - start_time
    hours = elapsed / 3600
    logger.info(f"Evaluation complete in {hours:.2f}h ({elapsed:.0f}s)")

    # Log metrics
    mrr = metric_results.get_metric("both.realistic.inverse_harmonic_mean_rank")
    hits1 = metric_results.get_metric("both.realistic.hits_at_1")
    hits3 = metric_results.get_metric("both.realistic.hits_at_3")
    hits10 = metric_results.get_metric("both.realistic.hits_at_10")
    mean_rank = metric_results.get_metric("both.realistic.arithmetic_mean_rank")

    logger.info("Test metrics:")
    logger.info(f"  MRR:       {mrr:.4f}")
    logger.info(f"  Hits@1:    {hits1:.4f}")
    logger.info(f"  Hits@3:    {hits3:.4f}")
    logger.info(f"  Hits@10:   {hits10:.4f}")
    logger.info(f"  Mean Rank: {mean_rank:.1f}")

    # Extract and save embeddings
    logger.info("Extracting embeddings...")
    entity_emb, relation_emb, entity_map, relation_map = extract_embeddings_from_model(
        model, training_factory
    )

    save_embeddings_csv(entity_emb, entity_map, base_dir / "entity_embeddings.csv")
    save_embeddings_csv(relation_emb, relation_map, base_dir / "relation_embeddings.csv", id_column="relation_id")

    logger.info(f"Entity embeddings: {entity_emb.shape}")
    logger.info(f"Relation embeddings: {relation_emb.shape}")

    logger.info("\n" + "=" * 60)
    logger.info(f"{model_name.upper()} evaluation complete!")
    logger.info(f"Output: {base_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
