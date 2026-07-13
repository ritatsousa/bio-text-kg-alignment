"""
Prepare and save the 95/5 train/val split for all methods.

Run once before HPO. Saves split to outputs/split/ for reuse by all training scripts.
This ensures RotatE, TuckER, and RDF2Vec all train on the same 95% of data.

Usage:
    python prepare_split.py
"""

import sys
import pickle
import yaml
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")

from src.data_loader import tsv_to_pykeen_factory, create_train_val_split, load_tsv_triples


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    kg_path = config["data"]["kg_path"]
    train_ratio = config["data"]["train_ratio"]
    split_method = config["data"].get("split_method", "cleanup")
    seed = config["data"]["random_seed"]

    logger.info(f"Loading {kg_path}...")
    factory = tsv_to_pykeen_factory(kg_path)
    logger.info(f"Loaded: {factory.num_triples:,} triples, {factory.num_entities:,} entities, {factory.num_relations} relations")

    logger.info(f"Splitting {train_ratio:.0%}/{1-train_ratio:.0%} (method={split_method}, seed={seed})...")
    training, validation = create_train_val_split(
        factory, train_ratio=train_ratio, random_state=seed, method=split_method
    )

    # Save splits
    split_dir = Path("outputs/split")
    split_dir.mkdir(parents=True, exist_ok=True)

    with open(split_dir / "training_factory.pkl", "wb") as f:
        pickle.dump(training, f)
    with open(split_dir / "validation_factory.pkl", "wb") as f:
        pickle.dump(validation, f)
    with open(split_dir / "full_factory.pkl", "wb") as f:
        pickle.dump(factory, f)

    # Also save training triples as TSV for RDF2Vec (which doesn't use PyKEEN factories)
    train_triples = training.triples
    tsv_path = split_dir / "train_triples.tsv"
    with open(tsv_path, "w") as f:
        for row in train_triples:
            f.write("\t".join(row) + "\n")
    logger.info(f"Saved training triples TSV: {tsv_path} ({len(train_triples):,} triples)")

    logger.info(f"Splits saved to {split_dir}/")
    logger.info(f"  training_factory.pkl:  {training.num_triples:,} triples")
    logger.info(f"  validation_factory.pkl: {validation.num_triples:,} triples")
    logger.info(f"  train_triples.tsv:     {len(train_triples):,} triples (for RDF2Vec)")


if __name__ == "__main__":
    main()
