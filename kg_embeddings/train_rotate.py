"""
RotatE Training Script (CPU, no evaluation)

Trains 200-dimensional RotatE embeddings on kg_full.tsv
with checkpoint support for resume. Evaluation runs separately.

Hyperparameters based on Sun et al. 2019 (https://arxiv.org/abs/1902.10197)

Usage:
    python train_rotate.py

Output:
    outputs/pykeen/rotate/
        - checkpoints/rotate_checkpoint.pt  (training checkpoint for resume)
        - model/trained_model.pt            (final model)
        - model/training_factory.pkl        (entity/relation mappings)
        - model/data_splits.pkl             (train/val/test splits)
"""

import sys
import pickle
import yaml
from pathlib import Path

from loguru import logger

# Configure loguru for real-time output (important for cluster monitoring)
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
    colorize=True,
)
# Also log to file for persistence
logger.add(
    "logs/rotate_training.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",
    rotation="100 MB",
)

from src.data_loader import tsv_to_pykeen_factory, create_train_test_split
from src.pykeen_trainer import train_no_eval


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_rotate(training, validation, testing, config, output_dir):
    """Train RotatE model without evaluation, save checkpoints and data splits."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    model_dir = output_dir / "model"

    # Get RotatE-specific config
    rotate_config = config['pykeen']['RotatE']

    # Build training config
    # RotatE uses complex embeddings: embedding_dim counts complex values,
    # so 100 complex → 200 real dimensions after extraction.
    pykeen_config = {
        'embedding_dim': rotate_config.get('embedding_dim', config['pykeen']['embedding_dim']),
        'num_epochs': config['pykeen']['num_epochs'],
        'random_seed': config['pykeen']['random_seed'],
        # RotatE-specific hyperparameters
        'batch_size': rotate_config.get('batch_size', 512),
        'learning_rate': rotate_config.get('learning_rate', 0.0005),
        'num_negs_per_pos': rotate_config.get('num_negs_per_pos', 50),
        # NSSALoss parameters (Sun et al. 2019, WN18RR best config)
        'margin': rotate_config.get('margin', 6.0),
        'adversarial_temperature': rotate_config.get('adversarial_temperature', 0.5),
    }

    logger.info(f"Training RotatE with config:")
    logger.info(f"  - learning_rate: {pykeen_config['learning_rate']}")
    logger.info(f"  - batch_size: {pykeen_config['batch_size']}")
    logger.info(f"  - num_negs_per_pos: {pykeen_config['num_negs_per_pos']}")
    logger.info(f"  - margin (gamma): {pykeen_config['margin']}")
    logger.info(f"  - adversarial_temperature (alpha): {pykeen_config['adversarial_temperature']}")
    logger.info(f"  - num_epochs: {pykeen_config['num_epochs']}")

    # Train (no evaluation - will be done separately)
    model = train_no_eval(
        training=training,
        model_name="RotatE",
        config=pykeen_config,
        checkpoint_dir=checkpoint_dir,
        output_dir=model_dir,
    )

    # Save data splits for later evaluation
    splits_path = model_dir / "data_splits.pkl"
    with open(splits_path, "wb") as f:
        pickle.dump({
            "training": training,
            "validation": validation,
            "testing": testing,
        }, f)
    logger.info(f"Saved data splits to {splits_path}")

    return model


def main():
    """Main training pipeline for RotatE."""
    config = load_config()

    logger.info("=" * 60)
    logger.info("RotatE Training Pipeline (CPU, no evaluation)")
    logger.info("=" * 60)
    logger.info(f"Data: {config['data']['kg_path']}")
    logger.info(f"Embedding dimension: {config['pykeen']['embedding_dim']}")

    # Load data
    logger.info("Loading data...")
    factory = tsv_to_pykeen_factory(config['data']['kg_path'])
    logger.info(f"Loaded {factory.num_triples:,} triples, {factory.num_entities:,} entities, {factory.num_relations} relations")

    # Split
    logger.info("Creating train/val/test split...")
    training, validation, testing = create_train_test_split(
        factory,
        train_ratio=config['data']['train_ratio'],
        validation_ratio=config['data']['validation_ratio'],
        random_state=config['data']['random_seed']
    )
    logger.info(f"  - Training: {training.num_triples:,} triples")
    logger.info(f"  - Validation: {validation.num_triples:,} triples")
    logger.info(f"  - Testing: {testing.num_triples:,} triples")

    # Train RotatE
    logger.info("\nTraining RotatE...")
    train_rotate(training, validation, testing, config, "outputs/pykeen/rotate")

    logger.info("\n" + "=" * 60)
    logger.info("RotatE training complete!")
    logger.info("Output: outputs/pykeen/rotate/")
    logger.info("Next step: sbatch evaluate_pykeen.sh rotate")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
