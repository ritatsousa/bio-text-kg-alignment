"""
PyKEEN Training Module

This module provides functions to train knowledge graph embeddings using PyKEEN.
Supports TuckER and RotatE models.

Includes detailed logging for monitoring training progress on clusters.
"""

import os
import time
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from loguru import logger

# Try importing PyKEEN components
try:
    import torch
    from torch.optim import Adam
    from pykeen.triples import TriplesFactory
    from pykeen.models import Model, RotatE, TuckER
    from pykeen.training import SLCWATrainingLoop
    from pykeen.training.callbacks import TrainingCallback
    from pykeen.losses import NSSALoss
    PYKEEN_AVAILABLE = True
except ImportError:
    PYKEEN_AVAILABLE = False
    logger.warning("PyKEEN not installed. PyKEEN functions will not work.")


if PYKEEN_AVAILABLE:
    class TrainingProgressCallback(TrainingCallback):
        """
        Custom callback for detailed training progress logging.
        Logs loss, timing, trend, and periodic summaries.
        """

        def __init__(self, log_interval: int = 5, summary_interval: int = 50, total_epochs: int = 1000):
            super().__init__()
            self.log_interval = log_interval
            self.summary_interval = summary_interval
            self.total_epochs = total_epochs
            self.start_time = None
            self.epoch_start_time = None
            self.losses = []
            self.epoch_durations = []
            self.best_loss = float('inf')
            self.best_epoch = 0

        def pre_epoch(self, epoch: int, **kwargs) -> None:
            """Called before each epoch - record start time."""
            self.epoch_start_time = time.time()

        def post_epoch(self, epoch: int, epoch_loss: float, **kwargs) -> None:
            """Called after each epoch - log progress."""
            now = time.time()
            if self.start_time is None:
                self.start_time = now

            self.losses.append(epoch_loss)

            # Track epoch duration
            if self.epoch_start_time is not None:
                epoch_duration = now - self.epoch_start_time
                self.epoch_durations.append(epoch_duration)

            # Track best loss
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.best_epoch = epoch

            elapsed = now - self.start_time

            # Log at specified interval or first epoch
            if epoch % self.log_interval == 0 or epoch == 1:
                # ETA based on average epoch time
                if len(self.epoch_durations) >= 2:
                    avg_epoch_time = np.mean(self.epoch_durations[-20:])
                    remaining_epochs = self.total_epochs - epoch
                    eta_seconds = avg_epoch_time * remaining_epochs
                    eta_str = self._format_time(eta_seconds)
                    epoch_time_str = f"{avg_epoch_time:.1f}s/epoch"
                else:
                    eta_str = "calculating..."
                    epoch_time_str = "..."

                # Loss trend arrow
                if len(self.losses) >= 10:
                    recent_loss = np.mean(self.losses[-10:])
                    older_loss = np.mean(self.losses[-20:-10]) if len(self.losses) >= 20 else self.losses[0]
                    loss_trend = "↓" if recent_loss < older_loss else "↑" if recent_loss > older_loss else "→"
                else:
                    loss_trend = ""

                logger.info(
                    f"Epoch {epoch:4d}/{self.total_epochs} | "
                    f"Loss: {epoch_loss:.6f} {loss_trend} | "
                    f"Best: {self.best_loss:.6f} (ep {self.best_epoch}) | "
                    f"{epoch_time_str} | "
                    f"Elapsed: {self._format_time(elapsed)} | "
                    f"ETA: {eta_str}"
                )

            # Periodic summary every summary_interval epochs
            if epoch % self.summary_interval == 0 and epoch > 0:
                self._log_summary(epoch, elapsed)

        def _log_summary(self, epoch: int, elapsed: float) -> None:
            """Log a detailed summary of training progress."""
            logger.info("=" * 70)
            logger.info(f"TRAINING SUMMARY — Epoch {epoch}/{self.total_epochs} ({100*epoch/self.total_epochs:.1f}%)")
            logger.info(f"  Current loss:    {self.losses[-1]:.6f}")
            logger.info(f"  Best loss:       {self.best_loss:.6f} (epoch {self.best_epoch})")
            if len(self.losses) >= 50:
                first_50_avg = np.mean(self.losses[:50])
                last_50_avg = np.mean(self.losses[-50:])
                improvement = (first_50_avg - last_50_avg) / first_50_avg * 100
                logger.info(f"  Loss improvement: {improvement:+.2f}% (first 50 avg → last 50 avg)")
            if len(self.epoch_durations) >= 5:
                avg_time = np.mean(self.epoch_durations[-20:])
                remaining = self.total_epochs - epoch
                logger.info(f"  Avg epoch time:  {avg_time:.1f}s ({self._format_time(avg_time)} per epoch)")
                logger.info(f"  Time elapsed:    {self._format_time(elapsed)}")
                logger.info(f"  ETA remaining:   {self._format_time(avg_time * remaining)}")
                logger.info(f"  Est. total time: {self._format_time(elapsed + avg_time * remaining)}")
            logger.info("=" * 70)

        def _format_time(self, seconds: float) -> str:
            """Format seconds into human-readable string."""
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            else:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                return f"{hours}h{minutes:02d}m"


    # Default hyperparameters based on biomedical benchmark best practices (Chang et al.)
    DEFAULT_PYKEEN_CONFIG = {
        # Model parameters
        "embedding_dim": 200,
        "relation_dim": 200,

        # Training parameters
        "num_epochs": 1000,
        "batch_size": 256,
        "learning_rate": 0.01,

        # Negative sampling
        "num_negs_per_pos": 50,

        # Early stopping
        "patience": 30,
        "metric": "hits@10",

        # Regularization
        "regularizer": "lp",
        "regularizer_weight": 1e-6,

        # Device
        "device": "cuda" if torch.cuda.is_available() else "cpu",

        # Random seed
        "random_seed": 42,
    }

# Model-specific configurations (TuckER and RotatE only)
MODEL_CONFIGS = {
    "TuckER": {
        "embedding_dim": 200,
        "relation_dim": 200,
        "dropout_0": 0.3,
        "dropout_1": 0.4,
        "dropout_2": 0.5,
        "apply_batch_normalization": True,
    },
    "RotatE": {
        "embedding_dim": 100,  # 100 complex values → 200 real dimensions
    },
}


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def get_model_kwargs(
    model_name: str,
    embedding_dim: int = 200,
    relation_dim: int = 200,
    **overrides,
) -> Dict:
    """
    Get model-specific keyword arguments.

    Args:
        model_name: Name of the model (TuckER, RotatE)
        embedding_dim: Entity embedding dimension
        relation_dim: Relation embedding dimension
        **overrides: Additional kwargs to override MODEL_CONFIGS defaults
                     (e.g., dropout_0, dropout_1, dropout_2 from config.yaml)

    Returns:
        Dictionary of model-specific kwargs
    """
    base_kwargs = {"embedding_dim": embedding_dim}

    if model_name in MODEL_CONFIGS:
        model_kwargs = MODEL_CONFIGS[model_name].copy()
        model_kwargs["embedding_dim"] = embedding_dim
        if "relation_dim" in model_kwargs:
            model_kwargs["relation_dim"] = relation_dim
        # Apply overrides for keys that exist in MODEL_CONFIGS
        for key, value in overrides.items():
            if key in model_kwargs:
                model_kwargs[key] = value
        return model_kwargs

    return base_kwargs


def save_model(
    model: "Model",
    training: "TriplesFactory",
    output_dir: Path,
) -> None:
    """
    Save a trained PyKEEN model and its training factory to disk.

    Args:
        model: Trained PyKEEN model
        training: TriplesFactory used for training (contains entity/relation mappings)
        output_dir: Directory to save model files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "trained_model.pt"
    factory_path = output_dir / "training_factory.pkl"

    torch.save(model, model_path)
    logger.info(f"Saved model to {model_path}")

    with open(factory_path, "wb") as f:
        pickle.dump(training, f)
    logger.info(f"Saved training factory to {factory_path}")


def load_model(
    model_dir: str,
    device: str = "cpu",
) -> Tuple["Model", "TriplesFactory"]:
    """
    Load a saved PyKEEN model and its training factory from disk.

    Args:
        model_dir: Directory containing trained_model.pt and training_factory.pkl
        device: Device to load model onto (default: 'cpu')

    Returns:
        Tuple of (model, training_factory)
    """
    if not PYKEEN_AVAILABLE:
        raise ImportError("PyKEEN not installed. Run: pip install pykeen")

    model_dir = Path(model_dir)

    model_path = model_dir / "trained_model.pt"
    factory_path = model_dir / "training_factory.pkl"

    model = torch.load(model_path, map_location=device)
    model.eval()
    logger.info(f"Loaded model from {model_path} (device={device})")

    with open(factory_path, "rb") as f:
        factory = pickle.load(f)
    logger.info(f"Loaded training factory: {factory.num_entities} entities, {factory.num_relations} relations")

    return model, factory


def train_no_eval(
    training: "TriplesFactory",
    model_name: str,
    config: Dict,
    checkpoint_dir: Path,
    output_dir: Path,
) -> "Model":
    """
    Train a PyKEEN model without evaluation (for CPU training where eval is too slow).

    Uses SLCWATrainingLoop directly with checkpoint support for resume.
    After training completes, saves the model via save_model().

    Args:
        training: TriplesFactory for training data
        model_name: Model name ('RotatE' or 'TuckER')
        config: Configuration dictionary with hyperparameters
        checkpoint_dir: Directory for training checkpoints (for resume)
        output_dir: Directory to save final model

    Returns:
        Trained model
    """
    if not PYKEEN_AVAILABLE:
        raise ImportError("PyKEEN not installed. Run: pip install pykeen")

    # Set CPU threading for optimal performance
    num_threads = int(os.environ.get("OMP_NUM_THREADS", 1))
    torch.set_num_threads(num_threads)
    logger.info(f"Set torch num_threads={num_threads}")

    # Merge with defaults
    cfg = DEFAULT_PYKEEN_CONFIG.copy()
    cfg.update(config)

    # Ensure directories exist
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model-specific kwargs (overrides propagate config.yaml values to MODEL_CONFIGS)
    model_kwargs = get_model_kwargs(
        model_name,
        embedding_dim=cfg["embedding_dim"],
        relation_dim=cfg.get("relation_dim", cfg["embedding_dim"]),
        dropout_0=cfg.get("dropout_0", 0.3),
        dropout_1=cfg.get("dropout_1", 0.4),
        dropout_2=cfg.get("dropout_2", 0.5),
        apply_batch_normalization=cfg.get("apply_batch_normalization", True),
    )

    # Create model
    model_cls = {"RotatE": RotatE, "TuckER": TuckER}[model_name]
    model = model_cls(triples_factory=training, **model_kwargs)

    # Override loss for RotatE: use NSSALoss (self-adversarial negative sampling)
    # per Sun et al. 2019. PyKEEN defaults to MarginRankingLoss(margin=1.0).
    if model_name == "RotatE":
        model.loss = NSSALoss(
            margin=cfg.get("margin", 6.0),
            adversarial_temperature=cfg.get("adversarial_temperature", 0.5),
        )

    logger.info(f"Created {model_name} model: {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Loss function: {model.loss.__class__.__name__}")
    logger.info(f"Training data: {training.num_triples:,} triples, {training.num_entities:,} entities, {training.num_relations} relations")
    logger.info(f"Config: embedding_dim={cfg['embedding_dim']}, batch_size={cfg['batch_size']}, epochs={cfg['num_epochs']}")
    logger.info(f"Device: cpu (no evaluation)")

    # Create optimizer
    optimizer = Adam(model.parameters(), lr=cfg["learning_rate"])

    # Create training loop with negative sampling config
    num_negs = cfg.get("num_negs_per_pos", 50)
    loop = SLCWATrainingLoop(
        model=model,
        triples_factory=training,
        optimizer=optimizer,
        negative_sampler_kwargs=dict(
            num_negs_per_pos=num_negs,
        ),
    )
    logger.info(f"Negative sampling: num_negs_per_pos={num_negs}")

    # Create progress callback
    progress_callback = TrainingProgressCallback(
        log_interval=10,
        total_epochs=cfg["num_epochs"],
    )

    # Check for existing checkpoint (for resume)
    checkpoint_name = f"{model_name.lower()}_checkpoint.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name
    if checkpoint_path.exists():
        logger.info(f"Found existing checkpoint: {checkpoint_path}")
        logger.info("Training will resume from checkpoint")

    start_time = time.time()

    # Train with checkpointing
    label_smoothing = cfg.get("label_smoothing", 0.0)
    if label_smoothing > 0:
        logger.info(f"Label smoothing: {label_smoothing}")
    loop.train(
        triples_factory=training,
        num_epochs=cfg["num_epochs"],
        batch_size=cfg["batch_size"],
        label_smoothing=label_smoothing,
        checkpoint_name=checkpoint_name,
        checkpoint_directory=str(checkpoint_dir),
        checkpoint_frequency=5,  # minutes (0 = every epoch)
        checkpoint_on_failure=True,
        callbacks=[progress_callback],
    )

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {format_elapsed_time(elapsed)}")

    # Save final model
    save_model(model, training, output_dir)

    return model


if __name__ == "__main__":
    # Test the module
    print("Testing pykeen_trainer module...")

    if PYKEEN_AVAILABLE:
        print("PyKEEN is available")
        print(f"Default config: {DEFAULT_PYKEEN_CONFIG}")
        print(f"Available device: {DEFAULT_PYKEEN_CONFIG['device']}")

        # Test model kwargs
        for model_name in ["TuckER", "RotatE"]:
            kwargs = get_model_kwargs(model_name)
            print(f"{model_name} kwargs: {kwargs}")
    else:
        print("PyKEEN not installed - skipping tests")
