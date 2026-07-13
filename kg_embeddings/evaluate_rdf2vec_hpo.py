"""
Evaluate all RDF2Vec HPO configs via k-NN entity type classification.

Loads entity type labels from metadata, computes k-NN accuracy for each config.

Usage:
    python evaluate_rdf2vec_hpo.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")


def load_type_labels(metadata_path):
    """Load entity type labels from metadata TSV."""
    entity_types = {}
    with open(metadata_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3 and parts[1] == "rdf:type":
                entity_types[parts[0]] = parts[2]
    return entity_types


def knn_type_accuracy(embeddings, entity_ids, entity_types, k=10):
    """Compute k-NN type classification accuracy."""
    # Build typed subset
    typed_indices = []
    typed_labels = []
    for i, eid in enumerate(entity_ids):
        if eid in entity_types:
            typed_indices.append(i)
            typed_labels.append(entity_types[eid])

    typed_emb = embeddings[typed_indices]
    n = len(typed_indices)
    logger.info(f"  Evaluating {n} typed entities (k={k})")

    # Normalize for cosine similarity
    norms = np.linalg.norm(typed_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = typed_emb / norms

    # Compute accuracy in batches to avoid OOM
    batch_size = 1000
    correct = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = normed[start:end]  # (batch, dim)
        sims = batch @ normed.T    # (batch, n)

        for i_local in range(end - start):
            i_global = start + i_local
            row = sims[i_local]
            # Exclude self
            row[i_global] = -1
            top_k_idx = np.argpartition(row, -k)[-k:]
            neighbor_labels = [typed_labels[j] for j in top_k_idx]
            most_common = Counter(neighbor_labels).most_common(1)[0][0]
            if most_common == typed_labels[i_global]:
                correct += 1

    return correct / n


def main():
    metadata_path = "kg_core_v2/metadata_full.tsv"
    hpo_dir = Path("outputs/hpo/rdf2vec")

    logger.info("Loading entity type labels...")
    entity_types = load_type_labels(metadata_path)
    logger.info(f"Loaded {len(entity_types)} typed entities")

    # Type distribution
    type_counts = Counter(entity_types.values())
    for t, c in type_counts.most_common():
        logger.info(f"  {t}: {c}")

    # Evaluate each config
    results = []
    config_dirs = sorted(hpo_dir.glob("config_*"))

    if not config_dirs:
        logger.error("No config directories found. Run hpo_rdf2vec.py first.")
        return

    for config_dir in config_dirs:
        emb_path = config_dir / "entity_embeddings.csv"
        summary_path = config_dir / "training_summary.json"

        if not emb_path.exists():
            logger.warning(f"Skipping {config_dir.name}: no embeddings found")
            continue

        logger.info(f"Evaluating {config_dir.name}...")

        # Load embeddings
        df = pd.read_csv(emb_path)
        entity_ids = df["entity_id"].tolist()
        dim_cols = [c for c in df.columns if c.startswith("dim_")]
        embeddings = df[dim_cols].values.astype(np.float32)

        # Load summary
        summary = {}
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)

        # Compute k-NN accuracy
        accuracy = knn_type_accuracy(embeddings, entity_ids, entity_types, k=10)
        logger.info(f"  k-NN type accuracy: {accuracy:.4f}")

        params = summary.get("params", {})
        results.append({
            "config_id": summary.get("config_id", config_dir.name),
            "depth": params.get("depth"),
            "walks": params.get("walks"),
            "vec_size": params.get("vec"),
            "window": params.get("window"),
            "epochs": params.get("epochs"),
            "neg": params.get("neg"),
            "knn_accuracy": round(accuracy, 4),
            "walk_w2v_time_s": summary.get("walk_w2v_time_s"),
            "entities_covered": summary.get("entities_covered"),
            "vocab_size": summary.get("vocab_size"),
        })

    # Save results
    results_df = pd.DataFrame(results).sort_values("knn_accuracy", ascending=False)
    results_df.to_csv(hpo_dir / "evaluation_results.csv", index=False)

    logger.info("\n" + "=" * 80)
    logger.info("RDF2Vec HPO Results (ranked by k-NN type accuracy):")
    logger.info("=" * 80)
    for _, row in results_df.iterrows():
        logger.info(
            f"  Config {row['config_id']:>2} | acc={row['knn_accuracy']:.4f} | "
            f"depth={row['depth']}, walks={row['walks']}, vec={row['vec_size']}, "
            f"window={row['window']}, epochs={row['epochs']}, neg={row['neg']}"
        )
    logger.info(f"\nResults saved to {hpo_dir / 'evaluation_results.csv'}")


if __name__ == "__main__":
    main()
