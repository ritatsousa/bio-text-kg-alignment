"""
Embeddings I/O Module

This module provides functions to save and load embeddings in various formats,
and to generate triple embeddings for downstream text alignment tasks.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union

from loguru import logger


def save_embeddings_csv(
    embeddings: np.ndarray,
    id_mapping: Dict[str, int],
    path: str,
    id_column: str = "entity_id"
) -> None:
    """
    Save embeddings to CSV format with entity IDs.

    Format:
        entity_id,dim_0,dim_1,...,dim_N
        ctd:D003520,0.123,-0.456,...,0.789

    Args:
        embeddings: numpy array of shape (n_entities, embedding_dim)
        id_mapping: Dictionary mapping entity/relation ID to index
        path: Output CSV file path
        id_column: Name for the ID column (default: "entity_id")

    Example:
        >>> save_embeddings_csv(entity_emb, entity_map, 'outputs/entities.csv')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Invert mapping: index -> id
    idx_to_id = {v: k for k, v in id_mapping.items()}

    # Create DataFrame
    n_entities, n_dims = embeddings.shape
    dim_columns = [f"dim_{i}" for i in range(n_dims)]

    data = {id_column: [idx_to_id[i] for i in range(n_entities)]}
    for i, col in enumerate(dim_columns):
        data[col] = embeddings[:, i]

    df = pd.DataFrame(data)
    df.to_csv(path, index=False)

    logger.info(f"Saved {n_entities} embeddings ({n_dims}-dim) to {path}")


def load_embeddings_csv(
    path: str,
    id_column: str = "entity_id"
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Load embeddings from CSV format.

    Args:
        path: Path to CSV file
        id_column: Name of the ID column

    Returns:
        Tuple of:
            - embeddings: numpy array of shape (n_entities, embedding_dim)
            - id_mapping: Dictionary mapping entity/relation ID to index

    Example:
        >>> embeddings, mapping = load_embeddings_csv('outputs/entities.csv')
    """
    df = pd.read_csv(path)

    # Extract IDs
    ids = df[id_column].tolist()
    id_mapping = {entity_id: idx for idx, entity_id in enumerate(ids)}

    # Extract embedding columns
    dim_columns = [col for col in df.columns if col.startswith("dim_")]
    embeddings = df[dim_columns].values.astype(np.float32)

    logger.info(f"Loaded {len(ids)} embeddings ({embeddings.shape[1]}-dim) from {path}")

    return embeddings, id_mapping


def save_embeddings_numpy(
    embeddings: np.ndarray,
    id_mapping: Dict[str, int],
    embeddings_path: str,
    mapping_path: str
) -> None:
    """
    Save embeddings in numpy format (faster for large embeddings).

    Args:
        embeddings: numpy array of embeddings
        id_mapping: ID to index mapping
        embeddings_path: Path for .npy file
        mapping_path: Path for .json mapping file
    """
    import json

    # Save embeddings
    np.save(embeddings_path, embeddings)

    # Save mapping
    with open(mapping_path, 'w') as f:
        json.dump(id_mapping, f)

    logger.info(f"Saved embeddings to {embeddings_path} and mapping to {mapping_path}")


def load_embeddings_numpy(
    embeddings_path: str,
    mapping_path: str
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Load embeddings from numpy format.

    Args:
        embeddings_path: Path to .npy file
        mapping_path: Path to .json mapping file

    Returns:
        Tuple of (embeddings, id_mapping)
    """
    import json

    embeddings = np.load(embeddings_path)

    with open(mapping_path, 'r') as f:
        id_mapping = json.load(f)

    logger.info(f"Loaded embeddings: shape={embeddings.shape}")

    return embeddings, id_mapping


def generate_triple_embedding(
    subject: str,
    predicate: str,
    obj: str,
    entity_emb: np.ndarray,
    relation_emb: np.ndarray,
    entity_map: Dict[str, int],
    relation_map: Dict[str, int],
    combination: str = "concat"
) -> Optional[np.ndarray]:
    """
    Generate a single embedding for a triple by combining S, P, O embeddings.

    Combination methods:
        - "concat": [s; p; o] - concatenation (3x embedding_dim)
        - "sum": s + p + o (embedding_dim)
        - "hadamard_sum": (s * p) + (p * o) (embedding_dim)
        - "transe": s + p (should approximate o) (embedding_dim)

    Args:
        subject: Subject entity ID
        predicate: Predicate/relation ID
        obj: Object entity ID
        entity_emb: Entity embedding matrix
        relation_emb: Relation embedding matrix
        entity_map: Entity ID to index mapping
        relation_map: Relation ID to index mapping
        combination: Method for combining embeddings

    Returns:
        Combined triple embedding or None if any component is missing
    """
    # Get indices
    s_idx = entity_map.get(subject)
    p_idx = relation_map.get(predicate)
    o_idx = entity_map.get(obj)

    if s_idx is None:
        logger.debug(f"Subject not found: {subject}")
        return None
    if p_idx is None:
        logger.debug(f"Predicate not found: {predicate}")
        return None
    if o_idx is None:
        logger.debug(f"Object not found: {obj}")
        return None

    # Get vectors
    s_vec = entity_emb[s_idx]
    p_vec = relation_emb[p_idx]
    o_vec = entity_emb[o_idx]

    # Combine based on method
    if combination == "concat":
        return np.concatenate([s_vec, p_vec, o_vec])
    elif combination == "sum":
        return s_vec + p_vec + o_vec
    elif combination == "hadamard_sum":
        return (s_vec * p_vec) + (p_vec * o_vec)
    elif combination == "transe":
        return s_vec + p_vec
    else:
        raise ValueError(f"Unknown combination method: {combination}")


def save_triple_embeddings(
    evidence_path: str,
    entity_emb: np.ndarray,
    relation_emb: np.ndarray,
    entity_map: Dict[str, int],
    relation_map: Dict[str, int],
    output_path: str,
    combination: str = "concat",
    subject_col: str = "subject_uri",
    predicate_col: str = "predicate_uri",
    object_col: str = "object_uri",
    triple_id_col: str = "triple_id"
) -> int:
    """
    Generate and save triple embeddings for all triples in evidence file.

    The output format is designed for downstream text alignment tasks,
    with triple_id linking back to the evidence_aligned_ided.tsv file.

    Args:
        evidence_path: Path to evidence_aligned_ided.tsv
        entity_emb: Entity embedding matrix
        relation_emb: Relation embedding matrix
        entity_map: Entity ID to index mapping
        relation_map: Relation ID to index mapping
        output_path: Output CSV path
        combination: Embedding combination method
        subject_col: Column name for subject in evidence file
        predicate_col: Column name for predicate in evidence file
        object_col: Column name for object in evidence file
        triple_id_col: Column name for triple ID in evidence file

    Returns:
        Number of triples successfully processed

    Output format:
        triple_id,subject_uri,predicate_uri,object_uri,embedding
        abc123,ctd:D003520,ctdflat:metabolic_processing,ctd:G1555,0.123;-0.456;...
    """
    # Load evidence file
    df = pd.read_csv(evidence_path, sep='\t')

    logger.info(f"Processing {len(df)} triples from {evidence_path}")

    results = []
    skipped = 0

    for _, row in df.iterrows():
        triple_id = row[triple_id_col]
        subject = row[subject_col]
        predicate = row[predicate_col]
        obj = row[object_col]

        embedding = generate_triple_embedding(
            subject, predicate, obj,
            entity_emb, relation_emb,
            entity_map, relation_map,
            combination
        )

        if embedding is not None:
            # Convert embedding to semicolon-separated string
            emb_str = ";".join(map(str, embedding))
            results.append({
                "triple_id": triple_id,
                "subject_uri": subject,
                "predicate_uri": predicate,
                "object_uri": obj,
                "embedding": emb_str
            })
        else:
            skipped += 1

    # Save results
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)

    logger.info(
        f"Saved {len(results)} triple embeddings to {output_path} "
        f"(skipped {skipped} due to missing components)"
    )

    return len(results)


def load_triple_embeddings(
    path: str
) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    """
    Load triple embeddings from CSV.

    Args:
        path: Path to triple embeddings CSV

    Returns:
        Tuple of:
            - triple_ids: List of triple IDs
            - embeddings: numpy array of shape (n_triples, embedding_dim)
            - metadata: DataFrame with triple metadata (subject, predicate, object)
    """
    df = pd.read_csv(path)

    triple_ids = df["triple_id"].tolist()

    # Parse embedding strings
    embeddings = np.array([
        list(map(float, emb.split(";")))
        for emb in df["embedding"]
    ])

    metadata = df[["triple_id", "subject_uri", "predicate_uri", "object_uri"]]

    logger.info(f"Loaded {len(triple_ids)} triple embeddings ({embeddings.shape[1]}-dim)")

    return triple_ids, embeddings, metadata


def export_for_alignment(
    entity_emb_path: str,
    relation_emb_path: str,
    evidence_path: str,
    output_dir: str,
    combination: str = "concat"
) -> str:
    """
    Complete export pipeline for text alignment task.

    Loads entity and relation embeddings, generates triple embeddings,
    and saves all outputs in the specified directory.

    Args:
        entity_emb_path: Path to entity embeddings CSV
        relation_emb_path: Path to relation embeddings CSV
        evidence_path: Path to evidence_aligned_ided.tsv
        output_dir: Output directory
        combination: Embedding combination method

    Returns:
        Path to generated triple embeddings file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    entity_emb, entity_map = load_embeddings_csv(entity_emb_path)
    relation_emb, relation_map = load_embeddings_csv(
        relation_emb_path,
        id_column="relation_id" if "relation" in relation_emb_path.lower() else "entity_id"
    )

    # Generate triple embeddings
    triple_output = output_dir / "triple_embeddings.csv"
    n_processed = save_triple_embeddings(
        evidence_path,
        entity_emb, relation_emb,
        entity_map, relation_map,
        str(triple_output),
        combination
    )

    logger.info(f"Export complete: {n_processed} triple embeddings saved to {output_dir}")

    return str(triple_output)


def compute_embedding_statistics(embeddings: np.ndarray) -> Dict:
    """
    Compute basic statistics about embeddings.

    Args:
        embeddings: numpy array of embeddings

    Returns:
        Dictionary of statistics
    """
    stats = {
        "n_embeddings": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
        "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
        "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
        "mean_value": float(np.mean(embeddings)),
        "std_value": float(np.std(embeddings)),
        "min_value": float(np.min(embeddings)),
        "max_value": float(np.max(embeddings)),
    }

    return stats


if __name__ == "__main__":
    # Test the module
    print("Testing embeddings_io module...")

    # Create dummy embeddings for testing
    n_entities = 100
    n_relations = 10
    dim = 64

    entity_emb = np.random.randn(n_entities, dim).astype(np.float32)
    relation_emb = np.random.randn(n_relations, dim).astype(np.float32)

    entity_map = {f"entity_{i}": i for i in range(n_entities)}
    relation_map = {f"relation_{i}": i for i in range(n_relations)}

    # Test saving/loading
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test CSV
        csv_path = os.path.join(tmpdir, "test_embeddings.csv")
        save_embeddings_csv(entity_emb, entity_map, csv_path)
        loaded_emb, loaded_map = load_embeddings_csv(csv_path)
        print(f"CSV: Original shape={entity_emb.shape}, Loaded shape={loaded_emb.shape}")
        print(f"CSV: Embeddings match: {np.allclose(entity_emb, loaded_emb)}")

        # Test triple embedding generation
        triple_emb = generate_triple_embedding(
            "entity_0", "relation_0", "entity_1",
            entity_emb, relation_emb,
            entity_map, relation_map,
            combination="concat"
        )
        print(f"Triple embedding shape (concat): {triple_emb.shape}")

        # Test statistics
        stats = compute_embedding_statistics(entity_emb)
        print(f"Statistics: {stats}")
