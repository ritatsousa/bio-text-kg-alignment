"""
RDF2Vec Training Module

This module provides functions to train RDF2Vec embeddings using pyRDF2Vec.
RDF2Vec generates entity embeddings by performing random walks on the graph
and then training Word2Vec on the resulting sequences.
"""

import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict

from rdflib import Graph, URIRef

from loguru import logger


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

# Try importing pyRDF2Vec components
try:
    from pyrdf2vec import RDF2VecTransformer
    from pyrdf2vec.embedders import Word2Vec
    from pyrdf2vec.walkers import RandomWalker, WLWalker
    from pyrdf2vec.graphs import KG
    PYRDF2VEC_AVAILABLE = True
except ImportError:
    PYRDF2VEC_AVAILABLE = False
    logger.warning("pyRDF2Vec not installed. RDF2Vec functions will not work.")


# Default hyperparameters based on biomedical benchmark best practices
DEFAULT_RDF2VEC_CONFIG = {
    # Walker parameters
    "max_depth": 4,           # Walk depth (captures 2-hop neighbors)
    "max_walks": 200,         # Walks per entity (reduced for large graphs)
    "with_reverse": True,     # Include reverse walks for undirected traversal
    "random_state": 42,       # Reproducibility

    # Word2Vec parameters
    "vector_size": 200,       # Embedding dimension
    "window": 5,              # Context window size
    "negative": 10,           # Negative samples
    "epochs": 50,             # Training epochs
    "min_count": 1,           # Include all entities (important for KG)
    "workers": 8,             # Parallel workers

    # Walk strategy
    "walker_type": "random",  # "random" or "wl" (Weisfeiler-Lehman)
}


def create_walker(
    max_depth: int = 4,
    max_walks: int = 500,
    with_reverse: bool = True,
    random_state: int = 42,
    walker_type: str = "random"
) -> Any:
    """
    Create a random walker for RDF2Vec.

    Args:
        max_depth: Maximum depth of random walks (default: 4)
        max_walks: Number of walks per entity (default: 500)
        with_reverse: Include reverse edges in walks (default: True)
        random_state: Random seed for reproducibility
        walker_type: "random" for RandomWalker or "wl" for WeisfeilerLehmanWalker

    Returns:
        Walker instance configured for pyRDF2Vec

    Example:
        >>> walker = create_walker(max_depth=4, max_walks=500)
    """
    if not PYRDF2VEC_AVAILABLE:
        raise ImportError("pyRDF2Vec not installed. Run: pip install pyrdf2vec")

    if walker_type.lower() == "wl":
        walker = WLWalker(
            max_depth=max_depth,
            max_walks=max_walks,
            with_reverse=with_reverse,
            random_state=random_state,
        )
        logger.info(f"Created WL walker: depth={max_depth}, walks={max_walks}")
    else:
        walker = RandomWalker(
            max_depth=max_depth,
            max_walks=max_walks,
            with_reverse=with_reverse,
            random_state=random_state,
        )
        logger.info(f"Created Random walker: depth={max_depth}, walks={max_walks}")

    return walker


def create_embedder(
    vector_size: int = 512,
    window: int = 5,
    negative: int = 10,
    epochs: int = 100,
    min_count: int = 1,
    workers: int = 4,
) -> Any:
    """
    Create a Word2Vec embedder for RDF2Vec.

    Args:
        vector_size: Embedding dimension (default: 512)
        window: Context window size (default: 5)
        negative: Number of negative samples (default: 10)
        epochs: Training epochs (default: 100)
        min_count: Minimum word frequency (default: 1, include all)
        workers: Number of parallel workers (default: 4)

    Returns:
        Word2Vec embedder instance
    """
    if not PYRDF2VEC_AVAILABLE:
        raise ImportError("pyRDF2Vec not installed. Run: pip install pyrdf2vec")

    embedder = Word2Vec(
        vector_size=vector_size,
        window=window,
        negative=negative,
        epochs=epochs,
        min_count=min_count,
        workers=workers,
        sg=1,  # Skip-gram model (recommended for RDF2Vec)
    )

    logger.info(
        f"Created Word2Vec embedder: dim={vector_size}, window={window}, "
        f"negative={negative}, epochs={epochs}"
    )

    return embedder


def rdflib_to_pyrdf2vec_kg(graph: Graph) -> Any:
    """
    Convert an rdflib Graph to pyRDF2Vec KG format.

    pyRDF2Vec can work directly with SPARQL endpoints or local RDF files.
    For in-memory graphs, we serialize and reload.

    Args:
        graph: rdflib.Graph instance

    Returns:
        pyRDF2Vec KG instance
    """
    if not PYRDF2VEC_AVAILABLE:
        raise ImportError("pyRDF2Vec not installed. Run: pip install pyrdf2vec")

    # Serialize graph to temporary file (pyRDF2Vec needs file path or SPARQL endpoint)
    import tempfile
    import os

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "kg_temp.nt")

    # Serialize as N-Triples (simple, reliable format)
    graph.serialize(destination=temp_path, format="nt")

    # Create pyRDF2Vec KG from file
    kg = KG(location=temp_path, fmt="nt")

    logger.info(f"Converted rdflib Graph ({len(graph)} triples) to pyRDF2Vec KG")

    return kg, temp_path


def train_rdf2vec(
    graph: Graph,
    entities: List[str],
    config: Optional[Dict] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Train RDF2Vec embeddings on a knowledge graph.

    Args:
        graph: rdflib.Graph containing the knowledge graph
        entities: List of entity URIs (as strings) to embed
        config: Configuration dictionary (uses DEFAULT_RDF2VEC_CONFIG if None)
        verbose: Print progress information

    Returns:
        Tuple of:
            - embeddings: numpy array of shape (n_entities, embedding_dim)
            - entity_to_idx: Dictionary mapping entity URI string to index

    Example:
        >>> graph, entities = tsv_to_rdflib_graph('kg_core.tsv', 'metadata_core.tsv')
        >>> embeddings, entity_map = train_rdf2vec(graph, entities)
        >>> embeddings.shape
        (7682, 512)
    """
    if not PYRDF2VEC_AVAILABLE:
        raise ImportError("pyRDF2Vec not installed. Run: pip install pyrdf2vec")

    # Merge with default config
    cfg = DEFAULT_RDF2VEC_CONFIG.copy()
    if config:
        cfg.update(config)

    # Estimate training time
    num_entities = len(entities)
    walks_total = num_entities * cfg["max_walks"]
    # Rough estimate: ~0.001s per walk on CPU
    estimated_walk_time = walks_total * 0.001
    # Word2Vec training: ~0.1s per 1000 walks per epoch
    estimated_w2v_time = (walks_total / 1000) * cfg["epochs"] * 0.1

    estimated_total = estimated_walk_time + estimated_w2v_time
    if estimated_total < 60:
        time_str = f"~{int(estimated_total)} seconds"
    elif estimated_total < 3600:
        time_str = f"~{int(estimated_total / 60)} minutes"
    else:
        time_str = f"~{estimated_total / 3600:.1f} hours"

    logger.info(f"Starting RDF2Vec training")
    logger.info(f"Entities: {num_entities:,}, Walks per entity: {cfg['max_walks']}, Total walks: {walks_total:,}")
    logger.info(f"Config: dim={cfg['vector_size']}, depth={cfg['max_depth']}, epochs={cfg['epochs']}")
    logger.info(f"Estimated training time: {time_str}")

    start_time = time.time()

    # Create walker
    walker = create_walker(
        max_depth=cfg["max_depth"],
        max_walks=cfg["max_walks"],
        with_reverse=cfg["with_reverse"],
        random_state=cfg["random_state"],
        walker_type=cfg["walker_type"],
    )

    # Create embedder
    embedder = create_embedder(
        vector_size=cfg["vector_size"],
        window=cfg["window"],
        negative=cfg["negative"],
        epochs=cfg["epochs"],
        min_count=cfg["min_count"],
        workers=cfg["workers"],
    )

    # Convert graph to pyRDF2Vec format
    kg, temp_path = rdflib_to_pyrdf2vec_kg(graph)

    # Create transformer
    transformer = RDF2VecTransformer(
        walkers=[walker],
        embedder=embedder,
        verbose=2 if verbose else 0,
    )

    # Convert entity strings to URIRefs (pyRDF2Vec expects these)
    entity_uris = [URIRef(e) if not isinstance(e, URIRef) else e for e in entities]

    # Train embeddings
    logger.info(f"Training RDF2Vec on {len(entity_uris)} entities...")
    embeddings, _ = transformer.fit_transform(kg, entity_uris)

    # Convert to numpy array if needed
    embeddings = np.array(embeddings)

    # Create entity to index mapping
    entity_to_idx = {str(uri): idx for idx, uri in enumerate(entity_uris)}

    # Cleanup temp file
    import os
    import shutil
    try:
        os.remove(temp_path)
        shutil.rmtree(os.path.dirname(temp_path))
    except Exception as e:
        logger.warning(f"Could not clean up temp file: {e}")

    elapsed_time = time.time() - start_time
    logger.info(
        f"RDF2Vec training complete in {format_elapsed_time(elapsed_time)}: "
        f"{embeddings.shape[0]} entities, {embeddings.shape[1]} dimensions"
    )

    return embeddings, entity_to_idx


def compute_relation_embeddings(
    entity_embeddings: np.ndarray,
    triples: List[Tuple[str, str, str]],
    entity_to_idx: Dict[str, int],
    method: str = "average_difference"
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Compute relation embeddings post-hoc from entity embeddings.

    RDF2Vec only produces entity embeddings. This function computes
    relation embeddings by aggregating over all triples using that relation.

    Methods:
        - "average_difference": r = avg(o - s) for all (s, r, o) triples
        - "average_concat": r = avg([s; o]) for all (s, r, o) triples
        - "hadamard": r = avg(s * o) for all (s, r, o) triples

    Args:
        entity_embeddings: numpy array of entity embeddings
        triples: List of (subject, predicate, object) tuples
        entity_to_idx: Mapping from entity string to embedding index
        method: Aggregation method (default: "average_difference")

    Returns:
        Tuple of:
            - relation_embeddings: numpy array of shape (n_relations, embedding_dim)
            - relation_to_idx: Dictionary mapping relation string to index

    Example:
        >>> rel_emb, rel_map = compute_relation_embeddings(
        ...     entity_emb, triples, entity_map, method="average_difference"
        ... )
    """
    # Group triples by relation
    relation_triples = defaultdict(list)
    for subj, pred, obj in triples:
        # Only include triples where both entities have embeddings
        if subj in entity_to_idx and obj in entity_to_idx:
            relation_triples[pred].append((subj, obj))

    relations = sorted(relation_triples.keys())
    relation_to_idx = {rel: idx for idx, rel in enumerate(relations)}

    embedding_dim = entity_embeddings.shape[1]
    relation_embeddings = np.zeros((len(relations), embedding_dim))

    for rel, pairs in relation_triples.items():
        rel_idx = relation_to_idx[rel]
        vectors = []

        for subj, obj in pairs:
            s_idx = entity_to_idx[subj]
            o_idx = entity_to_idx[obj]
            s_vec = entity_embeddings[s_idx]
            o_vec = entity_embeddings[o_idx]

            if method == "average_difference":
                # TransE-style: relation as translation from subject to object
                vec = o_vec - s_vec
            elif method == "average_concat":
                # Concatenation approach (doubles dimension)
                vec = np.concatenate([s_vec, o_vec])
            elif method == "hadamard":
                # Element-wise product
                vec = s_vec * o_vec
            else:
                raise ValueError(f"Unknown method: {method}")

            vectors.append(vec)

        # Average all vectors for this relation
        relation_embeddings[rel_idx] = np.mean(vectors, axis=0)

    logger.info(
        f"Computed relation embeddings: {len(relations)} relations, "
        f"method={method}"
    )

    return relation_embeddings, relation_to_idx


def get_entity_embedding(
    entity: str,
    embeddings: np.ndarray,
    entity_to_idx: Dict[str, int]
) -> Optional[np.ndarray]:
    """
    Get embedding vector for a single entity.

    Args:
        entity: Entity URI string
        embeddings: Full embedding matrix
        entity_to_idx: Entity to index mapping

    Returns:
        Embedding vector or None if entity not found
    """
    if entity not in entity_to_idx:
        logger.warning(f"Entity not found: {entity}")
        return None

    idx = entity_to_idx[entity]
    return embeddings[idx]


def get_nearest_neighbors(
    query_entity: str,
    embeddings: np.ndarray,
    entity_to_idx: Dict[str, int],
    k: int = 10,
    metric: str = "cosine"
) -> List[Tuple[str, float]]:
    """
    Find k nearest neighbors for a query entity.

    Args:
        query_entity: Entity URI string
        embeddings: Full embedding matrix
        entity_to_idx: Entity to index mapping
        k: Number of neighbors to return
        metric: Distance metric ("cosine" or "euclidean")

    Returns:
        List of (entity_uri, distance) tuples, sorted by distance
    """
    query_vec = get_entity_embedding(query_entity, embeddings, entity_to_idx)
    if query_vec is None:
        return []

    idx_to_entity = {v: k for k, v in entity_to_idx.items()}

    if metric == "cosine":
        # Normalize vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings / norms

        query_norm = query_vec / (np.linalg.norm(query_vec) or 1)

        # Cosine similarity (higher is better, so we negate for sorting)
        similarities = normalized @ query_norm
        distances = 1 - similarities
    else:
        # Euclidean distance
        distances = np.linalg.norm(embeddings - query_vec, axis=1)

    # Get k+1 nearest (includes self)
    nearest_idx = np.argsort(distances)[:k + 1]

    # Exclude self and return
    results = []
    for idx in nearest_idx:
        entity = idx_to_entity[idx]
        if entity != query_entity:
            results.append((entity, float(distances[idx])))
        if len(results) >= k:
            break

    return results


if __name__ == "__main__":
    # Test the module
    print("Testing rdf2vec_trainer module...")

    if PYRDF2VEC_AVAILABLE:
        print("pyRDF2Vec is available")
        print(f"Default config: {DEFAULT_RDF2VEC_CONFIG}")

        # Test walker creation
        walker = create_walker(max_depth=2, max_walks=10)
        print(f"Walker created: {type(walker)}")

        # Test embedder creation
        embedder = create_embedder(vector_size=64, epochs=10)
        print(f"Embedder created: {type(embedder)}")
    else:
        print("pyRDF2Vec not installed - skipping tests")
