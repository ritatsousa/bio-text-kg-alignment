"""
Data Loading Utilities for KG Embeddings

This module provides functions to load TSV-formatted knowledge graph triples
and convert them to formats required by pyRDF2Vec (rdflib.Graph) and PyKEEN (TriplesFactory).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict

import rdflib
from rdflib import Graph, URIRef, Namespace, Literal
from rdflib.namespace import RDF, RDFS

from loguru import logger

# Try importing PyKEEN (may not be installed yet)
try:
    from pykeen.triples import TriplesFactory
    PYKEEN_AVAILABLE = True
except ImportError:
    PYKEEN_AVAILABLE = False
    logger.warning("PyKEEN not installed. PyKEEN-related functions will not work.")


# Define namespaces for RDF conversion
CTD = Namespace("http://ctd.org/")
CTD_FLAT = Namespace("http://ctd.org/flat/")
CTD_VOCAB = Namespace("http://ctd.org/vocabulary/")


def load_tsv_triples(path: str) -> List[Tuple[str, str, str]]:
    """
    Load triples from a TSV file (no headers, 3 columns).

    Args:
        path: Path to TSV file with format: subject<TAB>predicate<TAB>object

    Returns:
        List of (subject, predicate, object) tuples as strings

    Example:
        >>> triples = load_tsv_triples('kg_core.tsv')
        >>> triples[0]
        ('ctd:D003520', 'ctdflat:metabolic_processing', 'ctd:G1555')
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    triples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) != 3:
                logger.warning(f"Line {line_num}: Expected 3 columns, got {len(parts)}. Skipping.")
                continue

            subj, pred, obj = parts
            triples.append((subj, pred, obj))

    logger.info(f"Loaded {len(triples)} triples from {path}")
    return triples


def _prefix_to_uri(prefixed: str) -> URIRef:
    """
    Convert a prefixed string to a URIRef.

    Handles prefixes:
        - ctd: -> http://ctd.org/
        - ctdflat: -> http://ctd.org/flat/
        - ctd_vocabulary: -> http://ctd.org/vocabulary/
        - rdf:type -> standard RDF type
        - rdfs:label -> standard RDFS label
    """
    if prefixed == "rdf:type":
        return RDF.type
    elif prefixed == "rdfs:label":
        return RDFS.label
    elif prefixed.startswith("ctd:"):
        return CTD[prefixed[4:]]
    elif prefixed.startswith("ctdflat:"):
        return CTD_FLAT[prefixed[8:]]
    elif prefixed.startswith("ctd_vocabulary:"):
        return CTD_VOCAB[prefixed[15:]]
    else:
        # Fallback: treat as full URI or create generic URIRef
        return URIRef(prefixed)


def tsv_to_rdflib_graph(
    path: str,
    metadata_path: Optional[str] = None,
    include_types: bool = True,
    include_labels: bool = False
) -> Tuple[Graph, List[str]]:
    """
    Convert TSV triples to an rdflib Graph for pyRDF2Vec.

    Args:
        path: Path to main KG triples TSV (kg_core.tsv)
        metadata_path: Optional path to metadata TSV (metadata_core.tsv)
        include_types: If True, include rdf:type triples from metadata
        include_labels: If True, include rdfs:label triples from metadata
                       (Note: labels are literals and create dead-ends in walks)

    Returns:
        Tuple of (rdflib.Graph, list of entity URIs as strings)

    Example:
        >>> graph, entities = tsv_to_rdflib_graph(
        ...     'kg_core.tsv',
        ...     'metadata_core.tsv',
        ...     include_types=True
        ... )
    """
    g = Graph()

    # Bind namespaces for cleaner serialization
    g.bind("ctd", CTD)
    g.bind("ctdflat", CTD_FLAT)
    g.bind("ctd_vocab", CTD_VOCAB)

    entities_set: Set[str] = set()

    # Load main triples
    triples = load_tsv_triples(path)
    for subj, pred, obj in triples:
        s_uri = _prefix_to_uri(subj)
        p_uri = _prefix_to_uri(pred)
        o_uri = _prefix_to_uri(obj)

        g.add((s_uri, p_uri, o_uri))
        entities_set.add(str(s_uri))
        entities_set.add(str(o_uri))

    logger.info(f"Added {len(triples)} core triples to graph")

    # Load metadata if provided
    if metadata_path:
        metadata_triples = load_tsv_triples(metadata_path)
        metadata_added = 0

        for subj, pred, obj in metadata_triples:
            # Filter based on include_types and include_labels
            if pred == "rdf:type" and not include_types:
                continue
            if pred == "rdfs:label" and not include_labels:
                continue

            s_uri = _prefix_to_uri(subj)
            p_uri = _prefix_to_uri(pred)

            # Labels are literals (strings), types are URIs
            if pred == "rdfs:label":
                # Remove triple quotes if present
                label_text = obj.strip('"').strip("'")
                o_node = Literal(label_text)
            else:
                o_node = _prefix_to_uri(obj)
                # Add type classes as entities too
                if include_types:
                    entities_set.add(str(o_node))

            g.add((s_uri, p_uri, o_node))
            entities_set.add(str(s_uri))
            metadata_added += 1

        logger.info(f"Added {metadata_added} metadata triples to graph")

    entities = sorted(list(entities_set))
    logger.info(f"Graph has {len(g)} total triples and {len(entities)} unique entities")

    return g, entities


def get_entities_from_triples(triples: List[Tuple[str, str, str]]) -> List[str]:
    """
    Extract unique entities (subjects and objects) from triples.

    Args:
        triples: List of (subject, predicate, object) tuples

    Returns:
        Sorted list of unique entity strings
    """
    entities = set()
    for subj, _, obj in triples:
        entities.add(subj)
        entities.add(obj)
    return sorted(list(entities))


def get_relations_from_triples(triples: List[Tuple[str, str, str]]) -> List[str]:
    """
    Extract unique relations (predicates) from triples.

    Args:
        triples: List of (subject, predicate, object) tuples

    Returns:
        Sorted list of unique relation strings
    """
    relations = set()
    for _, pred, _ in triples:
        relations.add(pred)
    return sorted(list(relations))


def tsv_to_pykeen_factory(
    path: str,
    create_inverse_triples: bool = False
) -> "TriplesFactory":
    """
    Convert TSV triples to PyKEEN TriplesFactory.

    Args:
        path: Path to TSV file with triples
        create_inverse_triples: If True, create inverse triples for each relation

    Returns:
        PyKEEN TriplesFactory ready for training

    Example:
        >>> factory = tsv_to_pykeen_factory('kg_core.tsv')
        >>> factory.num_triples
        23087
    """
    if not PYKEEN_AVAILABLE:
        raise ImportError("PyKEEN is not installed. Run: pip install pykeen")

    triples = load_tsv_triples(path)

    # Convert to numpy array of shape (n_triples, 3)
    triples_array = np.array(triples, dtype=str)

    factory = TriplesFactory.from_labeled_triples(
        triples=triples_array,
        create_inverse_triples=create_inverse_triples
    )

    logger.info(
        f"Created TriplesFactory: {factory.num_triples} triples, "
        f"{factory.num_entities} entities, {factory.num_relations} relations"
    )

    return factory


def create_train_val_split(
    factory: "TriplesFactory",
    train_ratio: float = 0.95,
    random_state: int = 42,
    method: str = "cleanup"
) -> Tuple["TriplesFactory", "TriplesFactory"]:
    """
    Split TriplesFactory into train/validation sets (no test set).

    Uses cleanup-based splitting to ensure all entities and relations
    appear in training set. Degree-1 entities' triples are pinned to training.

    Args:
        factory: PyKEEN TriplesFactory to split
        train_ratio: Fraction of triples for training (default: 0.95)
        random_state: Random seed for reproducibility
        method: Split method ('cleanup' moves orphaned entities back to training)

    Returns:
        Tuple of (training, validation) TriplesFactory objects
    """
    if not PYKEEN_AVAILABLE:
        raise ImportError("PyKEEN is not installed. Run: pip install pykeen")

    validation_ratio = 1.0 - train_ratio

    training, validation = factory.split(
        ratios=[train_ratio, validation_ratio],
        random_state=random_state,
        method=method
    )

    # Log actual ratios (cleanup may shift them slightly)
    actual_train_ratio = training.num_triples / factory.num_triples
    actual_val_ratio = validation.num_triples / factory.num_triples
    logger.info(
        f"Split complete: train={training.num_triples} ({actual_train_ratio:.4f}), "
        f"val={validation.num_triples} ({actual_val_ratio:.4f})"
    )

    # Verify relation coverage
    train_rels = training.num_relations
    val_rels = validation.num_relations
    logger.info(f"Relation coverage: train={train_rels}/14, val={val_rels}/14")
    if train_rels < factory.num_relations:
        logger.warning(f"Training set missing {factory.num_relations - train_rels} relations!")

    # Verify entity coverage
    logger.info(
        f"Entity coverage: train={training.num_entities}/{factory.num_entities}, "
        f"val={validation.num_entities}/{factory.num_entities}"
    )

    return training, validation


def create_train_test_split(
    factory: "TriplesFactory",
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    method: str = "cleanup"
) -> Tuple["TriplesFactory", "TriplesFactory", "TriplesFactory"]:
    """
    Split TriplesFactory into train/validation/test sets.
    Kept for backwards compatibility. Prefer create_train_val_split() for new code.
    """
    if not PYKEEN_AVAILABLE:
        raise ImportError("PyKEEN is not installed. Run: pip install pykeen")

    test_ratio = 1.0 - train_ratio - validation_ratio

    if test_ratio < 0:
        raise ValueError(
            f"Invalid ratios: train={train_ratio}, val={validation_ratio}. "
            f"Sum must be <= 1.0"
        )

    training, rest = factory.split(
        ratios=[train_ratio, 1 - train_ratio],
        random_state=random_state,
        method=method
    )

    val_fraction = validation_ratio / (validation_ratio + test_ratio)
    validation, testing = rest.split(
        ratios=[val_fraction, 1 - val_fraction],
        random_state=random_state,
        method=method
    )

    logger.info(
        f"Split complete: train={training.num_triples}, "
        f"val={validation.num_triples}, test={testing.num_triples}"
    )

    return training, validation, testing


def group_triples_by_pair(
    triples: List[Tuple[str, str, str]]
) -> Dict[Tuple[str, str], List[Tuple[str, str, str]]]:
    """
    Group triples by (subject, object) pairs.

    Useful for identifying multi-edges and ensuring they're kept together
    during train/test splitting.

    Args:
        triples: List of (subject, predicate, object) tuples

    Returns:
        Dictionary mapping (subject, object) -> list of triples with that pair

    Example:
        >>> groups = group_triples_by_pair(triples)
        >>> multi_edges = {k: v for k, v in groups.items() if len(v) > 1}
        >>> len(multi_edges)  # Number of pairs with multiple relations
    """
    groups = defaultdict(list)
    for triple in triples:
        subj, pred, obj = triple
        groups[(subj, obj)].append(triple)
    return dict(groups)


def get_multi_edge_stats(triples: List[Tuple[str, str, str]]) -> Dict:
    """
    Compute statistics about multi-edges in the graph.

    Args:
        triples: List of (subject, predicate, object) tuples

    Returns:
        Dictionary with multi-edge statistics
    """
    groups = group_triples_by_pair(triples)

    multi_edge_pairs = {k: v for k, v in groups.items() if len(v) > 1}
    edge_counts = [len(v) for v in groups.values()]

    stats = {
        "total_triples": len(triples),
        "unique_pairs": len(groups),
        "multi_edge_pairs": len(multi_edge_pairs),
        "multi_edge_ratio": len(multi_edge_pairs) / len(groups) if groups else 0,
        "max_edges_per_pair": max(edge_counts) if edge_counts else 0,
        "avg_edges_per_pair": np.mean(edge_counts) if edge_counts else 0,
    }

    logger.info(
        f"Multi-edge stats: {stats['multi_edge_pairs']} pairs with multiple edges "
        f"({stats['multi_edge_ratio']:.1%} of {stats['unique_pairs']} unique pairs)"
    )

    return stats


if __name__ == "__main__":
    # Test the module with sample data
    import sys

    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        test_path = "kg_core_v2/kg_core.tsv"

    print(f"Testing data_loader with: {test_path}")

    # Test load_tsv_triples
    triples = load_tsv_triples(test_path)
    print(f"First triple: {triples[0]}")

    # Test entity/relation extraction
    entities = get_entities_from_triples(triples)
    relations = get_relations_from_triples(triples)
    print(f"Entities: {len(entities)}, Relations: {len(relations)}")

    # Test multi-edge stats
    stats = get_multi_edge_stats(triples)
    print(f"Multi-edge stats: {stats}")

    # Test rdflib conversion
    graph, ents = tsv_to_rdflib_graph(test_path)
    print(f"RDFLib graph: {len(graph)} triples")

    # Test PyKEEN conversion (if available)
    if PYKEEN_AVAILABLE:
        factory = tsv_to_pykeen_factory(test_path)
        print(f"PyKEEN factory: {factory.num_triples} triples")
