"""
KG Embeddings Package for CTD Data

This package provides tools for training knowledge graph embeddings
using pyRDF2Vec and PyKEEN on CTD-derived biomedical knowledge graphs.
"""

from .data_loader import (
    load_tsv_triples,
    tsv_to_rdflib_graph,
    create_train_test_split,
)
from .rdf2vec_trainer import (
    create_walker,
    train_rdf2vec,
    compute_relation_embeddings,
)
from .embeddings_io import (
    save_embeddings_csv,
    load_embeddings_csv,
    save_triple_embeddings,
)

__all__ = [
    "load_tsv_triples",
    "tsv_to_rdflib_graph",
    "create_train_test_split",
    "create_walker",
    "train_rdf2vec",
    "compute_relation_embeddings",
    "save_embeddings_csv",
    "load_embeddings_csv",
    "save_triple_embeddings",
]

# PyKEEN imports are optional (not available in rdf2vec venv)
try:
    from .data_loader import tsv_to_pykeen_factory
    from .pykeen_trainer import (
        train_no_eval,
        save_model,
        load_model,
    )
    __all__ += [
        "tsv_to_pykeen_factory",
        "train_no_eval",
        "save_model",
        "load_model",
    ]
except (ImportError, NameError):
    pass

__version__ = "0.1.0"
