"""
RDF2Vec Training Script (CPU, phased execution with resume)

Trains 200-dimensional embeddings using RDF2Vec on kg_full.tsv
Based on pyrdf2vec documentation: https://pyrdf2vec.readthedocs.io/en/latest/

Phased execution for long-running jobs with resume support:
    Phase 1 (train):   Generate walks + train Word2Vec, save model
    Phase 2 (extract): Extract entity embeddings, compute relation embeddings, save CSVs

On resubmit, completed phases are skipped automatically.

Usage:
    python train_rdf2vec.py

Output:
    outputs/rdf2vec/
        - word2vec_model.bin        (Phase 1: trained Word2Vec)
        - entity_embeddings.csv     (Phase 2: entity embeddings)
        - relation_embeddings.csv   (Phase 2: relation embeddings)
"""

import sys
import time
import yaml
import numpy as np
import pandas as pd
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
    "logs/rdf2vec_training.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",
    rotation="100 MB",
)

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec.graphs import KG, Vertex
import gensim


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_tsv_triples(path: str):
    """Load triples from TSV file."""
    triples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 3:
                triples.append(tuple(parts))
    logger.info(f"Loaded {len(triples)} triples from {path}")
    return triples


def build_kg_from_triples(triples):
    """Build pyrdf2vec KG from triples."""
    kg = KG()
    entities = set()

    logger.info(f"Building KG from {len(triples)} triples...")

    for i, (subj_name, pred_name, obj_name) in enumerate(triples):
        subj = Vertex(subj_name)
        obj = Vertex(obj_name)
        pred = Vertex(pred_name, predicate=True, vprev=subj, vnext=obj)

        kg.add_walk(subj, pred, obj)

        entities.add(subj_name)
        entities.add(obj_name)

        if (i + 1) % 100000 == 0:
            logger.info(f"  Processed {i + 1:,} triples...")

    entities = sorted(list(entities))
    logger.info(f"Built KG with {len(triples)} triples and {len(entities)} entities")

    return kg, entities


def save_embeddings_csv(embeddings, entities, path, id_column="entity_id"):
    """Save embeddings to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_entities, n_dims = embeddings.shape

    data = {id_column: entities}
    for i in range(n_dims):
        data[f"dim_{i}"] = embeddings[:, i]

    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    logger.info(f"Saved {n_entities} embeddings ({n_dims}-dim) to {path}")


def main():
    config = load_config()

    logger.info("=" * 60)
    logger.info("RDF2Vec Training Pipeline (phased, with resume)")
    logger.info("=" * 60)

    kg_path = config['data']['kg_path']
    rdf2vec_config = config['rdf2vec']
    output_dir = Path(config['output']['rdf2vec_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Data: {kg_path}")
    logger.info(f"Config: dim={rdf2vec_config['vector_size']}, depth={rdf2vec_config['max_depth']}, walks={rdf2vec_config['max_walks']}")
    logger.info(f"Output: {output_dir}")

    # Paths for checkpoints
    w2v_model_path = output_dir / "word2vec_model.bin"
    entity_emb_path = output_dir / "entity_embeddings.csv"
    relation_emb_path = output_dir / "relation_embeddings.csv"

    # Load triples and build KG
    triples = load_tsv_triples(kg_path)
    kg, entities = build_kg_from_triples(triples)

    # Track embeddings across phases
    embeddings = None

    # =====================================================================
    # Phase 1: Generate walks + train Word2Vec
    # =====================================================================
    if w2v_model_path.exists():
        logger.info(f"Phase 1 SKIP: Word2Vec model already exists at {w2v_model_path}")
        w2v_model = gensim.models.Word2Vec.load(str(w2v_model_path))
        logger.info(f"Loaded Word2Vec model: {w2v_model.wv.vector_size}-dim, {len(w2v_model.wv)} vocabulary")
    else:
        logger.info("Phase 1 START: Generating walks + training Word2Vec...")
        start_time = time.time()

        walker = RandomWalker(
            rdf2vec_config['max_depth'],
            rdf2vec_config['max_walks'],
            with_reverse=rdf2vec_config['with_reverse'],
            n_jobs=rdf2vec_config['workers'],
            md5_bytes=None,  # Keep full entity names in walks (no hashing)
        )
        logger.info(f"RandomWalker: depth={rdf2vec_config['max_depth']}, walks={rdf2vec_config['max_walks']}")
        logger.info(f"Total walks to generate: {len(entities) * rdf2vec_config['max_walks']:,}")

        embedder = Word2Vec(
            vector_size=rdf2vec_config['vector_size'],
            window=rdf2vec_config['window'],
            negative=rdf2vec_config['negative'],
            epochs=rdf2vec_config['epochs'],
            min_count=rdf2vec_config['min_count'],
            workers=rdf2vec_config['workers'],
            sg=1  # Skip-gram
        )
        logger.info(f"Word2Vec: dim={rdf2vec_config['vector_size']}, epochs={rdf2vec_config['epochs']}")

        transformer = RDF2VecTransformer(
            embedder,
            walkers=[walker],
            verbose=2
        )

        # fit_transform generates walks AND trains Word2Vec
        embeddings_list, literals = transformer.fit_transform(kg, entities)
        embeddings = np.array(embeddings_list)

        # Save the underlying Word2Vec model for resume
        transformer.embedder._model.save(str(w2v_model_path))

        elapsed = time.time() - start_time
        logger.info(f"Phase 1 DONE: Training complete in {elapsed/3600:.2f}h ({elapsed:.0f}s)")
        logger.info(f"Saved Word2Vec model to {w2v_model_path}")

    # =====================================================================
    # Phase 2: Extract and save embeddings
    # =====================================================================
    if entity_emb_path.exists() and relation_emb_path.exists():
        logger.info(f"Phase 2 SKIP: Embeddings already exist")
        logger.info(f"  - {entity_emb_path}")
        logger.info(f"  - {relation_emb_path}")
    else:
        logger.info("Phase 2 START: Extracting embeddings...")
        start_time = time.time()

        # If Phase 1 ran in this session, embeddings are already in memory
        # Otherwise, extract from saved Word2Vec model
        if embeddings is None:
            if not w2v_model_path.exists():
                logger.error("Word2Vec model not found! Run Phase 1 first.")
                sys.exit(1)
            w2v_model = gensim.models.Word2Vec.load(str(w2v_model_path))
            embeddings = np.array([
                w2v_model.wv[entity] if entity in w2v_model.wv else np.zeros(rdf2vec_config['vector_size'])
                for entity in entities
            ])

        logger.info(f"Entity embeddings shape: {embeddings.shape}")

        # Save entity embeddings
        save_embeddings_csv(embeddings, entities, entity_emb_path)

        # Compute and save relation embeddings
        logger.info("Computing relation embeddings from entity embeddings...")
        from src.rdf2vec_trainer import compute_relation_embeddings

        entity_to_idx = {e: i for i, e in enumerate(entities)}
        relation_emb, relation_to_idx = compute_relation_embeddings(
            entity_embeddings=embeddings,
            triples=triples,
            entity_to_idx=entity_to_idx,
            method="average_difference"
        )

        logger.info(f"Relation embeddings shape: {relation_emb.shape}")

        # Convert relation_to_idx to list format for save function
        relations = [None] * len(relation_to_idx)
        for rel, idx in relation_to_idx.items():
            relations[idx] = rel

        save_embeddings_csv(relation_emb, relations, relation_emb_path, id_column="relation_id")

        elapsed = time.time() - start_time
        logger.info(f"Phase 2 DONE: Extraction complete in {elapsed:.1f}s")

    logger.info("=" * 60)
    logger.info("RDF2Vec training complete!")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
