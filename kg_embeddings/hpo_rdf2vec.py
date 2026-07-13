"""
RDF2Vec HPO — single-config runner (called by SLURM array job)

Trains RDF2Vec with one configuration from the curated grid.
Saves entity + relation embeddings for evaluation.

Usage:
    python hpo_rdf2vec.py --config_id 3
"""

import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec.graphs import KG, Vertex
import gensim

# 15 curated configurations — vary one dim at a time from baseline + key interactions
CONFIGS = {
    1:  dict(depth=4, walks=200, vec=200, window=5, epochs=50, neg=10),   # baseline
    2:  dict(depth=6, walks=200, vec=200, window=5, epochs=50, neg=10),   # deeper
    3:  dict(depth=8, walks=200, vec=200, window=5, epochs=50, neg=10),   # paper best depth
    4:  dict(depth=4, walks=100, vec=200, window=5, epochs=50, neg=10),   # fewer walks
    5:  dict(depth=4, walks=500, vec=200, window=5, epochs=50, neg=10),   # more walks
    6:  dict(depth=4, walks=200, vec=100, window=5, epochs=50, neg=10),   # smaller dim
    7:  dict(depth=4, walks=200, vec=300, window=5, epochs=50, neg=10),   # larger dim
    8:  dict(depth=4, walks=200, vec=200, window=3, epochs=50, neg=10),   # narrow context
    9:  dict(depth=4, walks=200, vec=200, window=7, epochs=50, neg=10),   # wide context
    10: dict(depth=4, walks=200, vec=200, window=5, epochs=10, neg=10),   # fewer W2V epochs
    11: dict(depth=4, walks=200, vec=200, window=5, epochs=25, neg=10),   # mid W2V epochs
    12: dict(depth=4, walks=200, vec=200, window=5, epochs=50, neg=25),   # paper neg value
    13: dict(depth=8, walks=500, vec=200, window=5, epochs=25, neg=25),   # paper combo
    14: dict(depth=6, walks=200, vec=300, window=5, epochs=25, neg=15),   # deeper + larger
    15: dict(depth=8, walks=200, vec=100, window=5, epochs=10, neg=25),   # deep + small
}


def load_triples(path):
    triples = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def build_kg(triples):
    kg = KG()
    entities = set()
    for i, (s, p, o) in enumerate(triples):
        subj = Vertex(s)
        obj = Vertex(o)
        pred = Vertex(p, predicate=True, vprev=subj, vnext=obj)
        kg.add_walk(subj, pred, obj)
        entities.add(s)
        entities.add(o)
        if (i + 1) % 200000 == 0:
            logger.info(f"  KG build: {i+1:,}/{len(triples):,} triples")
    return kg, sorted(entities)


def compute_relation_embeddings(entity_embeddings, triples, entity_to_idx):
    relation_triples = defaultdict(list)
    for s, p, o in triples:
        if s in entity_to_idx and o in entity_to_idx:
            relation_triples[p].append((s, o))

    relations = sorted(relation_triples.keys())
    dim = entity_embeddings.shape[1]
    rel_emb = np.zeros((len(relations), dim))

    for i, rel in enumerate(relations):
        vecs = []
        for s, o in relation_triples[rel]:
            vecs.append(entity_embeddings[entity_to_idx[o]] - entity_embeddings[entity_to_idx[s]])
        rel_emb[i] = np.mean(vecs, axis=0)

    return rel_emb, {r: i for i, r in enumerate(relations)}


def save_embeddings_csv(embeddings, ids, path, id_col="entity_id"):
    data = {id_col: ids}
    for i in range(embeddings.shape[1]):
        data[f"dim_{i}"] = embeddings[:, i]
    pd.DataFrame(data).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", type=int, required=True, choices=range(1, 16))
    args = parser.parse_args()

    cfg = CONFIGS[args.config_id]
    logger.add(f"logs/hpo_rdf2vec_config{args.config_id}.log", format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}", level="DEBUG", rotation="100 MB")

    logger.info("=" * 60)
    logger.info(f"RDF2Vec HPO — Config {args.config_id}")
    logger.info(f"  depth={cfg['depth']}, walks={cfg['walks']}, vec={cfg['vec']}, "
                f"window={cfg['window']}, epochs={cfg['epochs']}, neg={cfg['neg']}")
    logger.info("=" * 60)

    # Load training triples (95% split)
    triples_path = "outputs/split/train_triples.tsv"
    logger.info(f"Loading triples from {triples_path}...")
    triples = load_triples(triples_path)
    logger.info(f"Loaded {len(triples):,} triples")

    # Build KG
    logger.info("Building KG...")
    t0 = time.time()
    kg, entities = build_kg(triples)
    kg_time = time.time() - t0
    logger.info(f"KG built: {len(entities):,} entities in {kg_time:.1f}s")

    # Create walker + embedder
    walker = RandomWalker(
        cfg["depth"], cfg["walks"],
        with_reverse=False,
        n_jobs=16,
        md5_bytes=None,
    )

    embedder = Word2Vec(
        vector_size=cfg["vec"],
        window=cfg["window"],
        negative=cfg["neg"],
        epochs=cfg["epochs"],
        min_count=1,
        workers=16,
        sg=1,
    )

    transformer = RDF2VecTransformer(embedder, walkers=[walker], verbose=2)

    # Train
    logger.info("Phase 1: Walk generation + Word2Vec training...")
    t_walk = time.time()
    embeddings_list, _ = transformer.fit_transform(kg, entities)
    walk_total = time.time() - t_walk
    embeddings = np.array(embeddings_list)
    logger.info(f"Phase 1 done: {embeddings.shape} in {walk_total:.1f}s ({walk_total/3600:.2f}h)")

    # Get vocab size
    vocab_size = len(transformer.embedder._model.wv) if hasattr(transformer.embedder, '_model') else 0

    # Output directory
    out_dir = Path(f"outputs/hpo/rdf2vec/config_{args.config_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save entity embeddings
    save_embeddings_csv(embeddings, entities, out_dir / "entity_embeddings.csv")
    logger.info(f"Saved entity embeddings: {embeddings.shape}")

    # Compute + save relation embeddings
    logger.info("Computing relation embeddings (average_difference)...")
    entity_to_idx = {e: i for i, e in enumerate(entities)}
    rel_emb, rel_to_idx = compute_relation_embeddings(embeddings, triples, entity_to_idx)
    relations = [None] * len(rel_to_idx)
    for r, i in rel_to_idx.items():
        relations[i] = r
    save_embeddings_csv(rel_emb, relations, out_dir / "relation_embeddings.csv", id_col="relation_id")
    logger.info(f"Saved relation embeddings: {rel_emb.shape}")

    # Save Word2Vec model
    if hasattr(transformer.embedder, '_model'):
        transformer.embedder._model.save(str(out_dir / "word2vec_model.bin"))

    # Save training summary
    summary = {
        "config_id": args.config_id,
        "params": cfg,
        "dataset": triples_path,
        "num_triples": len(triples),
        "num_entities": len(entities),
        "entity_embedding_shape": list(embeddings.shape),
        "relation_embedding_shape": list(rel_emb.shape),
        "vocab_size": vocab_size,
        "entities_covered": int(np.sum(np.any(embeddings != 0, axis=1))),
        "total_time_s": walk_total + kg_time,
        "kg_build_time_s": kg_time,
        "walk_w2v_time_s": walk_total,
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Config {args.config_id} complete. Output: {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
