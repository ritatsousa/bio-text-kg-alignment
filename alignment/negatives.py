"""Hard negative sampling for contrastive training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from data import apply_spo_combination
from utils import get_logger

logger = get_logger(__name__)


STRATEGIES = [
    "same_s_same_o_vary_p",   # change predicate only
    "same_s_vary_o",          # change object only
    "vary_s_same_p_same_o",   # change subject only
    "same_p_vary_both",       # change subject AND object, keep predicate
]


# Semantic predicate clusters derived from per-predicate error analysis.
PREDICATE_CLUSTERS: Dict[str, int] = {
    # Cluster 0 -- the "confusion cluster", ~82 % of test data
    "ctdflat:increases_expression": 0,
    "ctdflat:decreases_expression": 0,
    "ctdflat:metabolic_processing": 0,
    "ctdflat:increases_activity": 0,
    "ctdflat:decreases_activity": 0,
    # Cluster 1 -- distinctive mechanisms
    "ctdflat:increases_response_to_substance": 1,
    "ctdflat:decreases_response_to_substance": 1,
    "ctdflat:increases_transport": 1,
    "ctdflat:decreases_transport": 1,
    "ctdflat:increases_abundance": 1,
    # Cluster 2 -- mutagenesis, very distinctive
    "ctdflat:increases_mutagenesis": 2,
}


@dataclass
class NegativeSampler:
    """Samples K hard negative triples per anchor from the KG embedding tables."""

    entity_lookup: Dict[str, np.ndarray]
    relation_lookup: Dict[str, np.ndarray]
    train_subjects: List[str]
    train_predicates: List[str]
    train_objects: List[str]
    ent_dim: int
    rel_dim: int
    seed: int = 42
    cluster_aware: bool = False
    combination: str = "concat"
    explicit_strategies: Optional[List[str]] = None  # None means use all STRATEGIES
    predicate_clusters: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._subj_arr = np.array(self.train_subjects, dtype=object)
        self._pred_arr = np.array(self.train_predicates, dtype=object)
        self._obj_arr = np.array(self.train_objects, dtype=object)
        self._ent = self.entity_lookup
        self._rel = self.relation_lookup

        self._cluster_pools: Dict[int, List[str]] = {}
        if self.cluster_aware:
            for p in self.train_predicates:
                c = self.predicate_clusters.get(p)
                if c is None:
                    continue
                self._cluster_pools.setdefault(c, []).append(p)
            logger.info(
                "cluster_aware_negs enabled: %d clusters, sizes=%s",
                len(self._cluster_pools),
                {c: len(pool) for c, pool in self._cluster_pools.items()},
            )

        if self.cluster_aware:
            self._strategies = [
                "same_cluster_vary_p",
                "same_s_vary_o",
                "vary_s_same_p_same_o",
                "same_p_vary_both",
            ]
        elif self.explicit_strategies is not None:
            if "none" in self.explicit_strategies:
                if len(self.explicit_strategies) != 1:
                    raise ValueError("'none' cannot be combined with other neg strategies.")
                self._strategies = []
                logger.info("Hard negative sampling disabled; using only in-batch negatives.")
                return
            invalid = set(self.explicit_strategies) - set(STRATEGIES)
            if invalid:
                raise ValueError(
                    f"Unknown neg strategies: {invalid}. Must be subset of {STRATEGIES}."
                )
            self._strategies = list(self.explicit_strategies)
            logger.info("Using explicit neg strategies: %s", self._strategies)
        else:
            self._strategies = list(STRATEGIES)

    def _combine(self, s: str, p: str, o: str) -> np.ndarray:
        return apply_spo_combination(
            self._ent[s], self._rel[p], self._ent[o], self.combination
        )

    def _sample_one(
        self, anchor: Tuple[str, str, str], strategy: str, max_retry: int = 32
    ) -> np.ndarray:
        s, p, o = anchor
        for _ in range(max_retry):
            if strategy == "same_s_same_o_vary_p":
                p2 = str(self._rng.choice(self._pred_arr))
                s2, o2 = s, o
                if p2 == p:
                    continue
            elif strategy == "same_cluster_vary_p":
                cluster = self.predicate_clusters.get(p)
                pool = self._cluster_pools.get(cluster) if cluster is not None else None
                if not pool or len(pool) < 2:
                    p2 = str(self._rng.choice(self._pred_arr))
                else:
                    p2 = str(self._rng.choice(pool))
                s2, o2 = s, o
                if p2 == p:
                    continue
            elif strategy == "same_s_vary_o":
                o2 = str(self._rng.choice(self._obj_arr))
                s2, p2 = s, p
                if o2 == o:
                    continue
            elif strategy == "vary_s_same_p_same_o":
                s2 = str(self._rng.choice(self._subj_arr))
                p2, o2 = p, o
                if s2 == s:
                    continue
            elif strategy == "same_p_vary_both":
                s2 = str(self._rng.choice(self._subj_arr))
                o2 = str(self._rng.choice(self._obj_arr))
                p2 = p
                if s2 == s and o2 == o:
                    continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            if s2 not in self._ent or o2 not in self._ent or p2 not in self._rel:
                continue
            return self._combine(s2, p2, o2)
        return self._combine(s, p, o)

    def sample(self, anchor: Tuple[str, str, str], k: int) -> np.ndarray:
        """Sample k hard negatives for one anchor. Returns (k, triple_dim) array."""
        if k == 0 or not self._strategies:
            out_dim = (
                2 * self.ent_dim + self.rel_dim
                if self.combination == "concat"
                else min(self.ent_dim, self.rel_dim)
            )
            return np.zeros((0, out_dim), dtype=np.float32)
        negs = []
        for i in range(k):
            strategy = self._strategies[i % len(self._strategies)]
            negs.append(self._sample_one(anchor, strategy))
        return np.stack(negs).astype(np.float32)


def build_sampler_from_train(
    evidence_df: pd.DataFrame,
    train_idx: np.ndarray,
    entity_lookup: Dict[str, np.ndarray],
    relation_lookup: Dict[str, np.ndarray],
    seed: int,
    cluster_aware: bool = False,
    combination: str = "concat",
    neg_strategies: Optional[List[str]] = None,
) -> NegativeSampler:
    train_df = evidence_df.iloc[train_idx]
    subjects = sorted(set(train_df["subject_uri"].tolist()))
    predicates = sorted(set(train_df["predicate_uri"].tolist()))
    objects = sorted(set(train_df["object_uri"].tolist()))
    subjects = [s for s in subjects if s in entity_lookup]
    predicates = [p for p in predicates if p in relation_lookup]
    objects = [o for o in objects if o in entity_lookup]
    ent_dim = len(next(iter(entity_lookup.values())))
    rel_dim = len(next(iter(relation_lookup.values())))
    return NegativeSampler(
        entity_lookup=entity_lookup,
        relation_lookup=relation_lookup,
        train_subjects=subjects,
        train_predicates=predicates,
        train_objects=objects,
        ent_dim=ent_dim,
        rel_dim=rel_dim,
        seed=seed,
        cluster_aware=cluster_aware,
        combination=combination,
        explicit_strategies=neg_strategies,
        predicate_clusters=PREDICATE_CLUSTERS if cluster_aware else {},
    )


def compute_sample_weights(
    evidence_df: pd.DataFrame,
    train_idx: np.ndarray,
    scheme: str = "inv_sqrt_freq",
) -> np.ndarray:
    """Compute per-row weights based on predicate frequency.

    Weights are normalised so their mean over train is 1.0.
    Rows outside train_idx get weight 1.0.
    """
    n = len(evidence_df)
    weights = np.ones(n, dtype=np.float32)
    if scheme == "none":
        return weights

    train_preds = evidence_df["predicate_uri"].iloc[train_idx].values
    counts = pd.Series(train_preds).value_counts().to_dict()
    if scheme == "inv_freq":
        raw = {p: 1.0 / c for p, c in counts.items()}
    elif scheme == "inv_sqrt_freq":
        raw = {p: 1.0 / np.sqrt(c) for p, c in counts.items()}
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")

    train_w = np.array([raw[p] for p in train_preds], dtype=np.float32)
    train_w /= train_w.mean()
    weights[train_idx] = train_w
    return weights


def make_collate_fn(
    sampler: NegativeSampler,
    evidence_df: pd.DataFrame,
    n_hard_negs: int,
    sample_weights: Optional[np.ndarray] = None,
):
    """Factory that produces a DataLoader collate_fn sampling negatives on the fly."""
    s_col = evidence_df["subject_uri"].tolist()
    p_col = evidence_df["predicate_uri"].tolist()
    o_col = evidence_df["object_uri"].tolist()
    use_weights = sample_weights is not None

    def _collate(batch):
        texts = torch.stack([b[0] for b in batch])
        triples = torch.stack([b[1] for b in batch])
        row_indices = [b[2] for b in batch]
        if n_hard_negs == 0:
            hard = torch.zeros((len(batch), 0, triples.shape[-1]))
        else:
            negs = np.stack(
                [
                    sampler.sample((s_col[ri], p_col[ri], o_col[ri]), n_hard_negs)
                    for ri in row_indices
                ]
            )
            hard = torch.from_numpy(negs).float()
        if use_weights:
            w = torch.from_numpy(
                np.asarray([sample_weights[ri] for ri in row_indices], dtype=np.float32)
            )
            return texts, triples, hard, w
        return texts, triples, hard

    return _collate
