"""Data loading: text embeddings, KG embeddings, alignment, splits, dataset."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import Dataset

from config import DefaultConfig, SplitConfig
from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_evidence(path: Path) -> pd.DataFrame:
    """Load deduplicated evidence TSV."""
    df = pd.read_csv(path, sep="\t")
    logger.info("Loaded evidence: %d rows from %s", len(df), path.name)
    return df


def load_text_embeddings(text_model: str, text_emb_dir: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load raw text embedding matrix and its index table.

    Returns:
        embs: (N_full, 768) float32
        index_df: DataFrame with columns [idx, text_id]
    """
    emb_path = text_emb_dir / f"embeddings_{text_model}.npy"
    idx_path = text_emb_dir / "index.tsv"
    if not emb_path.exists():
        raise FileNotFoundError(f"Text embedding file not found: {emb_path}")
    embs = np.load(emb_path).astype(np.float32)
    index_df = pd.read_csv(idx_path, sep="\t")
    if len(index_df) != len(embs):
        raise ValueError(
            f"index.tsv ({len(index_df)}) != embeddings ({len(embs)}) for {text_model}"
        )
    logger.info("Loaded text embeddings '%s': shape=%s", text_model, embs.shape)
    return embs, index_df


def align_text_to_evidence(
    evidence_df: pd.DataFrame,
    text_embs: np.ndarray,
    index_df: pd.DataFrame,
) -> np.ndarray:
    """Build a (N_evidence, 768) matrix of text embeddings aligned to evidence rows."""
    text_id_to_row = dict(zip(index_df["text_id"].tolist(), index_df["idx"].tolist()))
    gather_idx = np.array(
        [text_id_to_row[t] for t in evidence_df["text_id"].tolist()], dtype=np.int64
    )
    aligned = text_embs[gather_idx]
    logger.info("Aligned text embeddings: %s", aligned.shape)
    return aligned


def load_kg_tables(
    family: str, config: str, kg_emb_dir: Path
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int, int]:
    """Load entity and relation embedding tables from a KG config dir.

    Returns:
        entity_lookup: {entity_id: vector}
        relation_lookup: {relation_id: vector}
        ent_dim, rel_dim
    """
    cfg_dir = kg_emb_dir / family / config
    ent_path = cfg_dir / "entity_embeddings.csv"
    rel_path = cfg_dir / "relation_embeddings.csv"
    ent_df = pd.read_csv(ent_path)
    rel_df = pd.read_csv(rel_path)
    ent_ids = ent_df.iloc[:, 0].astype(str).tolist()
    rel_ids = rel_df.iloc[:, 0].astype(str).tolist()
    ent_vecs = ent_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    rel_vecs = rel_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    entity_lookup = {eid: ent_vecs[i] for i, eid in enumerate(ent_ids)}
    relation_lookup = {rid: rel_vecs[i] for i, rid in enumerate(rel_ids)}
    ent_dim = ent_vecs.shape[1]
    rel_dim = rel_vecs.shape[1]
    logger.info(
        "Loaded KG '%s/%s': %d entities (dim=%d), %d relations (dim=%d)",
        family, config, len(entity_lookup), ent_dim, len(relation_lookup), rel_dim,
    )
    return entity_lookup, relation_lookup, ent_dim, rel_dim


# ---------------------------------------------------------------------------
# Triple combination strategies
# ---------------------------------------------------------------------------

COMBINATION_STRATEGIES = ["concat", "hadamard", "l1", "l2"]


def triple_dim_for_strategy(ent_dim: int, rel_dim: int, strategy: str) -> int:
    """Output dimension of a triple embedding for the given combination strategy."""
    if strategy == "concat":
        return 2 * ent_dim + rel_dim
    if strategy in ("hadamard", "l1", "l2"):
        return min(ent_dim, rel_dim)
    raise ValueError(f"Unknown strategy '{strategy}'. Must be one of {COMBINATION_STRATEGIES}.")


def apply_spo_combination(
    s: np.ndarray, p: np.ndarray, o: np.ndarray, combination: str
) -> np.ndarray:
    """Combine subject, predicate, object vectors according to *combination* strategy.

    Supports both 1-D (single triple) and 2-D (batch) inputs; output mirrors input.

    Strategies
    ----------
    concat   : [s; p; o]                    → dim = 2*ent_dim + rel_dim
    hadamard : s[:d] * p[:d] * o[:d]        → dim = min(ent_dim, rel_dim)
    l1       : |s[:d] + p[:d] - o[:d]|      → dim = min(ent_dim, rel_dim)
    l2       : (s[:d] + p[:d] - o[:d]) ** 2 → dim = min(ent_dim, rel_dim)

    For hadamard/l1/l2, d = min(ent_dim, rel_dim) so all three are truncated to
    a common size, handling KG models where entity and relation dimensions differ.
    """
    scalar = s.ndim == 1
    if scalar:
        s, p, o = s[np.newaxis], p[np.newaxis], o[np.newaxis]
    if combination == "concat":
        result = np.concatenate([s, p, o], axis=1)
    else:
        d = min(s.shape[1], p.shape[1])
        sv, pv, ov = s[:, :d], p[:, :d], o[:, :d]
        if combination == "hadamard":
            result = sv * pv * ov
        elif combination == "l1":
            result = np.abs(sv + pv - ov)
        elif combination == "l2":
            result = (sv + pv - ov) ** 2
        else:
            raise ValueError(
                f"Unknown triple combination strategy '{combination}'. "
                f"Must be one of {COMBINATION_STRATEGIES}."
            )
    return result[0] if scalar else result


def _extract_spo_vectors(
    evidence_df: pd.DataFrame,
    entity_lookup: Dict[str, np.ndarray],
    relation_lookup: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract raw s/p/o vectors for each evidence row.

    Returns:
        s_vecs:     (N, ent_dim) float32 — zero rows where lookup failed
        p_vecs:     (N, rel_dim) float32 — zero rows where lookup failed
        o_vecs:     (N, ent_dim) float32 — zero rows where lookup failed
        valid_mask: (N,) bool
    """
    s_uris = evidence_df["subject_uri"].tolist()
    p_uris = evidence_df["predicate_uri"].tolist()
    o_uris = evidence_df["object_uri"].tolist()
    ent_dim = len(next(iter(entity_lookup.values())))
    rel_dim = len(next(iter(relation_lookup.values())))
    n = len(evidence_df)
    s_vecs = np.zeros((n, ent_dim), dtype=np.float32)
    p_vecs = np.zeros((n, rel_dim), dtype=np.float32)
    o_vecs = np.zeros((n, ent_dim), dtype=np.float32)
    valid_mask = np.zeros(n, dtype=bool)
    for i, (s, p, o) in enumerate(zip(s_uris, p_uris, o_uris)):
        sv = entity_lookup.get(s)
        pv = relation_lookup.get(p)
        ov = entity_lookup.get(o)
        if sv is None or pv is None or ov is None:
            continue
        s_vecs[i] = sv
        p_vecs[i] = pv
        o_vecs[i] = ov
        valid_mask[i] = True
    return s_vecs, p_vecs, o_vecs, valid_mask


def build_triple_embeddings(
    evidence_df: pd.DataFrame,
    entity_lookup: Dict[str, np.ndarray],
    relation_lookup: Dict[str, np.ndarray],
    combination: str = "concat",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build triple embeddings for a single combination strategy.

    Returns:
        triple_embs: (N, triple_dim) float32
        valid_mask:  (N,) bool — rows where all three lookups succeeded.
    """
    s_vecs, p_vecs, o_vecs, valid_mask = _extract_spo_vectors(
        evidence_df, entity_lookup, relation_lookup
    )
    triple_embs = apply_spo_combination(s_vecs, p_vecs, o_vecs, combination)
    logger.info(
        "Built triple embeddings (%s): %d/%d valid, triple_dim=%d",
        combination, int(valid_mask.sum()), len(evidence_df), triple_embs.shape[1],
    )
    return triple_embs, valid_mask


def build_all_triple_embeddings(
    evidence_df: pd.DataFrame,
    entity_lookup: Dict[str, np.ndarray],
    relation_lookup: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Build triple embeddings for all combination strategies in one pass.

    Extracts s/p/o vectors once and applies each strategy, avoiding redundant
    KG lookup iteration.

    Returns:
        triple_embs_by_strategy: dict strategy → (N, dim) float32 array
        valid_mask:              (N,) bool
    """
    s_vecs, p_vecs, o_vecs, valid_mask = _extract_spo_vectors(
        evidence_df, entity_lookup, relation_lookup
    )
    result: Dict[str, np.ndarray] = {}
    for strategy in COMBINATION_STRATEGIES:
        embs = apply_spo_combination(s_vecs, p_vecs, o_vecs, strategy)
        result[strategy] = embs
        logger.info(
            "Built triple embeddings (%s): triple_dim=%d", strategy, embs.shape[1]
        )
    logger.info(
        "All strategies: %d/%d rows valid.", int(valid_mask.sum()), len(evidence_df)
    )
    return result, valid_mask


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def k_fold_grouped_split(
    df: pd.DataFrame, split_cfg: SplitConfig, seed: int
) -> List[Dict[str, np.ndarray]]:
    """K-fold CV split by PMID groups, stratified by majority predicate per PMID.

    Each fold's train portion is further split to carve out a val set
    (``split_cfg.val_fraction`` of the train set) for early stopping.

    Returns:
        List of ``n_folds`` dicts, each with keys ``"train"``, ``"val"``, ``"test"``
        containing row indices into *df*.
    """
    pmid_pred = df.groupby("pmid")["predicate_uri"].agg(lambda x: x.mode()[0])
    pmids = pmid_pred.index.values
    pmid_labels = pmid_pred.values
    pmid_arr = df["pmid"].values

    skf = StratifiedKFold(n_splits=split_cfg.n_folds, shuffle=True, random_state=seed)
    folds: List[Dict[str, np.ndarray]] = []

    for fold_i, (trainval_pmid_idx, test_pmid_idx) in enumerate(
        skf.split(pmids, pmid_labels)
    ):
        trainval_pmids = pmids[trainval_pmid_idx]
        trainval_labels = pmid_labels[trainval_pmid_idx]
        test_pmids = set(pmids[test_pmid_idx])

        # Carve val out of the trainval portion, stratified.
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=split_cfg.val_fraction,
            random_state=seed + fold_i,
        )
        train_pmid_idx, val_pmid_idx = next(sss.split(trainval_pmids, trainval_labels))
        train_pmids = set(trainval_pmids[train_pmid_idx])
        val_pmids = set(trainval_pmids[val_pmid_idx])

        train_idx = np.where(np.isin(pmid_arr, list(train_pmids)))[0]
        val_idx   = np.where(np.isin(pmid_arr, list(val_pmids)))[0]
        test_idx  = np.where(np.isin(pmid_arr, list(test_pmids)))[0]

        logger.info(
            "Fold %d/%d: train=%d, val=%d, test=%d",
            fold_i + 1, split_cfg.n_folds,
            len(train_idx), len(val_idx), len(test_idx),
        )
        folds.append({"train": train_idx, "val": val_idx, "test": test_idx})

    return folds


# ---------------------------------------------------------------------------
# Dataset bundle
# ---------------------------------------------------------------------------


@dataclass
class PreparedData:
    text_embs: np.ndarray        # (N, 768)
    triple_embs: np.ndarray      # (N, triple_dim) — active combination strategy
    evidence_df: pd.DataFrame    # (N,) aligned
    ent_dim: int
    rel_dim: int
    triple_dim: int
    entity_lookup: Dict[str, np.ndarray]
    relation_lookup: Dict[str, np.ndarray]
    split_indices: Dict[str, np.ndarray]
    triple_embs_by_strategy: Dict[str, np.ndarray] = field(default_factory=dict)
    triple_dim_by_strategy: Dict[str, int] = field(default_factory=dict)

    @property
    def text_dim(self) -> int:
        return self.text_embs.shape[1]

    def with_strategy(self, strategy: str) -> "PreparedData":
        """Return a shallow copy using the given triple combination strategy.

        Only ``triple_embs`` and ``triple_dim`` are updated; all other fields
        (including ``text_embs``, ``evidence_df``, ``split_indices``) are shared
        by reference so no extra memory is allocated.
        """
        if strategy not in self.triple_embs_by_strategy:
            raise ValueError(
                f"Strategy '{strategy}' not precomputed. "
                f"Available: {list(self.triple_embs_by_strategy)}"
            )
        return replace(
            self,
            triple_embs=self.triple_embs_by_strategy[strategy],
            triple_dim=self.triple_dim_by_strategy[strategy],
        )


def prepare_data(cfg: DefaultConfig) -> List[PreparedData]:
    """Load all data and return one PreparedData per CV fold."""
    if cfg.paths is None:
        raise ValueError("cfg.paths must be set before calling prepare_data")

    evidence_path = Path(cfg.paths.evidence_path)
    text_emb_dir = Path(cfg.paths.text_emb_dir)
    kg_emb_dir = Path(cfg.paths.kg_emb_dir)

    evidence_df = load_evidence(evidence_path)
    text_embs_full, index_df = load_text_embeddings(cfg.text_model, text_emb_dir)
    text_embs_aligned = align_text_to_evidence(evidence_df, text_embs_full, index_df)

    entity_lookup, relation_lookup, ent_dim, rel_dim = load_kg_tables(
        cfg.kg_family, cfg.kg_config, kg_emb_dir
    )
    triple_embs_all, valid_mask = build_all_triple_embeddings(
        evidence_df, entity_lookup, relation_lookup
    )
    if not valid_mask.all():
        n_drop = (~valid_mask).sum()
        logger.warning("Dropping %d rows with missing KG lookups", n_drop)
        evidence_df = evidence_df[valid_mask].reset_index(drop=True)
        text_embs_aligned = text_embs_aligned[valid_mask]
        triple_embs_all = {s: e[valid_mask] for s, e in triple_embs_all.items()}

    triple_dim_by_strategy = {
        s: triple_dim_for_strategy(ent_dim, rel_dim, s) for s in COMBINATION_STRATEGIES
    }
    default_combination = cfg.triple_combination
    folds = k_fold_grouped_split(evidence_df, cfg.split, cfg.seed)
    logger.info("Prepared %d CV folds.", len(folds))

    return [
        PreparedData(
            text_embs=text_embs_aligned,
            triple_embs=triple_embs_all[default_combination],
            evidence_df=evidence_df,
            ent_dim=ent_dim,
            rel_dim=rel_dim,
            triple_dim=triple_dim_by_strategy[default_combination],
            entity_lookup=entity_lookup,
            relation_lookup=relation_lookup,
            split_indices=fold_indices,
            triple_embs_by_strategy=triple_embs_all,
            triple_dim_by_strategy=triple_dim_by_strategy,
        )
        for fold_indices in folds
    ]


def prepare_text_only_data(cfg: DefaultConfig) -> List[PreparedData]:
    """Load evidence/text embeddings and return CV folds without KG lookups.

    This is intended for text-only baselines such as Verbalized-Triple-NN. The
    returned ``PreparedData`` objects keep KG fields empty because the baseline
    uses only ``text_embs``, ``evidence_df``, and ``split_indices``.
    """
    if cfg.paths is None:
        raise ValueError("cfg.paths must be set before calling prepare_text_only_data")

    evidence_path = Path(cfg.paths.evidence_path)
    text_emb_dir = Path(cfg.paths.text_emb_dir)

    evidence_df = load_evidence(evidence_path)
    text_embs_full, index_df = load_text_embeddings(cfg.text_model, text_emb_dir)
    text_embs_aligned = align_text_to_evidence(evidence_df, text_embs_full, index_df)

    folds = k_fold_grouped_split(evidence_df, cfg.split, cfg.seed)
    logger.info("Prepared %d text-only CV folds.", len(folds))

    empty_triples = np.zeros((len(evidence_df), 0), dtype=np.float32)
    return [
        PreparedData(
            text_embs=text_embs_aligned,
            triple_embs=empty_triples,
            evidence_df=evidence_df,
            ent_dim=0,
            rel_dim=0,
            triple_dim=0,
            entity_lookup={},
            relation_lookup={},
            split_indices=fold_indices,
            triple_embs_by_strategy={},
            triple_dim_by_strategy={},
        )
        for fold_indices in folds
    ]


class AlignmentDataset(Dataset):
    """Dataset returning (text_emb, triple_emb, row_index_in_evidence)."""

    def __init__(
        self,
        text_embs: np.ndarray,
        triple_embs: np.ndarray,
        indices: np.ndarray,
    ):
        self.text = torch.from_numpy(text_embs[indices]).float()
        self.triple = torch.from_numpy(triple_embs[indices]).float()
        self.indices = torch.from_numpy(np.asarray(indices, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.text[i], self.triple[i], int(self.indices[i].item())
