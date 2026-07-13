"""Evaluation metrics: cosine sim, Hits@k, MRR, median rank, per-predicate."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + 1e-12)


def _compute_rankings_sim(
    sim: np.ndarray, true_indices: np.ndarray, direction: str
) -> np.ndarray:
    """Compute 1-based ranks from an explicit candidate similarity matrix.

    For s2t, sim is (n_query_sentences, n_candidate_triples), and
    true_indices[i] is query i's true triple row in the candidate triple matrix.

    For t2s, sim is (n_candidate_sentences, n_query_triples), and
    true_indices[j] is query triple j's true sentence row in the candidate
    sentence matrix.
    """
    true_indices = np.asarray(true_indices, dtype=np.int64)
    if direction == "s2t":
        true_scores = sim[np.arange(sim.shape[0]), true_indices]
        return (sim > true_scores[:, None]).sum(axis=1) + 1
    if direction == "t2s":
        true_scores = sim[true_indices, np.arange(sim.shape[1])]
        return (sim > true_scores[None, :]).sum(axis=0) + 1
    raise ValueError(f"direction must be 's2t' or 't2s', got '{direction}'")


def compute_rankings(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """For each row i, rank truth[i] among all rows in truth."""
    pred_n = _l2_normalize(pred)
    truth_n = _l2_normalize(truth)
    sim = pred_n @ truth_n.T
    return _compute_rankings_sim(sim, np.arange(len(pred)), "s2t")


def compute_cosine(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred_n = _l2_normalize(pred)
    truth_n = _l2_normalize(truth)
    return np.sum(pred_n * truth_n, axis=1)


def _direction_metrics(
    ranks: np.ndarray,
    cos: np.ndarray,
    hits_k: List[int],
    prefix: str,
    pool_size: int,
    predicates: Optional[List[str]] = None,
) -> Dict:
    """Compute scalar metrics for one ranking direction, prefixed with prefix."""
    out: Dict = {
        f"{prefix}mrr": float((1.0 / ranks).mean()),
        f"{prefix}median_rank": float(np.median(ranks)),
        f"{prefix}median_rank_pct": float(np.median(ranks) / max(pool_size, 1)),
    }
    for k in hits_k:
        out[f"{prefix}hits@{k}"] = float((ranks <= k).mean())

    if predicates is not None:
        per_pred: Dict = {}
        pred_arr = np.array(predicates)
        for p in sorted(set(predicates)):
            mask = pred_arr == p
            if mask.sum() == 0:
                continue
            pname = p.split(":")[-1]
            sub_ranks = ranks[mask]
            sub_cos = cos[mask]
            entry: Dict = {
                "n": int(mask.sum()),
                "cosine_mean": float(sub_cos.mean()),
                f"{prefix}mrr": float((1.0 / sub_ranks).mean()),
                f"{prefix}median_rank": float(np.median(sub_ranks)),
            }
            for k in hits_k:
                entry[f"{prefix}hits@{k}"] = float((sub_ranks <= k).mean())
            per_pred[pname] = entry
        out[f"{prefix}per_predicate"] = per_pred

        if per_pred:
            mrrs = [e[f"{prefix}mrr"] for e in per_pred.values()]
            out[f"{prefix}macro_mrr"] = float(np.mean(mrrs))
            out[f"{prefix}macro_mrr_min"] = float(np.min(mrrs))
            for k in hits_k:
                hits_vals = [e[f"{prefix}hits@{k}"] for e in per_pred.values()]
                out[f"{prefix}macro_hits@{k}"] = float(np.mean(hits_vals))
    return out


def compute_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    hits_k: List[int],
    predicates: Optional[List[str]] = None,
    direction: str = "s2t",
    candidate_pred: Optional[np.ndarray] = None,
    candidate_truth: Optional[np.ndarray] = None,
    true_indices: Optional[np.ndarray] = None,
) -> Dict:
    """Compute cosine stats and ranking metrics.

    pred and truth are the paired query examples used for cosine_mean.

    For s2t, each predicted sentence embedding is ranked against
    candidate_truth, which should contain all KG triple embeddings.

    For t2s, each query triple is ranked against candidate_pred, which should
    contain projected embeddings for all sentences.

    true_indices gives the matching candidate row for each query. If omitted,
    positional alignment is assumed for backwards compatibility.
    """
    if direction not in {"s2t", "t2s"}:
        raise ValueError("direction must be either 's2t' or 't2s'")

    if true_indices is None:
        true_indices = np.arange(len(pred), dtype=np.int64)
    else:
        true_indices = np.asarray(true_indices, dtype=np.int64)

    cos = compute_cosine(pred, truth)
    pred_n = _l2_normalize(pred)
    truth_n = _l2_normalize(truth)

    if direction == "s2t":
        candidates = truth if candidate_truth is None else candidate_truth
        sim = pred_n @ _l2_normalize(candidates).T
        pool_size = sim.shape[1]
    else:
        candidates = pred if candidate_pred is None else candidate_pred
        sim = _l2_normalize(candidates) @ truth_n.T
        pool_size = sim.shape[0]

    ranks = _compute_rankings_sim(sim, true_indices, direction)
    out: Dict = {
        "cosine_mean": float(cos.mean()),
        "cosine_std": float(cos.std()),
        "pool_size": int(pool_size),
    }
    out.update(_direction_metrics(ranks, cos, hits_k, "", pool_size, predicates))
    return out


def _predict_in_batches(
    model: torch.nn.Module,
    text_embs: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for start in range(0, len(text_embs), batch_size):
            x = torch.from_numpy(text_embs[start : start + batch_size]).float().to(device)
            preds.append(model(x).cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_model(
    model: torch.nn.Module,
    text_embs: np.ndarray,
    triple_embs: np.ndarray,
    device: torch.device,
    hits_k: List[int],
    predicates: Optional[List[str]] = None,
    direction: str = "s2t",
    candidate_text_embs: Optional[np.ndarray] = None,
    candidate_triple_embs: Optional[np.ndarray] = None,
    true_indices: Optional[np.ndarray] = None,
) -> Dict:
    """Run model and compute retrieval metrics against explicit candidates."""
    model.eval()
    pred = _predict_in_batches(model, text_embs, device)

    candidate_pred = None
    if direction == "t2s" and candidate_text_embs is not None:
        candidate_pred = _predict_in_batches(model, candidate_text_embs, device)

    return compute_metrics(
        pred,
        triple_embs,
        hits_k,
        predicates=predicates,
        direction=direction,
        candidate_pred=candidate_pred,
        candidate_truth=candidate_triple_embs,
        true_indices=true_indices,
    )


def _project_text_to_kg(
    model: torch.nn.Module,
    text_embs: np.ndarray,
    device: torch.device,
    training_direction: str,
) -> np.ndarray:
    if training_direction == "bidirectional_random":
        fn = model.project_text_to_kg
    elif training_direction == "text_to_kg":
        fn = model
    else:
        raise ValueError("text_to_kg projection is unavailable for kg_to_text models")

    preds = []
    with torch.no_grad():
        for start in range(0, len(text_embs), 4096):
            x = torch.from_numpy(text_embs[start : start + 4096]).float().to(device)
            preds.append(fn(x).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _project_kg_to_text(
    model: torch.nn.Module,
    triple_embs: np.ndarray,
    device: torch.device,
    training_direction: str,
) -> np.ndarray:
    if training_direction == "bidirectional_random":
        fn = model.project_kg_to_text
    elif training_direction == "kg_to_text":
        fn = model
    else:
        raise ValueError("kg_to_text projection is unavailable for text_to_kg models")

    preds = []
    with torch.no_grad():
        for start in range(0, len(triple_embs), 4096):
            x = torch.from_numpy(triple_embs[start : start + 4096]).float().to(device)
            preds.append(fn(x).cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_alignment_model(
    model: torch.nn.Module,
    text_embs: np.ndarray,
    triple_embs: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    hits_k: List[int],
    training_direction: str,
    eval_direction: str,
    predicates: Optional[List[str]] = None,
) -> Dict:
    """Evaluate retrieval with fixed s2t/t2s semantics.

    s2t always means sentence queries ranked against all triples. t2s always
    means triple queries ranked against all sentences. The learned projection
    used to put query/candidates in a common space depends on
    ``training_direction``.
    """
    if eval_direction not in {"s2t", "t2s"}:
        raise ValueError("eval_direction must be either 's2t' or 't2s'")
    if training_direction not in {"text_to_kg", "kg_to_text", "bidirectional_random"}:
        raise ValueError(
            "training_direction must be one of 'text_to_kg', 'kg_to_text', "
            "or 'bidirectional_random'"
        )

    indices = np.asarray(indices, dtype=np.int64)
    model.eval()

    if eval_direction == "s2t":
        if training_direction in {"text_to_kg", "bidirectional_random"}:
            pred = _project_text_to_kg(model, text_embs[indices], device, training_direction)
            truth = triple_embs[indices]
            return compute_metrics(
                pred,
                truth,
                hits_k,
                predicates=predicates,
                direction="s2t",
                candidate_truth=triple_embs,
                true_indices=indices,
            )

        projected_triples = _project_kg_to_text(model, triple_embs, device, training_direction)
        return compute_metrics(
            text_embs[indices],
            projected_triples[indices],
            hits_k,
            predicates=predicates,
            direction="s2t",
            candidate_truth=projected_triples,
            true_indices=indices,
        )

    if training_direction in {"kg_to_text", "bidirectional_random"}:
        projected_queries = _project_kg_to_text(
            model, triple_embs[indices], device, training_direction
        )
        return compute_metrics(
            text_embs[indices],
            projected_queries,
            hits_k,
            predicates=predicates,
            direction="t2s",
            candidate_pred=text_embs,
            true_indices=indices,
        )

    projected_sentences = _project_text_to_kg(model, text_embs, device, training_direction)
    return compute_metrics(
        projected_sentences[indices],
        triple_embs[indices],
        hits_k,
        predicates=predicates,
        direction="t2s",
        candidate_pred=projected_sentences,
        true_indices=indices,
    )


def per_predicate_dataframe(metrics: Dict, direction: str = "s2t") -> pd.DataFrame:
    """Build a per-predicate DataFrame from a metrics dict."""
    if direction not in {"s2t", "t2s"}:
        raise ValueError("direction must be either 's2t' or 't2s'")

    pp = metrics.get("per_predicate", {})
    if not pp:
        return pd.DataFrame()
    rows = [{"predicate": name, **d} for name, d in pp.items()]
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def average_fold_metrics(fold_metrics: List[Dict]) -> Dict:
    """Average scalar metrics across folds; add a _std entry for each."""
    if not fold_metrics:
        return {}

    scalar_keys = [
        k for k in fold_metrics[0]
        if isinstance(fold_metrics[0][k], (int, float)) and k != "pool_size"
    ]
    averaged: Dict = {}
    for k in scalar_keys:
        values = [m[k] for m in fold_metrics if k in m and isinstance(m[k], (int, float))]
        averaged[k] = float(np.mean(values))
        averaged[f"{k}_std"] = float(np.std(values))
    averaged["pool_size"] = int(np.mean([m.get("pool_size", 0) for m in fold_metrics]))
    averaged["n_folds"] = len(fold_metrics)

    all_predicates: set = set()
    for m in fold_metrics:
        all_predicates.update(m.get("per_predicate", {}).keys())

    if all_predicates:
        per_pred_avg: Dict = {}
        for pred in sorted(all_predicates):
            entries = [
                m["per_predicate"][pred]
                for m in fold_metrics
                if pred in m.get("per_predicate", {})
            ]
            if not entries:
                continue
            pred_scalar_keys = [
                k for k in entries[0]
                if isinstance(entries[0][k], (int, float)) and k != "n"
            ]
            per_pred_avg[pred] = {
                k: float(np.mean([e[k] for e in entries if k in e]))
                for k in pred_scalar_keys
            }
            per_pred_avg[pred]["n"] = int(np.mean([e["n"] for e in entries if "n" in e]))
        averaged["per_predicate"] = per_pred_avg

    return averaged
