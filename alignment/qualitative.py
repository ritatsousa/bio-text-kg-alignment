"""Qualitative exports for retrieval examples and top-ranked predictions."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from baselines import verbalize_triple
from data import PreparedData


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + 1e-12)


def _sample_test_indices(data: PreparedData, n_examples: int, seed: int) -> np.ndarray:
    test_idx = np.asarray(data.split_indices["test"], dtype=np.int64)
    if len(test_idx) <= n_examples:
        return test_idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(test_idx, size=n_examples, replace=False))


def _topk(query_vectors: np.ndarray, candidate_vectors: np.ndarray, top_k: int) -> tuple:
    sims = _l2_normalize(query_vectors) @ _l2_normalize(candidate_vectors).T
    k = min(top_k, sims.shape[1])
    idx = np.argpartition(-sims, kth=np.arange(k), axis=1)[:, :k]
    scores = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-scores, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    scores = np.take_along_axis(scores, order, axis=1)
    return idx, scores


def _triple_fields(row: pd.Series, prefix: str) -> Dict:
    return {
        f"{prefix}_triple_id": row.get("triple_id", ""),
        f"{prefix}_subject_uri": row.get("subject_uri", ""),
        f"{prefix}_predicate_uri": row.get("predicate_uri", ""),
        f"{prefix}_object_uri": row.get("object_uri", ""),
        f"{prefix}_subject_name": row.get("subject_name", ""),
        f"{prefix}_object_name": row.get("object_name", ""),
        f"{prefix}_verbalized_triple": verbalize_triple(
            str(row.get("subject_uri", "")),
            str(row.get("predicate_uri", "")),
            str(row.get("object_uri", "")),
            str(row.get("subject_name", "")),
            str(row.get("object_name", "")),
        ),
    }


def _sentence_fields(row: pd.Series, prefix: str) -> Dict:
    return {
        f"{prefix}_pmid": row.get("pmid", ""),
        f"{prefix}_text_id": row.get("text_id", ""),
        f"{prefix}_text": row.get("text", ""),
    }


def save_verbalized_triple_examples(
    data: PreparedData,
    out_path: Path,
    n_examples: int = 10,
    seed: int = 42,
) -> None:
    """Save random examples of URI triples and their verbalized text."""
    n = len(data.evidence_df)
    if n == 0:
        pd.DataFrame().to_csv(out_path, index=False)
        return
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(np.arange(n), size=min(n_examples, n), replace=False))
    rows: List[Dict] = []
    for row_idx in indices:
        ev = data.evidence_df.iloc[int(row_idx)]
        rows.append({"row_index": int(row_idx), **_triple_fields(ev, "triple")})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def save_retrieval_top_predictions(
    data: PreparedData,
    direction: str,
    query_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    out_path: Path,
    n_queries: int = 10,
    top_k: int = 10,
    seed: int = 42,
) -> None:
    """Save top-k retrieval predictions for random test-set queries.

    In ``s2t``, queries are test-set sentences and candidates are triples.
    In ``t2s``, queries are test-set triples and candidates are sentences.
    """
    if direction not in {"s2t", "t2s"}:
        raise ValueError("direction must be either 's2t' or 't2s'")

    selected = _sample_test_indices(data, n_queries, seed)
    if len(selected) == 0:
        pd.DataFrame().to_csv(out_path, index=False)
        return

    top_idx, top_scores = _topk(query_vectors, candidate_vectors, top_k)
    rows: List[Dict] = []
    evidence = data.evidence_df

    for q_pos, query_row_idx in enumerate(selected):
        query_ev = evidence.iloc[int(query_row_idx)]
        if direction == "s2t":
            base = {
                "direction": direction,
                "query_number": q_pos,
                "query_row_index": int(query_row_idx),
                **_sentence_fields(query_ev, "query"),
                **_triple_fields(query_ev, "true"),
            }
            for rank, cand_row_idx in enumerate(top_idx[q_pos], start=1):
                cand_ev = evidence.iloc[int(cand_row_idx)]
                rows.append(
                    {
                        **base,
                        "prediction_rank": rank,
                        "candidate_row_index": int(cand_row_idx),
                        "score": float(top_scores[q_pos, rank - 1]),
                        **_triple_fields(cand_ev, "candidate"),
                        "is_true": int(cand_row_idx == query_row_idx),
                    }
                )
        else:
            base = {
                "direction": direction,
                "query_number": q_pos,
                "query_row_index": int(query_row_idx),
                **_triple_fields(query_ev, "query"),
                **_sentence_fields(query_ev, "true"),
            }
            for rank, cand_row_idx in enumerate(top_idx[q_pos], start=1):
                cand_ev = evidence.iloc[int(cand_row_idx)]
                rows.append(
                    {
                        **base,
                        "prediction_rank": rank,
                        "candidate_row_index": int(cand_row_idx),
                        "score": float(top_scores[q_pos, rank - 1]),
                        **_sentence_fields(cand_ev, "candidate"),
                        "is_true": int(cand_row_idx == query_row_idx),
                    }
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def save_baseline_top_predictions(
    data: PreparedData,
    verbalized_triple_embs: np.ndarray,
    direction: str,
    out_path: Path,
    n_queries: int = 10,
    top_k: int = 10,
    seed: int = 42,
) -> None:
    """Save top-k predictions for the Verbalized-Triple-NN baseline."""
    selected = _sample_test_indices(data, n_queries, seed)
    if direction == "s2t":
        query_vectors = data.text_embs[selected]
        candidate_vectors = verbalized_triple_embs
    elif direction == "t2s":
        query_vectors = verbalized_triple_embs[selected]
        candidate_vectors = data.text_embs
    else:
        raise ValueError("direction must be either 's2t' or 't2s'")

    save_retrieval_top_predictions(
        data,
        direction,
        query_vectors,
        candidate_vectors,
        out_path,
        n_queries=n_queries,
        top_k=top_k,
        seed=seed,
    )


def predict_in_batches(
    model: torch.nn.Module,
    text_embs: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(text_embs), batch_size):
            x = torch.from_numpy(text_embs[start : start + batch_size]).float().to(device)
            preds.append(model(x).cpu().numpy())
    return np.concatenate(preds, axis=0)


def save_model_top_predictions(
    model: torch.nn.Module,
    data: PreparedData,
    direction: str,
    out_path: Path,
    device: torch.device,
    training_direction: str,
    n_queries: int = 10,
    top_k: int = 10,
    seed: int = 42,
) -> None:
    """Save top-k predictions for a trained text-to-triple projector."""
    selected = _sample_test_indices(data, n_queries, seed)
    if direction == "s2t":
        if training_direction in {"text_to_kg", "bidirectional_random"}:
            if training_direction == "bidirectional_random":
                model.eval()
                preds = []
                with torch.no_grad():
                    x = torch.from_numpy(data.text_embs[selected]).float().to(device)
                    preds.append(model.project_text_to_kg(x).cpu().numpy())
                projected_queries = np.concatenate(preds, axis=0)
            else:
                projected_queries = predict_in_batches(model, data.text_embs[selected], device)
            candidate_vectors = data.triple_embs
        elif training_direction == "kg_to_text":
            projected_queries = data.text_embs[selected]
            model.eval()
            preds = []
            with torch.no_grad():
                for start in range(0, len(data.triple_embs), 4096):
                    x = torch.from_numpy(data.triple_embs[start : start + 4096]).float().to(device)
                    preds.append(model(x).cpu().numpy())
            candidate_vectors = np.concatenate(preds, axis=0)
        else:
            raise ValueError("Invalid training_direction")
        save_retrieval_top_predictions(
            data,
            direction,
            projected_queries,
            candidate_vectors,
            out_path,
            n_queries=n_queries,
            top_k=top_k,
            seed=seed,
        )
    elif direction == "t2s":
        if training_direction in {"kg_to_text", "bidirectional_random"}:
            if training_direction == "bidirectional_random":
                model.eval()
                with torch.no_grad():
                    x = torch.from_numpy(data.triple_embs[selected]).float().to(device)
                    query_vectors = model.project_kg_to_text(x).cpu().numpy()
            else:
                query_vectors = predict_in_batches(model, data.triple_embs[selected], device)
            candidate_vectors = data.text_embs
        elif training_direction == "text_to_kg":
            query_vectors = data.triple_embs[selected]
            projected_sentences = predict_in_batches(model, data.text_embs, device)
            candidate_vectors = projected_sentences
        else:
            raise ValueError("Invalid training_direction")
        save_retrieval_top_predictions(
            data,
            direction,
            query_vectors,
            candidate_vectors,
            out_path,
            n_queries=n_queries,
            top_k=top_k,
            seed=seed,
        )
    else:
        raise ValueError("direction must be either 's2t' or 't2s'")
