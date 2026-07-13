"""Non-learned baselines for text/triple retrieval."""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

import numpy as np

from data import PreparedData
from evaluation import compute_metrics
from utils import get_logger

logger = get_logger(__name__)


TEXT_MODEL_ENCODERS = {
    # Edit these if the stored text embeddings came from local/fine-tuned checkpoints.
    "biobert_mcpt": "dmis-lab/biobert-base-cased-v1.1",
    "pubmedbert_mcpt": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
}


_NS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*:")


def _strip_namespace(uri: str) -> str:
    label = _NS_RE.sub("", uri)
    return label.replace("_", " ").strip()


def verbalize_triple(
    subject_uri: str,
    predicate_uri: str,
    object_uri: str,
    subject_name: str = "",
    object_name: str = "",
) -> str:
    """Produce a short text string from triple labels.

    Subject/object names are preferred because URI identifiers such as
    ``ctd:D003520`` are not meaningful natural-language inputs. URI labels are
    used only as a fallback when names are unavailable.
    """
    s = str(subject_name).strip() or _strip_namespace(subject_uri)
    p = _strip_namespace(predicate_uri)
    o = str(object_name).strip() or _strip_namespace(object_uri)
    return f"{s} {p} {o}"


def make_transformer_mean_pool_encoder(
    model_name_or_path: str,
    device: str = "cpu",
    batch_size: int = 256,
) -> Callable[[List[str]], np.ndarray]:
    """Return a callable that encodes strings using Transformer mean pooling."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for the verbalized-triple baseline. "
            "Install them with:  pip install transformers torch"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).to(device)
    model.eval()
    logger.info("Loaded Transformer encoder '%s' on %s", model_name_or_path, device)

    def encode_fn(sentences: List[str]) -> np.ndarray:
        batches = []
        with torch.no_grad():
            for start in range(0, len(sentences), batch_size):
                batch = sentences[start : start + batch_size]
                tokens = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_special_tokens_mask=True,
                    return_tensors="pt",
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}
                special_tokens_mask = tokens.pop("special_tokens_mask")
                out = model(**tokens)
                token_embs = out.last_hidden_state
                mask = (
                    tokens["attention_mask"] * (1 - special_tokens_mask)
                ).unsqueeze(-1).to(token_embs.dtype)
                summed = (token_embs * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-12)
                batches.append((summed / counts).cpu().numpy())
        return np.concatenate(batches, axis=0).astype(np.float32)

    return encode_fn


def make_text_model_encoder(
    text_model: str,
    device: str = "cpu",
    batch_size: int = 256,
) -> Callable[[List[str]], np.ndarray]:
    """Build the verbalized-triple encoder matching a configured text model."""
    model_name_or_path = TEXT_MODEL_ENCODERS.get(text_model, text_model)
    return make_transformer_mean_pool_encoder(
        model_name_or_path, device=device, batch_size=batch_size
    )


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + 1e-12)


def _cosine_nn(
    queries: np.ndarray,
    keys: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Return for each query the index of the most similar key."""
    queries_n = _l2_normalize(queries)
    keys_n = _l2_normalize(keys)
    indices = np.empty(len(queries), dtype=np.int64)
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        sims = queries_n[start:end] @ keys_n.T
        indices[start:end] = sims.argmax(axis=1)
    return indices


def encode_verbalized_triples(
    data: PreparedData,
    encode_fn: Callable[[List[str]], np.ndarray],
    indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Verbalize and encode triples, returning an (N, D) text-embedding array."""
    triple_df = data.evidence_df if indices is None else data.evidence_df.iloc[indices]
    sentences = [
        verbalize_triple(
            row.subject_uri,
            row.predicate_uri,
            row.object_uri,
            getattr(row, "subject_name", ""),
            getattr(row, "object_name", ""),
        )
        for row in triple_df.itertuples(index=False)
    ]
    logger.info("Encoding %d verbalized triples with language model...", len(sentences))
    return encode_fn(sentences)


def encode_training_triples(
    data: PreparedData,
    encode_fn: Callable[[List[str]], np.ndarray],
    index_split: str = "train",
) -> np.ndarray:
    """Backward-compatible wrapper for callers that need train-only encodings."""
    return encode_verbalized_triples(data, encode_fn, data.split_indices[index_split])


def verbalized_triple_baseline(
    data: PreparedData,
    hits_k: List[int],
    encode_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
    query_split: str = "test",
    index_split: str = "train",
    precomputed_verb_embs: Optional[np.ndarray] = None,
    direction: str = "s2t",
) -> Dict:
    """KG-free retrieval baseline using verbalized triples as text.

    For s2t, each query sentence ranks all verbalized triples.
    For t2s, each query verbalized triple ranks all sentence embeddings.
    """
    del index_split  # kept for API compatibility
    if precomputed_verb_embs is None and encode_fn is None:
        raise ValueError("Either encode_fn or precomputed_verb_embs must be provided.")

    query_idx = data.split_indices[query_split]
    query_text = data.text_embs[query_idx]

    if precomputed_verb_embs is not None:
        verb_embs = precomputed_verb_embs
        logger.info(
            "Verbalized-Triple-NN: using precomputed embeddings (%d, %d)",
            verb_embs.shape[0], verb_embs.shape[1],
        )
    else:
        verb_embs = encode_verbalized_triples(data, encode_fn)  # type: ignore[arg-type]

    query_verb = verb_embs[query_idx]
    predicates = data.evidence_df["predicate_uri"].iloc[query_idx].tolist()
    metrics = compute_metrics(
        query_text,
        query_verb,
        hits_k,
        predicates=predicates,
        direction=direction,
        candidate_pred=data.text_embs if direction == "t2s" else None,
        candidate_truth=verb_embs if direction == "s2t" else None,
        true_indices=query_idx,
    )
    logger.info(
        "Verbalized-Triple-NN  | MRR=%.4f  Hits@1=%.4f  Hits@10=%.4f",
        metrics["mrr"],
        metrics.get("hits@1", float("nan")),
        metrics.get("hits@10", float("nan")),
    )
    return metrics


def run_all_baselines(
    data: PreparedData,
    hits_k: List[int],
    encode_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
    query_split: str = "test",
    index_split: str = "train",
    precomputed_verb_embs: Optional[np.ndarray] = None,
    direction: str = "s2t",
) -> Dict[str, Dict]:
    """Run available baselines and return a dict keyed by baseline name."""
    results: Dict[str, Dict] = {}
    if precomputed_verb_embs is not None or encode_fn is not None:
        results["verbalized_nn"] = verbalized_triple_baseline(
            data, hits_k, encode_fn=encode_fn,
            query_split=query_split, index_split=index_split,
            precomputed_verb_embs=precomputed_verb_embs,
            direction=direction,
        )
    else:
        logger.warning("encode_fn not provided; skipping verbalized-triple baseline.")
    return results
