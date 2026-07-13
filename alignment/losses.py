"""InfoNCE with optional hard negatives and per-sample reweighting."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def infonce_with_hard_negs(
    text_proj: torch.Tensor,
    pos_kg: torch.Tensor,
    hard_negs: torch.Tensor,
    temperature: float = 0.07,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """InfoNCE contrastive loss with in-batch negatives + optional hard negatives.

    Args:
        text_proj: (B, D) — projected text embeddings.
        pos_kg:   (B, D) — positive KG triple embeddings.
        hard_negs: (B, K, D) — per-anchor hard negatives. K=0 allowed.
        temperature: softmax temperature.
        sample_weights: optional (B,) per-anchor weight tensor for loss
            reweighting. Weights should be pre-normalised to mean ≈ 1.
            If None, the loss is a plain mean over the batch.

    Returns:
        Scalar cross-entropy loss.
    """
    B = text_proj.size(0)
    text_proj = F.normalize(text_proj, dim=-1)
    pos_kg = F.normalize(pos_kg, dim=-1)

    in_batch_logits = (text_proj @ pos_kg.T) / temperature  # (B, B)

    if hard_negs.numel() > 0 and hard_negs.size(1) > 0:
        hard_negs = F.normalize(hard_negs, dim=-1)
        hard_logits = torch.einsum("bd,bkd->bk", text_proj, hard_negs) / temperature
        logits = torch.cat([in_batch_logits, hard_logits], dim=-1)  # (B, B+K)
    else:
        logits = in_batch_logits

    labels = torch.arange(B, device=logits.device)

    if sample_weights is None:
        return F.cross_entropy(logits, labels)

    per_sample = F.cross_entropy(logits, labels, reduction="none")
    return (per_sample * sample_weights).mean()
