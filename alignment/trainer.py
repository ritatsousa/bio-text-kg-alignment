"""Training loop with checkpointing and early stopping."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import DefaultConfig
from data import AlignmentDataset, PreparedData
from evaluation import evaluate_alignment_model
from losses import infonce_with_hard_negs
from model import build_alignment_model
from negatives import build_sampler_from_train, compute_sample_weights, make_collate_fn
from qualitative import save_model_top_predictions
from utils import get_device, get_logger, set_seed

logger = get_logger(__name__)


@dataclass
class TrainState:
    epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_mrr: float = -float("inf")
    best_epoch: int = -1
    epochs_without_improvement: int = 0
    val_loss_overfit_streak: int = 0
    history: Dict[str, list] = field(default_factory=dict)

    def record(self, key: str, value: float) -> None:
        self.history.setdefault(key, []).append(value)


VAL_MRR_MIN_IMPROVEMENT = 1e-3
VAL_LOSS_OVERFIT_FACTOR = 1.05
OVERFIT_GUARD_PATIENCE = 5


def _model_checkpointing_enabled() -> bool:
    value = os.environ.get("SAVE_MODEL_CHECKPOINTS", "0").strip().lower()
    return value in {"1", "true", "yes", "y"}


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: TrainState,
    best_model_state: Optional[Dict[str, torch.Tensor]],
    cfg: DefaultConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": state.epoch,
        "best_val_loss": state.best_val_loss,
        "best_val_mrr": state.best_val_mrr,
        "best_epoch": state.best_epoch,
        "epochs_without_improvement": state.epochs_without_improvement,
        "val_loss_overfit_streak": state.val_loss_overfit_streak,
        "history": state.history,
        "best_model_state": best_model_state,
        "config": cfg.model_dump(),
    }
    torch.save(payload, path)


def _load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> TrainState:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    return TrainState(
        epoch=payload["epoch"],
        best_val_loss=payload["best_val_loss"],
        best_val_mrr=payload.get("best_val_mrr", -float("inf")),
        best_epoch=payload.get("best_epoch", -1),
        epochs_without_improvement=payload.get("epochs_without_improvement", 0),
        val_loss_overfit_streak=payload.get("val_loss_overfit_streak", 0),
        history=payload.get("history", {}),
    )


def _val_loss_eval(
    model: torch.nn.Module,
    val_text: torch.Tensor,
    val_triple: torch.Tensor,
    training_direction: str,
    temperature: float,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    losses = []
    n = val_text.size(0)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = val_text[i : i + batch_size].to(device)
            yb = val_triple[i : i + batch_size].to(device)
            if xb.size(0) < 2:
                continue
            if training_direction == "text_to_kg":
                pred = model(xb)
                target = yb
            elif training_direction == "kg_to_text":
                pred = model(yb)
                target = xb
            else:
                pred_t2k = model.project_text_to_kg(xb)
                pred_k2t = model.project_kg_to_text(yb)
                loss_t2k = infonce_with_hard_negs(
                    pred_t2k,
                    yb,
                    torch.zeros((xb.size(0), 0, yb.size(-1)), device=device),
                    temperature=temperature,
                )
                loss_k2t = infonce_with_hard_negs(
                    pred_k2t,
                    xb,
                    torch.zeros((xb.size(0), 0, xb.size(-1)), device=device),
                    temperature=temperature,
                )
                losses.append(float(((loss_t2k + loss_k2t) / 2).item()) * xb.size(0))
                continue
            loss = infonce_with_hard_negs(
                pred,
                target,
                torch.zeros((xb.size(0), 0, target.size(-1)), device=device),
                temperature=temperature,
            )
            losses.append(loss.item() * xb.size(0))
    if not losses:
        return float("inf")
    return float(sum(losses) / n)


def _val_cosine_eval(
    model: torch.nn.Module,
    val_text: torch.Tensor,
    val_triple: torch.Tensor,
    training_direction: str,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        text = val_text.to(device)
        triple = val_triple.to(device)
        if training_direction == "text_to_kg":
            pred = model(text)
            p = F.normalize(pred, dim=-1)
            t = F.normalize(triple, dim=-1)
            return float((p * t).sum(dim=-1).mean().item())
        if training_direction == "kg_to_text":
            pred = model(triple)
            p = F.normalize(pred, dim=-1)
            t = F.normalize(text, dim=-1)
            return float((p * t).sum(dim=-1).mean().item())
        pred_t2k = model.project_text_to_kg(text)
        pred_k2t = model.project_kg_to_text(triple)
        cos_t2k = (F.normalize(pred_t2k, dim=-1) * F.normalize(triple, dim=-1)).sum(dim=-1)
        cos_k2t = (F.normalize(pred_k2t, dim=-1) * F.normalize(text, dim=-1)).sum(dim=-1)
        return float(((cos_t2k.mean() + cos_k2t.mean()) / 2).item())


def _objective_from_val_metrics(training_direction: str, metrics_by_direction: Dict[str, Dict]) -> float:
    if training_direction == "text_to_kg":
        return float(metrics_by_direction["s2t"]["mrr"])
    if training_direction == "kg_to_text":
        return float(metrics_by_direction["t2s"]["mrr"])
    if training_direction == "bidirectional_random":
        return float((metrics_by_direction["s2t"]["mrr"] + metrics_by_direction["t2s"]["mrr"]) / 2.0)
    raise ValueError(
        "training_direction must be one of 'text_to_kg', 'kg_to_text', "
        "or 'bidirectional_random'"
    )


def train(
    cfg: DefaultConfig,
    data: PreparedData,
    checkpoint_dir: Path,
    resume: bool = True,
    log_fn=None,
    mlflow_log_step_fn=None,
) -> Dict[str, Any]:
    """Train one model end-to-end and evaluate on the test set.

    Returns dict with history, test_metrics, best_epoch, best_model_path, etc.
    """
    set_seed(cfg.seed)
    device = get_device()
    logger.info("Using device: %s", device)

    train_idx = data.split_indices["train"]
    val_idx = data.split_indices["val"]
    test_idx = data.split_indices["test"]

    train_ds = AlignmentDataset(data.text_embs, data.triple_embs, train_idx)
    val_ds = AlignmentDataset(data.text_embs, data.triple_embs, val_idx)

    sampler = build_sampler_from_train(
        data.evidence_df, train_idx, data.entity_lookup, data.relation_lookup,
        seed=cfg.seed, cluster_aware=cfg.loss.cluster_aware_negs,
        combination=cfg.triple_combination,
        neg_strategies=cfg.loss.neg_strategies,
    )

    loss_weights_np = None
    if cfg.loss.sample_reweighting != "none":
        loss_weights_np = compute_sample_weights(
            data.evidence_df, train_idx, scheme=cfg.loss.sample_reweighting
        )
    collate = make_collate_fn(
        sampler, data.evidence_df, cfg.loss.n_hard_negs, sample_weights=loss_weights_np,
    )

    if cfg.loss.class_balance:
        from torch.utils.data import WeightedRandomSampler

        balance_w_np = compute_sample_weights(data.evidence_df, train_idx, scheme="inv_freq")
        per_train_w = balance_w_np[train_idx].astype("float64")
        torch_sampler = WeightedRandomSampler(
            weights=per_train_w.tolist(), num_samples=len(train_idx), replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=cfg.training.batch_size, sampler=torch_sampler,
            collate_fn=collate, num_workers=0, drop_last=True,
        )
        logger.info(
            "class_balance=ON: WeightedRandomSampler (min=%.3g, max=%.3g) on %d rows",
            float(per_train_w.min()), float(per_train_w.max()), len(per_train_w),
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.training.batch_size, shuffle=True,
            collate_fn=collate, num_workers=0, drop_last=True,
        )

    val_text_t = val_ds.text
    val_triple_t = val_ds.triple

    model = build_alignment_model(
        training_direction=cfg.training_direction,
        architecture_type=cfg.model.architecture_type,
        text_dim=data.text_dim,
        triple_dim=data.triple_dim,
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
    ).to(device)
    logger.info(
        "Model: training_direction=%s type=%s text_dim=%d triple_dim=%d hidden=%s params=%d",
        cfg.training_direction,
        cfg.model.architecture_type,
        data.text_dim,
        data.triple_dim,
        cfg.model.hidden_dims,
        model.count_parameters(),
    )

    optimizer = AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.training.lr_plateau_factor,
        patience=cfg.training.lr_plateau_patience,
    )

    state = TrainState()
    best_model_state: Optional[Dict[str, torch.Tensor]] = None

    save_model_checkpoints = _model_checkpointing_enabled()
    if not save_model_checkpoints:
        logger.info("Model .pt checkpoint saving is disabled. Set SAVE_MODEL_CHECKPOINTS=1 to enable it.")

    last_ckpt = checkpoint_dir / "last.pt"
    if resume and save_model_checkpoints and last_ckpt.exists():
        try:
            state = _load_checkpoint(last_ckpt, model, optimizer, scheduler)
            payload = torch.load(last_ckpt, map_location="cpu", weights_only=False)
            best_model_state = payload.get("best_model_state", None)
            logger.info("Resumed from %s at epoch %d", last_ckpt, state.epoch)
        except Exception as e:
            logger.warning("Failed to resume from %s: %s; starting fresh", last_ckpt, e)

    start_epoch = state.epoch + 1 if state.epoch > 0 else 1
    end_epoch = cfg.training.epochs

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            if len(batch) == 4:
                text_b, triple_b, hard_b, w_b = batch
                w_b = w_b.to(device)
            else:
                text_b, triple_b, hard_b = batch
                w_b = None
            text_b = text_b.to(device)
            triple_b = triple_b.to(device)
            hard_b = hard_b.to(device)
            train_text_to_kg = cfg.training_direction == "text_to_kg" or (
                cfg.training_direction == "bidirectional_random"
                and torch.rand((), device=device).item() < 0.5
            )
            if train_text_to_kg:
                pred = (
                    model.project_text_to_kg(text_b)
                    if cfg.training_direction == "bidirectional_random"
                    else model(text_b)
                )
                target = triple_b
                hard_for_loss = hard_b
            else:
                pred = (
                    model.project_kg_to_text(triple_b)
                    if cfg.training_direction == "bidirectional_random"
                    else model(triple_b)
                )
                target = text_b
                hard_for_loss = torch.zeros(
                    (text_b.size(0), 0, text_b.size(-1)), device=device
                )
            loss = infonce_with_hard_negs(
                pred, target, hard_for_loss, temperature=cfg.loss.temperature,
                sample_weights=w_b,
            )
            optimizer.zero_grad()
            loss.backward()
            if cfg.training.grad_clip and cfg.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()
            running += loss.item()
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        val_loss = _val_loss_eval(
            model,
            val_text_t,
            val_triple_t,
            cfg.training_direction,
            cfg.loss.temperature,
            device,
            cfg.training.batch_size,
        )
        val_cosine = _val_cosine_eval(
            model, val_text_t, val_triple_t, cfg.training_direction, device
        )
        scheduler.step(val_loss)

        state.epoch = epoch
        state.record("train_loss", train_loss)
        state.record("val_loss", val_loss)
        state.record("val_cosine", val_cosine)

        val_metrics_by_direction = {
            direction: evaluate_alignment_model(
                model,
                data.text_embs,
                data.triple_embs,
                val_idx,
                device,
                cfg.eval.hits_k,
                training_direction=cfg.training_direction,
                eval_direction=direction,
            )
            for direction in ("s2t", "t2s")
        }
        val_mrr = _objective_from_val_metrics(cfg.training_direction, val_metrics_by_direction)
        state.record("val_objective_mrr", val_mrr)
        state.record("val_mrr_s2t", val_metrics_by_direction["s2t"]["mrr"])
        state.record("val_mrr_t2s", val_metrics_by_direction["t2s"]["mrr"])

        logger.info(
            "Epoch %d: train_loss=%.4f val_loss=%.4f val_cos=%.4f val_obj_mrr=%.4f val_mrr_s2t=%.4f val_mrr_t2s=%.4f",
            epoch,
            train_loss,
            val_loss,
            val_cosine,
            val_mrr,
            val_metrics_by_direction["s2t"]["mrr"],
            val_metrics_by_direction["t2s"]["mrr"],
        )

        if log_fn is not None:
            log_fn({"train_loss": train_loss, "val_loss": val_loss,
                    "val_cosine": val_cosine, "val_objective_mrr": val_mrr,
                    "val_mrr_s2t": val_metrics_by_direction["s2t"]["mrr"],
                    "val_mrr_t2s": val_metrics_by_direction["t2s"]["mrr"]}, epoch)
        if mlflow_log_step_fn is not None:
            mlflow_log_step_fn({"train_loss": train_loss, "val_loss": val_loss,
                                "val_cosine": val_cosine, "val_objective_mrr": val_mrr,
                                "val_mrr_s2t": val_metrics_by_direction["s2t"]["mrr"],
                                "val_mrr_t2s": val_metrics_by_direction["t2s"]["mrr"]}, epoch)

        if val_loss < state.best_val_loss - 1e-6:
            state.best_val_loss = val_loss

        improved = val_mrr > state.best_val_mrr + VAL_MRR_MIN_IMPROVEMENT
        if improved:
            state.best_val_mrr = val_mrr
            state.best_epoch = epoch
            state.epochs_without_improvement = 0
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            state.epochs_without_improvement += 1

        if val_loss > VAL_LOSS_OVERFIT_FACTOR * state.best_val_loss:
            state.val_loss_overfit_streak += 1
        else:
            state.val_loss_overfit_streak = 0

        if save_model_checkpoints:
            _save_checkpoint(last_ckpt, model, optimizer, scheduler, state, best_model_state, cfg)

        if state.val_loss_overfit_streak >= OVERFIT_GUARD_PATIENCE:
            logger.warning(
                "Overfitting guard triggered at epoch %d: val_loss=%.4f > %.2fx best=%.4f "
                "for %d consecutive epochs. Stopping.",
                epoch, val_loss, VAL_LOSS_OVERFIT_FACTOR, state.best_val_loss,
                state.val_loss_overfit_streak,
            )
            break

        if state.epochs_without_improvement >= cfg.training.early_stop_patience:
            logger.info(
                "Early stopping at epoch %d (best epoch %d, val_mrr=%.4f)",
                epoch, state.best_epoch, state.best_val_mrr,
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if save_model_checkpoints:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save({"model_state": best_model_state, "config": cfg.model_dump()}, best_path)

    test_preds = data.evidence_df.iloc[test_idx]["predicate_uri"].tolist()
    test_metrics_by_direction = {
        direction: evaluate_alignment_model(
            model,
            data.text_embs,
            data.triple_embs,
            test_idx,
            device,
            cfg.eval.hits_k,
            predicates=test_preds,
            training_direction=cfg.training_direction,
            eval_direction=direction,
        )
        for direction in ("s2t", "t2s")
    }
    test_metrics = test_metrics_by_direction[cfg.eval.eval_direction]
    logger.info(
        "Test objective direction=%s: cosine=%.4f mrr=%.4f hits@1=%.4f hits@10=%.4f median_rank=%.1f",
        cfg.eval.eval_direction,
        test_metrics["cosine_mean"],
        test_metrics["mrr"],
        test_metrics["hits@1"],
        test_metrics["hits@10"],
        test_metrics["median_rank"],
    )

    eval_json = checkpoint_dir / "test_metrics.json"
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(test_metrics_by_direction, f, indent=2)

    top_prediction_paths: Dict[str, str] = {}
    for direction in ("s2t", "t2s"):
        top_predictions_path = checkpoint_dir / f"top_predictions_{direction}.csv"
        save_model_top_predictions(
            model,
            data,
            direction,
            top_predictions_path,
            device,
            training_direction=cfg.training_direction,
            n_queries=10,
            top_k=10,
            seed=cfg.seed,
        )
        top_prediction_paths[direction] = str(top_predictions_path)

    return {
        "history": state.history,
        "test_metrics": test_metrics,
        "test_metrics_by_direction": test_metrics_by_direction,
        "best_epoch": state.best_epoch,
        "best_val_loss": state.best_val_loss,
        "best_val_mrr": state.best_val_mrr,
        "best_model_path": str(checkpoint_dir / "best_model.pt"),
        "eval_json_path": str(eval_json),
        "top_prediction_paths": top_prediction_paths,
    }
