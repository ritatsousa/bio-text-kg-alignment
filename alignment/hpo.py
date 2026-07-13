"""Optuna-based hyperparameter optimization."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Dict, List, Optional

import mlflow
import numpy as np
import optuna

from baselines import encode_verbalized_triples, run_all_baselines
from config import DefaultConfig, HPOConfig
from data import COMBINATION_STRATEGIES, PreparedData, prepare_data
from evaluation import average_fold_metrics, per_predicate_dataframe
from negatives import STRATEGIES as NEG_STRATEGIES
from tracking import init_mlflow, log_artifact, log_metrics, log_params
from trainer import train
from utils import get_logger

logger = get_logger(__name__)


def combo_name(text_model: str, kg_family: str, kg_config: str) -> str:
    return f"{text_model}__{kg_family}_{kg_config}"


def _suggest_from_space(trial: optuna.Trial, hpo: HPOConfig) -> Dict:
    ss = hpo.search_space
    arch_idx = trial.suggest_categorical(
        "architecture_idx", list(range(len(ss.architecture.choices)))
    )
    architecture = ss.architecture.choices[arch_idx]
    dropout = trial.suggest_float("dropout", ss.dropout.low, ss.dropout.high)
    lr = trial.suggest_float("learning_rate", ss.learning_rate.low, ss.learning_rate.high, log=True)
    wd = trial.suggest_float("weight_decay", ss.weight_decay.low, ss.weight_decay.high, log=True)
    n_hard = trial.suggest_categorical("n_hard_negs", ss.n_hard_negs.choices)
    triple_comb = trial.suggest_categorical("triple_combination", ss.triple_combination.choices)
    neg_strat = trial.suggest_categorical("neg_strategies", ss.neg_strategies.choices)
    arch_type = trial.suggest_categorical("architecture_type", ss.architecture_type.choices)
    return {
        "architecture": list(architecture),
        "dropout": float(dropout),
        "learning_rate": float(lr),
        "weight_decay": float(wd),
        "n_hard_negs": int(n_hard),
        "triple_combination": str(triple_comb),
        "neg_strategies": str(neg_strat),
        "architecture_type": str(arch_type),
    }


def build_trial_config(base_cfg: DefaultConfig, hpo: HPOConfig, params: Dict) -> DefaultConfig:
    cfg = copy.deepcopy(base_cfg)
    cfg.model.hidden_dims = params["architecture"]
    cfg.model.dropout = params["dropout"]
    cfg.optimizer.lr = params["learning_rate"]
    cfg.optimizer.weight_decay = params["weight_decay"]
    cfg.triple_combination = params["triple_combination"]
    cfg.model.architecture_type = params["architecture_type"]
    # "all" maps to the full strategy list; "none" disables hard negatives.
    chosen = params["neg_strategies"]
    if chosen == "all":
        cfg.loss.neg_strategies = list(NEG_STRATEGIES)
        cfg.loss.n_hard_negs = params["n_hard_negs"]
    elif chosen == "none":
        cfg.loss.neg_strategies = []
        cfg.loss.n_hard_negs = 0
    else:
        cfg.loss.neg_strategies = [chosen]
        cfg.loss.n_hard_negs = params["n_hard_negs"]
    cfg.training.batch_size = hpo.fixed.batch_size
    cfg.loss.temperature = hpo.fixed.temperature
    cfg.training.epochs = hpo.fixed.epochs
    cfg.training.early_stop_patience = hpo.fixed.early_stop_patience
    if cfg.training_direction == "kg_to_text":
        cfg.eval.eval_direction = "t2s"
    else:
        cfg.eval.eval_direction = "s2t"
    return cfg


def _log_baselines_to_mlflow(
    name: str,
    text_model: str,
    folds: List[PreparedData],
    hits_k: List[int],
    eval_direction: str,
    encode_fn: Optional[Callable],
) -> None:
    """Run verbalized baseline across all folds, average, and log to MLflow.

    Skips if a run named 'baselines' already exists for the experiment.
    """
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        existing = client.search_runs(
            [exp.experiment_id],
            filter_string="tags.mlflow.runName = 'baselines'",
        )
        if existing:
            logger.info("Baselines already logged for '%s'; skipping.", name)
            return

    logger.info(
        "Running baselines for '%s' over %d folds × %d strategies...",
        name, len(folds), len(COMBINATION_STRATEGIES),
    )
    # Keys: "{baseline_name}_{strategy}" -> list of per-fold metrics dicts
    fold_results: Dict[str, List[Dict]] = {}
    for fold_i, fold_data in enumerate(folds):
        logger.info("  Baseline fold %d/%d", fold_i + 1, len(folds))
        precomputed_verb_embs: Optional[np.ndarray] = None
        if encode_fn is not None:
            precomputed_verb_embs = encode_verbalized_triples(fold_data, encode_fn)
        for strategy in COMBINATION_STRATEGIES:
            results = run_all_baselines(
                fold_data.with_strategy(strategy),
                hits_k=hits_k,
                encode_fn=None,
                precomputed_verb_embs=precomputed_verb_embs,
                direction=eval_direction,
            )
            for baseline_name, metrics in results.items():
                key = f"{baseline_name}_{strategy}"
                fold_results.setdefault(key, []).append(metrics)

    mlflow.set_experiment(name)
    with mlflow.start_run(run_name="baselines"):
        log_params({"baseline": "all", "combo": name, "n_folds": len(folds)})
        for key, per_fold in fold_results.items():
            avg = average_fold_metrics(per_fold)
            flat = {
                f"{key}_{k}": v
                for k, v in avg.items()
                if isinstance(v, (int, float))
            }
            log_metrics(flat)
    logger.info("Baselines logged for '%s'.", name)


def run_hpo_for_combination(
    text_model: str,
    kg_family: str,
    kg_config: str,
    base_cfg: DefaultConfig,
    hpo: HPOConfig,
    n_trials: int,
    mlflow_enabled: bool = True,
    folds: Optional[List[PreparedData]] = None,
    encode_fn: Optional[Callable] = None,
    study_name: Optional[str] = None,
    checkpoint_base_dir: Optional[Path] = None,
    log_baselines: bool = True,
) -> optuna.Study:
    """Run an Optuna study for a single (text_model, kg_family, kg_config) combination.

    Each Optuna trial trains one model per CV fold and optimises the fold-averaged
    metric defined by ``hpo.optimization_metric``.
    """
    name = study_name or combo_name(text_model, kg_family, kg_config)
    logger.info("=== HPO combo: %s (n_trials=%d, n_folds=%d) ===", name, n_trials, len(folds) if folds else "?")

    base = copy.deepcopy(base_cfg)
    base.text_model = text_model
    base.kg_family = kg_family
    base.kg_config = kg_config
    if base.paths is None:
        raise ValueError("base_cfg.paths must be set before running HPO")
    if folds is None:
        folds = prepare_data(base)

    if mlflow_enabled:
        init_mlflow(base.paths.mlruns_dir)

    if mlflow_enabled and log_baselines:
        _log_baselines_to_mlflow(
            name, text_model, folds, hits_k=base.eval.hits_k,
            eval_direction=base.eval.eval_direction, encode_fn=encode_fn
        )

    storage = hpo.optuna.storage
    sampler = optuna.samplers.TPESampler(seed=base.seed)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        direction=hpo.direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    if mlflow_enabled:
        mlflow.set_experiment(name)

    completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    logger.info("Study '%s': %d completed trials, target %d", name, completed, n_trials)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_from_space(trial, hpo)
        cfg = build_trial_config(base, hpo, params)
        ckpt_root = checkpoint_base_dir or (Path(cfg.paths.checkpoints_dir) / name)
        ckpt_dir = ckpt_root / f"trial_{trial.number}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if mlflow_enabled:
            mlflow.set_experiment(name)
            run_ctx = mlflow.start_run(run_name=f"trial_{trial.number}")
        else:
            run_ctx = None

        try:
            if mlflow_enabled:
                log_params({
                    "arch": params["architecture"],
                    "dropout": params["dropout"],
                    "lr": params["learning_rate"],
                    "weight_decay": params["weight_decay"],
                    "n_hard_negs": params["n_hard_negs"],
                    "batch_size": cfg.training.batch_size,
                    "temperature": cfg.loss.temperature,
                    "text_model": text_model,
                    "kg_family": kg_family,
                    "kg_config": kg_config,
                    "training_direction": cfg.training_direction,
                    "n_folds": len(folds),
                    "architecture_type": params["architecture_type"],
                    "triple_combination": params["triple_combination"],
                    "neg_strategies": params["neg_strategies"],
                })

            def _mlflow_step(m: Dict, step: int):
                if mlflow_enabled:
                    log_metrics(m, step=step)

            # Train one model per fold and collect test metrics.
            fold_test_metrics_by_direction: Dict[str, List[Dict]] = {"s2t": [], "t2s": []}
            fold_objective_values: List[float] = []
            for fold_i, fold_data in enumerate(folds):
                fold_ckpt_dir = ckpt_dir / f"fold_{fold_i}"
                fold_ckpt_dir.mkdir(parents=True, exist_ok=True)
                result = train(
                    cfg,
                    fold_data.with_strategy(params["triple_combination"]),
                    checkpoint_dir=fold_ckpt_dir, resume=True,
                    mlflow_log_step_fn=_mlflow_step if mlflow_enabled else None,
                )
                fold_objective_values.append(float(result["best_val_mrr"]))
                for eval_direction, metrics in result["test_metrics_by_direction"].items():
                    fold_test_metrics_by_direction.setdefault(eval_direction, []).append(metrics)

                if mlflow_enabled:
                    fold_flat = {}
                    for eval_direction, metrics in result["test_metrics_by_direction"].items():
                        fold_flat.update({
                            f"fold{fold_i}_{eval_direction}_test_{k}": v
                            for k, v in metrics.items()
                            if isinstance(v, (int, float))
                        })
                    log_metrics(fold_flat)
                    for path in result.get("top_prediction_paths", {}).values():
                        log_artifact(Path(path))

            avg_metrics_by_direction = {
                eval_direction: average_fold_metrics(metrics)
                for eval_direction, metrics in fold_test_metrics_by_direction.items()
            }
            objective_value = float(np.mean(fold_objective_values))
            trial.set_user_attr("avg_metrics_by_direction", avg_metrics_by_direction)
            trial.set_user_attr("objective_metric", "best_validation_objective_mrr")
            trial.set_user_attr("params", params)

            if mlflow_enabled:
                flat_avg = {}
                for eval_direction, avg_metrics in avg_metrics_by_direction.items():
                    flat_avg.update({
                        f"avg_{eval_direction}_test_{k}": v
                        for k, v in avg_metrics.items()
                        if isinstance(v, (int, float))
                    })
                flat_avg["avg_validation_objective_mrr"] = objective_value
                log_metrics(flat_avg)
                for eval_direction, avg_metrics in avg_metrics_by_direction.items():
                    pp_df = per_predicate_dataframe(avg_metrics, direction=eval_direction)
                    if not pp_df.empty:
                        pp_path = ckpt_dir / f"per_predicate_avg_{eval_direction}.csv"
                        pp_df.to_csv(pp_path, index=False)
                        log_artifact(pp_path)

            return objective_value
        finally:
            if run_ctx is not None:
                mlflow.end_run()

    remaining = max(0, n_trials - completed)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    else:
        logger.info("Study '%s' already has %d trials; skipping", name, completed)

    # Tag best trial in MLflow
    if mlflow_enabled and len(study.trials) > 0:
        try:
            best = study.best_trial
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name(name)
            if exp is not None:
                runs = client.search_runs([exp.experiment_id], order_by=["start_time"])
                target_run_name = f"trial_{best.number}"
                for r in runs:
                    if r.data.tags.get("mlflow.runName") == target_run_name:
                        client.set_tag(r.info.run_id, "best_in_combo", "true")
                    elif r.data.tags.get("best_in_combo") == "true":
                        client.set_tag(r.info.run_id, "best_in_combo", "false")
        except Exception as e:
            logger.warning("Could not set best_in_combo tag: %s", e)

    return study
