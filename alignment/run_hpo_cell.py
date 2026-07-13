"""Run one HPO experimental cell.

The cell is selected by ``--cell-id`` or, on SLURM, ``SLURM_ARRAY_TASK_ID``.
Each cell gets its own Optuna SQLite database and one result CSV, avoiding
SQLite write contention between array jobs.
"""
from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import optuna
import pandas as pd

THIS = Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))

from config import (  # noqa: E402
    DefaultConfig,
    EvalConfig,
    HPOConfig,
    HPOFixed,
    HPOParam,
    HPOSearchSpace,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    OptunaConfig,
    PathConfig,
    SplitConfig,
    TrainingConfig,
)
from data import prepare_data  # noqa: E402
from hpo import combo_name, run_hpo_for_combination  # noqa: E402
from utils import get_logger  # noqa: E402

logger = get_logger("run_hpo_cell")


def _fixed_cell_hpo(
    template: HPOConfig,
    architecture_type: str,
    triple_combination: str,
    neg_strategy: str,
) -> HPOConfig:
    cfg = copy.deepcopy(template)
    cfg.search_space.architecture_type.choices = [architecture_type]
    cfg.search_space.triple_combination.choices = [triple_combination]
    cfg.search_space.neg_strategies.choices = [neg_strategy]
    return cfg


def _flatten_row(prefix: Dict, metrics: Dict) -> Dict:
    row = dict(prefix)
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            row[k] = v
    return row


def build_cells() -> List[Dict[str, str]]:
    cells: List[Dict[str, str]] = []
    for training_direction in TRAINING_DIRECTIONS:
        for combo in TEXT_KG_COMBINATIONS:
            for architecture_type in ARCHITECTURE_TYPES:
                for triple_combination in COMBINATION_STRATEGIES:
                    for neg_strategy in NEG_STRATEGIES:
                        cells.append(
                            {
                                "training_direction": training_direction,
                                "text_model": combo["text_model"],
                                "kg_family": combo["kg_family"],
                                "kg_config": combo["kg_config"],
                                "architecture_type": architecture_type,
                                "triple_combination": triple_combination,
                                "neg_strategy": neg_strategy,
                            }
                        )
    return cells


def make_paths(results_root: Path, study_name: str) -> PathConfig:
    data_root = THIS / "Data"
    return PathConfig(
        evidence_path=str(
            data_root
            / "CTD_to_PubTator"
            / "kg_data"
            / "evidence_aligned_ided_medcpt_dedup.tsv"
        ),
        text_emb_dir=str(data_root / "text_embeddings"),
        kg_emb_dir=str(data_root / "kg_embeddings" / "outputs" / "selected"),
        checkpoints_dir=str(results_root / "checkpoints"),
        mlruns_dir=str(results_root / "mlruns"),
        optuna_storage=f"sqlite:///{(results_root / 'optuna_cells' / f'{study_name}.db').as_posix()}",
    )


def make_base_config(paths: PathConfig, cell: Dict[str, str]) -> DefaultConfig:
    return DefaultConfig(
        seed=42,
        text_model=cell["text_model"],
        kg_family=cell["kg_family"],
        kg_config=cell["kg_config"],
        training_direction=cell["training_direction"],
        paths=paths,
        split=SplitConfig(
            n_folds=5,
            val_fraction=0.15,
            group_by="pmid",
            stratify_by="predicate",
        ),
        training=TrainingConfig(
            batch_size=256,
            epochs=200,
            early_stop_patience=15,
            lr_plateau_patience=5,
            lr_plateau_factor=0.5,
            grad_clip=1.0,
        ),
        model=ModelConfig(hidden_dims=[768, 512], dropout=0.3),
        loss=LossConfig(temperature=0.07, n_hard_negs=4),
        optimizer=OptimizerConfig(name="adamw", lr=1.0e-3, weight_decay=1.0e-4),
        eval=EvalConfig(hits_k=[1, 3, 5, 10, 25], eval_direction="s2t"),
    )


def make_hpo_template(paths: PathConfig) -> HPOConfig:
    return HPOConfig(
        n_trials_per_combination=3,
        search_space=HPOSearchSpace(
            architecture=HPOParam(type="categorical", choices=[[256], [512]]),
            architecture_type=HPOParam(type="categorical", choices=ARCHITECTURE_TYPES),
            triple_combination=HPOParam(type="categorical", choices=COMBINATION_STRATEGIES),
            neg_strategies=HPOParam(type="categorical", choices=NEG_STRATEGIES),
            dropout=HPOParam(type="uniform", low=0.1, high=0.3),
            learning_rate=HPOParam(type="loguniform", low=1.0e-4, high=5.0e-3),
            weight_decay=HPOParam(type="loguniform", low=1.0e-5, high=1.0e-2),
            n_hard_negs=HPOParam(type="categorical", choices=[4]),
        ),
        fixed=HPOFixed(
            batch_size=256,
            temperature=0.07,
            optimizer="adamw",
            epochs=100,
            early_stop_patience=15,
        ),
        optuna=OptunaConfig(
            storage=paths.optuna_storage,
            sampler="tpe",
            pruner="median",
        ),
        optimization_metric="mrr",
        direction="maximize",
    )


def cell_study_name(cell: Dict[str, str]) -> str:
    combo_id = combo_name(cell["text_model"], cell["kg_family"], cell["kg_config"])
    return (
        f"{cell['training_direction']}__{combo_id}__{cell['architecture_type']}"
        f"__{cell['triple_combination']}__{cell['neg_strategy']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one parallel HPO cell.")
    parser.add_argument(
        "--cell-id",
        type=int,
        default=None,
        help="Cell index. Defaults to SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=THIS / "Results",
        help="Root directory for outputs. Default: ./Results",
    )
    parser.add_argument(
        "--list-cells",
        action="store_true",
        help="Print the grid and exit.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging for this cell.",
    )
    return parser.parse_args()


def resolve_cell_id(args: argparse.Namespace, n_cells: int) -> int:
    if args.cell_id is not None:
        cell_id = args.cell_id
    else:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise ValueError("Provide --cell-id or run as a SLURM array task.")
        cell_id = int(raw)
    if cell_id < 0 or cell_id >= n_cells:
        raise ValueError(f"cell_id must be in [0, {n_cells - 1}], got {cell_id}")
    return cell_id


def result_rows_from_best_trial(cell: Dict[str, str], study_name: str, best) -> List[Dict]:
    avg_metrics = best.user_attrs.get("avg_metrics", {})
    avg_metrics_by_direction = best.user_attrs.get("avg_metrics_by_direction", {})
    if not avg_metrics_by_direction and avg_metrics:
        avg_metrics_by_direction = {"s2t": avg_metrics}
    if not avg_metrics_by_direction:
        raise RuntimeError(
            f"Best trial for {study_name} has no avg_metrics_by_direction. "
            "Old studies created before the bidirectional evaluation change cannot be aggregated."
        )

    rows: List[Dict] = []
    for eval_direction in EVAL_DIRECTIONS:
        if eval_direction not in avg_metrics_by_direction:
            continue
        rows.append(
            _flatten_row(
                {
                    "training_direction": cell["training_direction"],
                    "eval_direction": eval_direction,
                    "result_type": "hpo",
                    "baseline": "",
                    "text_model": cell["text_model"],
                    "kg_family": cell["kg_family"],
                    "kg_config": cell["kg_config"],
                    "architecture_type": cell["architecture_type"],
                    "triple_combination": cell["triple_combination"],
                    "neg_strategy": cell["neg_strategy"],
                    "trial_number": best.number,
                    "study_name": study_name,
                    "best_value": best.value,
                },
                avg_metrics_by_direction[eval_direction],
            )
        )
    return rows


def _sqlite_path(storage: str) -> Path:
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        raise ValueError(f"Only sqlite Optuna storage is supported here, got: {storage}")
    return Path(storage[len(prefix):])


def _assert_inside(parent: Path, child: Path) -> None:
    parent = parent.resolve()
    child = child.resolve()
    try:
        child.relative_to(parent)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to delete path outside {parent}: {child}") from exc


def _fold_outputs_complete(trial_dir: Path, n_folds: int) -> bool:
    for fold_i in range(n_folds):
        fold_dir = trial_dir / f"fold_{fold_i}"
        required = [
            fold_dir / "test_metrics.json",
            fold_dir / "top_predictions_s2t.csv",
            fold_dir / "top_predictions_t2s.csv",
        ]
        if any(not path.exists() for path in required):
            return False
    return True


def _study_complete_with_fold_outputs(
    study_name: str,
    storage: str,
    checkpoint_base_dir: Path,
    n_trials: int,
    n_folds: int,
) -> bool:
    db_path = _sqlite_path(storage)
    if not db_path.exists():
        return False

    study = optuna.load_study(study_name=study_name, storage=storage)
    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if len(complete_trials) < n_trials:
        logger.warning(
            "Existing study %s has only %d/%d complete trials.",
            study_name,
            len(complete_trials),
            n_trials,
        )
        return False

    for trial in complete_trials:
        if "avg_metrics_by_direction" not in trial.user_attrs:
            logger.warning(
                "Existing study %s trial_%d is complete but lacks avg_metrics_by_direction.",
                study_name,
                trial.number,
            )
            return False
        trial_dir = checkpoint_base_dir / f"trial_{trial.number}"
        if not _fold_outputs_complete(trial_dir, n_folds):
            logger.warning(
                "Existing study %s trial_%d is missing one or more fold output files.",
                study_name,
                trial.number,
            )
            return False
    return True


def reset_existing_cell_if_incomplete(
    study_name: str,
    paths: PathConfig,
    checkpoint_base_dir: Path,
    results_root: Path,
    n_trials: int,
    n_folds: int,
) -> None:
    db_path = _sqlite_path(paths.optuna_storage)
    has_existing = db_path.exists() or checkpoint_base_dir.exists()
    if not has_existing:
        return

    try:
        complete = _study_complete_with_fold_outputs(
            study_name,
            paths.optuna_storage,
            checkpoint_base_dir,
            n_trials=n_trials,
            n_folds=n_folds,
        )
    except Exception as exc:
        logger.warning("Could not validate existing study %s: %s", study_name, exc)
        complete = False

    if complete:
        logger.info(
            "Existing study %s has complete Optuna trials and fold outputs; reusing it.",
            study_name,
        )
        return

    logger.warning(
        "Existing study/checkpoints for %s are incomplete or inconsistent; deleting this cell and restarting.",
        study_name,
    )
    _assert_inside(results_root, db_path)
    _assert_inside(results_root, checkpoint_base_dir)
    if db_path.exists():
        db_path.unlink()
    if checkpoint_base_dir.exists():
        shutil.rmtree(checkpoint_base_dir)

    result_csv = results_root / "cell_results" / f"{study_name}.csv"
    _assert_inside(results_root, result_csv)
    if result_csv.exists():
        result_csv.unlink()


def main() -> None:
    args = parse_args()
    cells = build_cells()

    if args.list_cells:
        for i, cell in enumerate(cells):
            print(i, cell_study_name(cell), cell)
        print(f"Total cells: {len(cells)}")
        return

    cell_id = resolve_cell_id(args, len(cells))
    cell = cells[cell_id]
    study_name = cell_study_name(cell)
    results_root = args.results_root.resolve()

    paths = make_paths(results_root, study_name)
    Path(paths.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(paths.mlruns_dir).mkdir(parents=True, exist_ok=True)
    Path(paths.optuna_storage.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)

    base_cfg = make_base_config(paths, cell)
    hpo_template = make_hpo_template(paths)
    cell_hpo = _fixed_cell_hpo(
        hpo_template,
        architecture_type=cell["architecture_type"],
        triple_combination=cell["triple_combination"],
        neg_strategy=cell["neg_strategy"],
    )

    combo_id = combo_name(cell["text_model"], cell["kg_family"], cell["kg_config"])
    checkpoint_base_dir = (
        Path(paths.checkpoints_dir)
        / cell["training_direction"]
        / combo_id
        / cell["architecture_type"]
        / cell["triple_combination"]
        / cell["neg_strategy"]
    )

    reset_existing_cell_if_incomplete(
        study_name=study_name,
        paths=paths,
        checkpoint_base_dir=checkpoint_base_dir,
        results_root=results_root,
        n_trials=cell_hpo.n_trials_per_combination,
        n_folds=base_cfg.split.n_folds,
    )

    logger.info("Running cell %d/%d: %s", cell_id, len(cells) - 1, study_name)
    folds = prepare_data(copy.deepcopy(base_cfg))
    study = run_hpo_for_combination(
        text_model=cell["text_model"],
        kg_family=cell["kg_family"],
        kg_config=cell["kg_config"],
        base_cfg=base_cfg,
        hpo=cell_hpo,
        n_trials=cell_hpo.n_trials_per_combination,
        mlflow_enabled=not args.no_mlflow,
        folds=folds,
        encode_fn=None,
        study_name=study_name,
        checkpoint_base_dir=checkpoint_base_dir,
        log_baselines=False,
    )

    best = study.best_trial
    rows = result_rows_from_best_trial(cell, study_name, best)
    output_dir = results_root / "cell_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{study_name}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logger.info("Cell results written to %s", output_path)



TEXT_KG_COMBINATIONS: List[Dict[str, str]] = [
    {"text_model": "pubmedbert_mcpt", "kg_family": "rdf2vec", "kg_config": "A_best_vec200"},
    {"text_model": "biobert_mcpt", "kg_family": "rdf2vec", "kg_config": "A_best_vec200"},
    {"text_model": "pubmedbert_mcpt", "kg_family": "rotate", "kg_config": "A_best_dim500"},
    {"text_model": "biobert_mcpt", "kg_family": "rotate", "kg_config": "A_best_dim500"},
    {"text_model": "pubmedbert_mcpt", "kg_family": "tucker", "kg_config": "A_best_dim200_rd65"},
    {"text_model": "biobert_mcpt", "kg_family": "tucker", "kg_config": "A_best_dim200_rd65"},
]

# ARCHITECTURE_TYPES = ["mlp", "linear", "residual", "bilinear", "low_rank", "cross_attention"]
ARCHITECTURE_TYPES = ["mlp", "linear", "cross_attention"]
TRAINING_DIRECTIONS = ["text_to_kg", "kg_to_text", "bidirectional_random"]
COMBINATION_STRATEGIES = ["concat", "hadamard", "l1", "l2"]
NEG_STRATEGIES = ["all", "none", "same_s_same_o_vary_p", "same_s_vary_o", "vary_s_same_p_same_o", "same_p_vary_both",]
EVAL_DIRECTIONS = ["s2t", "t2s"]

# To expand the grid later, edit the lists above and update the SLURM array
# range in run_hpo_cell.sh to 0-(len(build_cells()) - 1).


if __name__ == "__main__":
    main()
