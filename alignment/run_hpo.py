"""Run HPO cells sequentially.

This is a simpler alternative to ``run_hpo_cell.py`` for local or single-job
execution. Instead of relying on a SLURM array, it iterates over the selected
cell IDs one after another and writes the same per-cell result CSV files.
"""
import copy
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

THIS = Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))

from data import prepare_data  # noqa: E402
from hpo import combo_name, run_hpo_for_combination  # noqa: E402
from run_hpo_cell import (  # noqa: E402
    _fixed_cell_hpo,
    build_cells,
    cell_study_name,
    make_base_config,
    make_hpo_template,
    make_paths,
    reset_existing_cell_if_incomplete,
    result_rows_from_best_trial,
)
from utils import get_logger  # noqa: E402

logger = get_logger("run_hpo")


def run_cell(
    cell_id: int,
    cell: Dict[str, str],
    results_root: Path,
    mlflow_enabled: bool,
) -> Path:
    study_name = cell_study_name(cell)
    paths = make_paths(results_root, study_name)

    Path(paths.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(paths.mlruns_dir).mkdir(parents=True, exist_ok=True)
    Path(paths.optuna_storage.replace("sqlite:///", "")).parent.mkdir(
        parents=True, exist_ok=True
    )

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

    logger.info("Running cell %d: %s", cell_id, study_name)
    folds = prepare_data(copy.deepcopy(base_cfg))
    study = run_hpo_for_combination(
        text_model=cell["text_model"],
        kg_family=cell["kg_family"],
        kg_config=cell["kg_config"],
        base_cfg=base_cfg,
        hpo=cell_hpo,
        n_trials=cell_hpo.n_trials_per_combination,
        mlflow_enabled=mlflow_enabled,
        folds=folds,
        encode_fn=None,
        study_name=study_name,
        checkpoint_base_dir=checkpoint_base_dir,
        log_baselines=False,
    )

    rows = result_rows_from_best_trial(cell, study_name, study.best_trial)
    output_dir = results_root / "cell_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{study_name}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logger.info("Cell %d results written to %s", cell_id, output_path)
    return output_path


def run_cells(
    cells: List[Dict[str, str]],
    cell_ids: Iterable[int],
    results_root: Path,
    mlflow_enabled: bool,
) -> None:
    selected = list(cell_ids)
    for position, cell_id in enumerate(selected, start=1):
        logger.info("Sequential progress: %d/%d", position, len(selected))
        run_cell(
            cell_id=cell_id,
            cell=cells[cell_id],
            results_root=results_root,
            mlflow_enabled=mlflow_enabled,
        )


if __name__ == "__main__":
    cells = build_cells()
    results_root = (THIS / "Results").resolve()
    cell_ids = list(range(len(cells)))
    logger.info("Running %d/%d cells sequentially", len(cell_ids), len(cells))
    run_cells(
        cells=cells,
        cell_ids=cell_ids,
        results_root=results_root,
        mlflow_enabled=True,
    )
