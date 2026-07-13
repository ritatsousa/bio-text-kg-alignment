"""Run the verbalized-triple baseline once.

This script is intended for the parallel HPO setup where baselines are not part
of every array task. It writes:

  Results/baselines/final_results.csv
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

THIS = Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))

from config import (  # noqa: E402
    DefaultConfig,
    EvalConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    PathConfig,
    SplitConfig,
    TrainingConfig,
)
from baselines import encode_verbalized_triples, make_text_model_encoder, run_all_baselines  # noqa: E402
from data import prepare_text_only_data  # noqa: E402
from evaluation import average_fold_metrics  # noqa: E402
from qualitative import save_baseline_top_predictions, save_verbalized_triple_examples  # noqa: E402
from utils import get_logger  # noqa: E402

logger = get_logger("run_baselines")


def _flatten_row(prefix: Dict, metrics: Dict) -> Dict:
    row = dict(prefix)
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            row[k] = v
    return row


def run_baselines(
    base_cfg: DefaultConfig,
    text_models: List[str],
    eval_directions: List[str],
) -> List[Dict]:
    """Run verbalized-triple baselines without depending on run_hpo.py."""
    rows: List[Dict] = []
    encoders: Dict[str, object] = {}

    for text_model in text_models:
        cfg = copy.deepcopy(base_cfg)
        cfg.text_model = text_model

        folds = prepare_text_only_data(cfg)
        if text_model not in encoders:
            encoders[text_model] = make_text_model_encoder(text_model)
        encode_fn = encoders[text_model]

        per_direction_baseline: Dict[str, Dict[str, List[Dict]]] = {
            direction: {"verbalized_nn": []}
            for direction in eval_directions
        }
        for fold_i, fold_data in enumerate(folds):
            verb_embs = encode_verbalized_triples(fold_data, encode_fn)  # type: ignore[arg-type]
            for direction in eval_directions:
                results = run_all_baselines(
                    fold_data,
                    hits_k=cfg.eval.hits_k,
                    precomputed_verb_embs=verb_embs,
                    direction=direction,
                )
                for baseline_name, metrics in results.items():
                    out_dir = (
                        Path(cfg.paths.checkpoints_dir)
                        / "baselines"
                        / direction
                        / baseline_name
                        / text_model
                        / f"fold_{fold_i}"
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)
                    save_verbalized_triple_examples(
                        fold_data,
                        out_dir / "verbalized_triples_examples.csv",
                        n_examples=10,
                        seed=cfg.seed + fold_i,
                    )
                    save_baseline_top_predictions(
                        fold_data,
                        verb_embs,
                        direction,
                        out_dir / f"top_predictions_{direction}.csv",
                        n_queries=10,
                        top_k=10,
                        seed=cfg.seed + fold_i,
                    )
                    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
                        json.dump(metrics, f, indent=2)
                    per_direction_baseline[direction].setdefault(baseline_name, []).append(metrics)

        for direction, per_baseline in per_direction_baseline.items():
            for baseline_name, fold_metrics in per_baseline.items():
                if not fold_metrics:
                    continue
                avg = average_fold_metrics(fold_metrics)
                rows.append(
                    _flatten_row(
                        {
                            "training_direction": "baseline",
                            "eval_direction": direction,
                            "result_type": "baseline",
                            "baseline": baseline_name,
                            "text_model": text_model,
                            "kg_family": "",
                            "kg_config": "",
                            "architecture_type": "",
                            "triple_combination": "",
                            "neg_strategy": "",
                            "trial_number": "",
                        },
                        avg,
                    )
                )
    return rows


def make_paths(results_root: Path) -> PathConfig:
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
        optuna_storage=f"sqlite:///{(results_root / 'optuna_baselines_unused.db').as_posix()}",
    )


def make_base_config(paths: PathConfig) -> DefaultConfig:
    return DefaultConfig(
        seed=42,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run verbalized-triple baselines once.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=THIS / "Results",
        help="Root directory for outputs. Default: ./Results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_dir = results_root / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = make_paths(results_root)
    Path(paths.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(paths.mlruns_dir).mkdir(parents=True, exist_ok=True)

    base_cfg = make_base_config(paths)
    rows = run_baselines(
        base_cfg,
        text_models=["biobert_mcpt", "pubmedbert_mcpt"],
        eval_directions=["s2t", "t2s"],
    )

    output_path = output_dir / "final_results.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logger.info("Baseline results written to %s", output_path)


if __name__ == "__main__":
    main()
