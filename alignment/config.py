"""Pydantic configs for training and HPO."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class SplitConfig(BaseModel):
    n_folds: int = 5
    val_fraction: float = 0.15  # fraction of each fold's train portion held out for early stopping
    group_by: str = "pmid"
    stratify_by: str = "predicate"

    @field_validator("val_fraction")
    @classmethod
    def _fraction(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("val_fraction must be in (0,1)")
        return v

    @field_validator("n_folds")
    @classmethod
    def _n_folds(cls, v: int) -> int:
        if v < 2:
            raise ValueError("n_folds must be >= 2")
        return v


class TrainingConfig(BaseModel):
    batch_size: int = 256
    epochs: int = 200
    early_stop_patience: int = 15
    lr_plateau_patience: int = 5
    lr_plateau_factor: float = 0.5
    grad_clip: float = 1.0


class ModelConfig(BaseModel):
    hidden_dims: List[int] = Field(default_factory=lambda: [768, 512])
    dropout: float = 0.3
    architecture_type: str = "mlp"  # "mlp" | "linear" | "residual" | "bilinear" | "low_rank" | "cross_attention"


class LossConfig(BaseModel):
    temperature: float = 0.07
    n_hard_negs: int = 4
    class_balance: bool = False
    cluster_aware_negs: bool = False
    sample_reweighting: str = "none"  # "none" | "inv_sqrt_freq" | "inv_freq"
    neg_strategies: List[str] = Field(
        default_factory=lambda: [
            "same_s_same_o_vary_p",
            "same_s_vary_o",
            "vary_s_same_p_same_o",
            "same_p_vary_both",
        ]
    )  # subset or full list of STRATEGIES from negatives.py


class OptimizerConfig(BaseModel):
    name: str = "adamw"
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4


class EvalConfig(BaseModel):
    hits_k: List[int] = Field(default_factory=lambda: [1, 3, 5, 10, 25])
    val_mrr_every_n_epochs: int = 5
    eval_direction: str = "s2t"  # "s2t" | "t2s"

    @field_validator("eval_direction")
    @classmethod
    def _eval_direction(cls, v: str) -> str:
        if v not in {"s2t", "t2s"}:
            raise ValueError("eval_direction must be either 's2t' or 't2s'")
        return v


class PathConfig(BaseModel):
    evidence_path: str
    text_emb_dir: str
    kg_emb_dir: str
    checkpoints_dir: str
    mlruns_dir: str
    optuna_storage: str


class DefaultConfig(BaseModel):
    """Config for a single training run."""

    seed: int = 42
    text_model: str = "biobert_mcpt"
    kg_family: str = "rdf2vec"
    kg_config: str = "A_best_vec200"
    split: SplitConfig = Field(default_factory=SplitConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    paths: Optional[PathConfig] = None
    triple_combination: str = "concat"  # "concat" | "hadamard" | "l1" | "l2"
    training_direction: str = "text_to_kg"  # "text_to_kg" | "kg_to_text" | "bidirectional_random"

    @field_validator("training_direction")
    @classmethod
    def _training_direction(cls, v: str) -> str:
        allowed = {"text_to_kg", "kg_to_text", "bidirectional_random"}
        if v not in allowed:
            raise ValueError(
                "training_direction must be one of "
                "'text_to_kg', 'kg_to_text', or 'bidirectional_random'"
            )
        return v


# ---------------------------------------------------------------------------
# HPO config
# ---------------------------------------------------------------------------


class HPOParam(BaseModel):
    type: str  # categorical | uniform | loguniform
    choices: Optional[List[Any]] = None
    low: Optional[float] = None
    high: Optional[float] = None


class HPOSearchSpace(BaseModel):
    architecture: HPOParam
    architecture_type: HPOParam
    dropout: HPOParam
    learning_rate: HPOParam
    weight_decay: HPOParam
    n_hard_negs: HPOParam
    triple_combination: HPOParam
    neg_strategies: HPOParam


class HPOFixed(BaseModel):
    batch_size: int = 256
    temperature: float = 0.07
    optimizer: str = "adamw"
    epochs: int = 100
    early_stop_patience: int = 15


class OptunaConfig(BaseModel):
    storage: str = "sqlite:///optuna.db"
    sampler: str = "tpe"
    pruner: str = "median"


class HPOConfig(BaseModel):
    n_trials_per_combination: int = 10
    search_space: HPOSearchSpace
    fixed: HPOFixed = Field(default_factory=HPOFixed)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)
    optimization_metric: str = "test_mrr"
    direction: str = "maximize"
