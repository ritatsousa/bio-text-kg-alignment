"""MLflow tracking helpers."""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import mlflow


def init_mlflow(tracking_uri: Optional[str] = None) -> None:
    """Configure MLflow tracking URI to the local mlruns dir by default."""
    if tracking_uri is None:
        path = Path("mlruns")
        path.mkdir(parents=True, exist_ok=True)
        tracking_uri = path.resolve().as_uri()
    elif not tracking_uri.startswith(("http://", "https://", "databricks:", "file:")):
        path = Path(tracking_uri)
        path.mkdir(parents=True, exist_ok=True)
        tracking_uri = path.resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)


def ensure_experiment(experiment_name: str) -> str:
    """Create experiment if missing; return its id."""
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return client.create_experiment(experiment_name)
    return exp.experiment_id


@contextlib.contextmanager
def start_run(
    experiment_name: str,
    run_name: str,
    tags: Optional[Dict[str, str]] = None,
) -> Iterator[Any]:
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run


def log_params(params: Dict[str, Any]) -> None:
    clean: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, (list, tuple)):
            clean[k] = "-".join(str(x) for x in v)
        else:
            clean[k] = v
    mlflow.log_params(clean)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                mlflow.log_metric(k, float(v), step=step)
            except Exception:
                pass


def log_artifact(path: Path) -> None:
    if path.exists():
        mlflow.log_artifact(str(path))
