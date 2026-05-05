from __future__ import annotations

from src.train.runner import run_experiment
from src.utils.config import ExperimentConfig


if __name__ == "__main__":
    config = ExperimentConfig(
        M=10,
        N=120,
        J=2,
        k=0.2,
        model_list=["counterfactual_residual_baseline", "logistic_regression"],
        seed=42,
        base_dir=".",
    )
    result = run_experiment(config)
    print(f"Run finished: {result['run_id']}")
    print(f"Metrics: {result['metrics_path']}")

