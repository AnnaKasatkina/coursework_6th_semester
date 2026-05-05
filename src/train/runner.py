from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.data.experiment_generator import GeneratorParams, generate_experiment_dataset
from src.data.loaders import build_meter_matrix, load_data_prepared_csv
from src.metrics.classification import compute_binary_metrics
from src.models.base import BaseModel
from src.models.counterfactual_residual_baseline import CounterfactualResidualBaseline
from src.models.logistic_regression_model import LogisticRegressionModel
from src.utils.config import ExperimentConfig


def _build_model(model_name: str, model_params: dict[str, dict]) -> BaseModel:
    params = model_params.get(model_name, {})
    if model_name == "counterfactual_residual_baseline":
        return CounterfactualResidualBaseline(**params)
    if model_name == "logistic_regression":
        return LogisticRegressionModel(**params)
    raise ValueError(f"Unknown model '{model_name}'")


def _save_split_csvs(dataset: pd.DataFrame, output_data_dir: Path) -> None:
    output_data_dir.mkdir(parents=True, exist_ok=True)
    dataset.loc[dataset["split"] == "train"].to_csv(output_data_dir / "train.csv", index=False)
    dataset.loc[dataset["split"] == "val"].to_csv(output_data_dir / "val.csv", index=False)
    dataset.loc[dataset["split"] == "test"].to_csv(output_data_dir / "test.csv", index=False)


def run_experiment(config: ExperimentConfig) -> dict[str, str | pd.DataFrame]:
    base_dir = Path(config.base_dir).resolve()
    input_csv = (base_dir / config.data_csv_path).resolve()
    output_data_dir = (base_dir / config.output_data_dir).resolve()
    runs_dir = (base_dir / config.runs_dir).resolve()

    raw_df = load_data_prepared_csv(input_csv)
    meter_matrix = build_meter_matrix(raw_df, expected_t=1440, expected_meters=None)

    gen_params = GeneratorParams(
        M=config.M,
        N=config.N,
        J=config.J,
        k=config.k,
        seed=config.seed,
        noise_std=config.noise_std,
        split_train=config.split_train,
        split_val=config.split_val,
        split_test=config.split_test,
    )
    dataset = generate_experiment_dataset(meter_matrix, gen_params)
    _save_split_csvs(dataset, output_data_dir)

    train_df = dataset.loc[dataset["split"] == "train"].reset_index(drop=True)
    val_df = dataset.loc[dataset["split"] == "val"].reset_index(drop=True)
    test_df = dataset.loc[dataset["split"] == "test"].reset_index(drop=True)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = runs_dir / run_id
    plots_dir = run_dir / "plots"
    model_root_dir = run_dir / "model"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_root_dir.mkdir(parents=True, exist_ok=True)
    config.save_json(run_dir / "config.json")

    all_predictions: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, object]] = []

    for model_name in config.model_list:
        model = _build_model(model_name, config.model_params)
        train_start = perf_counter()
        model.fit(train_df, val_df)
        train_runtime = perf_counter() - train_start

        predict_start = perf_counter()
        y_score = model.predict_proba(test_df)
        y_pred = model.predict(test_df)
        predict_runtime = perf_counter() - predict_start

        y_true = test_df["label"].to_numpy(dtype=int)
        metric_dict = compute_binary_metrics(y_true=y_true, y_pred=y_pred)
        metric_dict.update(
            {
                "model": model_name,
                "split": "test",
                "runtime_train_sec": float(train_runtime),
                "runtime_predict_sec": float(predict_runtime),
            }
        )
        metrics_rows.append(metric_dict)

        pred_df = test_df[["sample_id", "group_id", "meter_id", "label", "split"]].copy()
        pred_df["model"] = model_name
        pred_df["score"] = y_score
        pred_df["prediction"] = y_pred
        all_predictions.append(pred_df)

        cm_df = pd.DataFrame(
            [[metric_dict["tn"], metric_dict["fp"]], [metric_dict["fn"], metric_dict["tp"]]],
            index=["true_0", "true_1"],
            columns=["pred_0", "pred_1"],
        )
        cm_df.to_csv(plots_dir / f"{model_name}_confusion_matrix.csv")

        if model.is_trainable:
            model_dir = model_root_dir / model_name
            model.save(model_dir)
        else:
            baseline_cfg_dir = run_dir / f"{model_name}_artifacts"
            model.save(baseline_cfg_dir)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df.to_csv(run_dir / "predictions.csv", index=False)
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "predictions_path": str(run_dir / "predictions.csv"),
        "metrics_path": str(run_dir / "metrics.csv"),
        "train_path": str(output_data_dir / "train.csv"),
        "val_path": str(output_data_dir / "val.csv"),
        "test_path": str(output_data_dir / "test.csv"),
        "metrics_df": metrics_df,
        "predictions_df": predictions_df,
    }
