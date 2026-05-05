from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.utils.series import deserialize_series


class CounterfactualResidualBaseline(BaseModel):
    name = "counterfactual_residual_baseline"

    def __init__(self, threshold_z: float = 0.75) -> None:
        self.threshold_z = threshold_z

    @property
    def is_trainable(self) -> bool:
        return False

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "CounterfactualResidualBaseline":
        return self

    def _scores_for_group(self, group_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        eps = 1e-12
        true_agg = deserialize_series(group_df["true_aggregate_series"].iloc[0])
        observed_sum = deserialize_series(group_df["observed_sum_series"].iloc[0])
        mse_before = float(np.mean((true_agg - observed_sum) ** 2))

        raw_scores: list[float] = []
        for _, row in group_df.iterrows():
            meter = deserialize_series(row["meter_series"])

            # Fit best scalar correction for this meter only:
            # corrected_sum = observed_sum - meter + alpha * meter
            target = true_agg - (observed_sum - meter)
            denom = float(np.dot(meter, meter)) + eps
            alpha_hat = float(np.dot(target, meter) / denom)
            corrected_sum = observed_sum - meter + alpha_hat * meter
            mse_after = float(np.mean((true_agg - corrected_sum) ** 2))

            improvement_ratio = max(0.0, mse_before - mse_after) / (mse_before + eps)
            scale_bonus = max(0.0, alpha_hat - 1.0)
            score = improvement_ratio * scale_bonus
            raw_scores.append(score)

        raw = np.asarray(raw_scores, dtype=float)
        if raw.max() > raw.min():
            normalized = (raw - raw.min()) / (raw.max() - raw.min())
        else:
            normalized = np.zeros_like(raw)

        thr = float(normalized.mean() + self.threshold_z * normalized.std())
        thr = min(max(thr, 0.30), 0.80)
        preds = (normalized >= thr).astype(int)
        return normalized, preds

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        all_scores = np.zeros(len(df), dtype=float)
        for _, idx in df.groupby("group_id").groups.items():
            group_df = df.loc[idx]
            scores, _ = self._scores_for_group(group_df)
            all_scores[idx] = scores
        return all_scores

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        all_preds = np.zeros(len(df), dtype=int)
        for _, idx in df.groupby("group_id").groups.items():
            group_df = df.loc[idx]
            _, preds = self._scores_for_group(group_df)
            all_preds[idx] = preds
        return all_preds

    def save(self, model_dir: str | Path) -> None:
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        payload = {"name": self.name, "threshold_z": self.threshold_z}
        (path / "baseline_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: str | Path) -> "CounterfactualResidualBaseline":
        path = Path(model_dir) / "baseline_config.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(threshold_z=float(payload["threshold_z"]))

