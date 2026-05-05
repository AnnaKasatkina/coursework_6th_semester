from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import json
import numpy as np
import pandas as pd

from src.models.base import BaseModel


@dataclass
class StandardScalerState:
    mean_: np.ndarray
    scale_: np.ndarray


class StandardScaler:
    def __init__(self) -> None:
        self.state: StandardScalerState | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        mean_ = x.mean(axis=0)
        scale_ = x.std(axis=0)
        scale_ = np.where(scale_ < 1e-12, 1.0, scale_)
        self.state = StandardScalerState(mean_=mean_, scale_=scale_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Scaler is not fitted.")
        return (x - self.state.mean_) / self.state.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def save(self, path: str | Path) -> None:
        if self.state is None:
            raise RuntimeError("Scaler is not fitted.")
        path = Path(path)
        np.savez(path, mean_=self.state.mean_, scale_=self.state.scale_)

    @classmethod
    def load(cls, path: str | Path) -> "StandardScaler":
        payload = np.load(Path(path))
        scaler = cls()
        scaler.state = StandardScalerState(mean_=payload["mean_"], scale_=payload["scale_"])
        return scaler


class LogisticRegressionModel(BaseModel):
    name = "logistic_regression"

    def __init__(
        self,
        lr: float = 0.05,
        n_iter: int = 1200,
        l2: float = 1e-4,
        threshold: float = 0.5,
    ) -> None:
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.threshold = threshold
        self.feature_columns: list[str] = []
        self.scaler = StandardScaler()
        self.weights: np.ndarray | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _extract_feature_columns(df: pd.DataFrame) -> list[str]:
        exclude = {
            "sample_id",
            "group_id",
            "meter_id",
            "label",
            "M",
            "J",
            "k",
            "seed",
            "split",
            "meter_series",
            "true_aggregate_series",
            "observed_sum_series",
            "residual_series",
        }
        return [col for col in df.columns if col not in exclude]

    def _to_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.feature_columns].to_numpy(dtype=float)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "LogisticRegressionModel":
        self.feature_columns = self._extract_feature_columns(train_df)
        x = self._to_matrix(train_df)
        y = train_df["label"].to_numpy(dtype=float)

        x_scaled = self.scaler.fit_transform(x)
        xb = np.hstack([np.ones((x_scaled.shape[0], 1)), x_scaled])
        w = np.zeros(xb.shape[1], dtype=float)

        for _ in range(self.n_iter):
            p = self._sigmoid(xb @ w)
            grad = (xb.T @ (p - y)) / y.size
            grad[1:] += self.l2 * w[1:]
            w -= self.lr * grad
        self.weights = w
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not fitted.")
        x = self._to_matrix(df)
        x_scaled = self.scaler.transform(x)
        xb = np.hstack([np.ones((x_scaled.shape[0], 1)), x_scaled])
        return self._sigmoid(xb @ self.weights)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(df)
        return (proba >= self.threshold).astype(int)

    def save(self, model_dir: str | Path) -> None:
        if self.weights is None:
            raise RuntimeError("Model is not fitted.")
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        np.save(model_dir / "weights.npy", self.weights)
        self.scaler.save(model_dir / "preprocessing_scaler.npz")
        (model_dir / "feature_columns.json").write_text(json.dumps(self.feature_columns, indent=2), encoding="utf-8")
        config = {
            "name": self.name,
            "lr": self.lr,
            "n_iter": self.n_iter,
            "l2": self.l2,
            "threshold": self.threshold,
        }
        (model_dir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: str | Path) -> "LogisticRegressionModel":
        model_dir = Path(model_dir)
        config = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
        model = cls(lr=config["lr"], n_iter=config["n_iter"], l2=config["l2"], threshold=config["threshold"])
        model.weights = np.load(model_dir / "weights.npy")
        model.scaler = StandardScaler.load(model_dir / "preprocessing_scaler.npz")
        model.feature_columns = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))
        return model

