from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseModel(ABC):
    name: str = "base_model"

    @property
    def is_trainable(self) -> bool:
        return True

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "BaseModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def save(self, model_dir: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, model_dir: str | Path) -> "BaseModel":
        raise NotImplementedError

