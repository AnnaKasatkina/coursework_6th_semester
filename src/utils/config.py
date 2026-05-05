from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import json


@dataclass
class ExperimentConfig:
    M: int
    N: int
    J: int
    k: float
    model_list: list[str]
    seed: int = 42
    noise_std: float = 0.1
    split_train: float = 0.7
    split_val: float = 0.15
    split_test: float = 0.15
    base_dir: str = "."
    data_csv_path: str = "spring_semester/data/data_prepared.csv"
    output_data_dir: str = "spring_semester/data"
    runs_dir: str = "spring_semester/runs"
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

