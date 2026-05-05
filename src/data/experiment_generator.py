from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.tabular import build_tabular_features
from src.train.split import assign_group_splits
from src.utils.series import serialize_series


@dataclass
class GeneratorParams:
    M: int
    N: int
    J: int
    k: float
    seed: int = 42
    noise_std: float = 0.1
    split_train: float = 0.7
    split_val: float = 0.15
    split_test: float = 0.15


def generate_experiment_dataset(meter_matrix: pd.DataFrame, params: GeneratorParams) -> pd.DataFrame:
    if params.M <= 0 or params.N <= 0:
        raise ValueError("M and N must be positive.")
    if params.J < 0 or params.J > params.M:
        raise ValueError("J must satisfy 0 <= J <= M.")
    if not (0 < params.k <= 1):
        raise ValueError("k must satisfy 0 < k <= 1.")
    if params.M > meter_matrix.shape[0]:
        raise ValueError(f"M={params.M} exceeds available meters={meter_matrix.shape[0]}.")

    rng = np.random.default_rng(params.seed)
    meter_ids = meter_matrix.index.to_numpy(dtype=object)
    meter_values = meter_matrix.to_numpy(dtype=float)
    t = meter_values.shape[1]

    rows: list[dict[str, object]] = []
    group_ids: list[str] = []

    for group_idx in range(params.N):
        group_id = f"group_{group_idx:05d}"
        group_ids.append(group_id)

        sampled_indices = rng.choice(len(meter_ids), size=params.M, replace=False)
        true_group = meter_values[sampled_indices].copy()  # (M, T)
        true_aggregate = true_group.sum(axis=0)
        true_aggregate_noisy = true_aggregate + rng.normal(0.0, params.noise_std, size=t)

        fraud_positions = set()
        if params.J > 0:
            fraud_positions = set(rng.choice(params.M, size=params.J, replace=False).tolist())

        observed_group = true_group.copy()
        if fraud_positions:
            observed_group[list(fraud_positions), :] *= params.k
        observed_sum = observed_group.sum(axis=0)
        residual = true_aggregate_noisy - observed_sum

        for pos in range(params.M):
            meter_id = str(meter_ids[sampled_indices[pos]])
            label = int(pos in fraud_positions)
            meter_series = observed_group[pos]

            feature_dict = build_tabular_features(
                meter_series=meter_series,
                true_aggregate_series=true_aggregate_noisy,
                observed_sum_series=observed_sum,
                residual_series=residual,
            )

            row: dict[str, object] = {
                "sample_id": f"{group_id}_{meter_id}",
                "group_id": group_id,
                "meter_id": meter_id,
                "label": label,
                "M": params.M,
                "J": params.J,
                "k": params.k,
                "seed": params.seed,
                "meter_series": serialize_series(meter_series),
                "true_aggregate_series": serialize_series(true_aggregate_noisy),
                "observed_sum_series": serialize_series(observed_sum),
                "residual_series": serialize_series(residual),
            }
            row.update(feature_dict)
            rows.append(row)

    dataset = pd.DataFrame(rows)
    split_map = assign_group_splits(
        group_ids=group_ids,
        train_ratio=params.split_train,
        val_ratio=params.split_val,
        test_ratio=params.split_test,
        seed=params.seed,
    )
    dataset["split"] = dataset["group_id"].map(split_map)
    return dataset

