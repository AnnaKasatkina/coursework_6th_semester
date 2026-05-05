from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data_prepared_csv(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def _extract_and_sort_m1_columns(df: pd.DataFrame) -> list[str]:
    m1_cols = [col for col in df.columns if col.startswith("m1_")]
    if not m1_cols:
        raise ValueError("No columns with prefix 'm1_' found.")
    return sorted(m1_cols, key=lambda c: int(c.split("_")[1]))


def build_meter_matrix(df: pd.DataFrame, expected_t: int = 1440, expected_meters: int | None = None) -> pd.DataFrame:
    m1_cols = _extract_and_sort_m1_columns(df)
    if len(m1_cols) < expected_t:
        raise ValueError(f"Expected at least {expected_t} m1 columns, found {len(m1_cols)}.")
    m1_cols = m1_cols[:expected_t]

    if "id" in df.columns:
        meter_ids = df["id"].astype(str)
    else:
        meter_ids = pd.Series(df.index.astype(str), index=df.index)
    matrix = df[m1_cols].copy()

    # Keep rows with manageable missingness and interpolate over time.
    row_nan_frac = matrix.isna().mean(axis=1)
    matrix = matrix.loc[row_nan_frac <= 0.5].copy()
    meter_ids = meter_ids.loc[matrix.index]
    matrix = matrix.interpolate(axis=1, limit_direction="both")
    matrix = matrix.fillna(0.0)

    # Drop fully zero rows and keep stable meter count for experiments.
    non_zero_mask = matrix.sum(axis=1) != 0
    matrix = matrix.loc[non_zero_mask].copy()
    meter_ids = meter_ids.loc[matrix.index]

    matrix.index = meter_ids.values
    matrix.columns = [f"t_{i+1:04d}" for i in range(len(matrix.columns))]

    if expected_meters is not None:
        if matrix.shape[0] < expected_meters:
            raise ValueError(f"Need at least {expected_meters} meters, found {matrix.shape[0]}.")
        if matrix.shape[0] > expected_meters:
            matrix = matrix.iloc[:expected_meters].copy()
    return matrix
