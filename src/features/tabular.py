from __future__ import annotations

import numpy as np


EPS = 1e-12


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std < EPS or b_std < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    q25, q50, q75 = np.quantile(values, [0.25, 0.50, 0.75])
    mean = float(np.mean(values))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_q25": float(q25),
        f"{prefix}_q50": float(q50),
        f"{prefix}_q75": float(q75),
        f"{prefix}_sum": float(np.sum(values)),
        f"{prefix}_l1": float(np.sum(np.abs(values))),
        f"{prefix}_energy": float(np.mean(values**2)),
        f"{prefix}_zero_frac": float(np.mean(values == 0)),
        f"{prefix}_peak_to_mean": float(np.max(values) / (abs(mean) + EPS)),
        f"{prefix}_mean_abs_diff": float(np.mean(np.abs(np.diff(values)))),
    }


def build_tabular_features(
    meter_series: np.ndarray,
    true_aggregate_series: np.ndarray,
    observed_sum_series: np.ndarray,
    residual_series: np.ndarray,
) -> dict[str, float]:
    meter = np.asarray(meter_series, dtype=float)
    true_agg = np.asarray(true_aggregate_series, dtype=float)
    observed_sum = np.asarray(observed_sum_series, dtype=float)
    residual = np.asarray(residual_series, dtype=float)

    features: dict[str, float] = {}
    features.update(_stats(meter, "meter"))
    features.update(_stats(true_agg, "true_agg"))
    features.update(_stats(observed_sum, "observed_sum"))
    features.update(_stats(residual, "residual"))

    meter_sum = float(np.sum(meter))
    true_sum = float(np.sum(true_agg))
    observed_sum_total = float(np.sum(observed_sum))

    features.update(
        {
            "meter_to_true_sum_ratio": meter_sum / (true_sum + EPS),
            "meter_to_observed_sum_ratio": meter_sum / (observed_sum_total + EPS),
            "meter_corr_true_agg": _safe_corr(meter, true_agg),
            "meter_corr_observed_sum": _safe_corr(meter, observed_sum),
            "meter_corr_residual": _safe_corr(meter, residual),
            "meter_cos_true_agg": float(np.dot(meter, true_agg) / ((np.linalg.norm(meter) * np.linalg.norm(true_agg)) + EPS)),
            "meter_cos_observed_sum": float(
                np.dot(meter, observed_sum) / ((np.linalg.norm(meter) * np.linalg.norm(observed_sum)) + EPS)
            ),
            "meter_residual_dot_norm": float(np.dot(meter, residual) / (np.linalg.norm(residual) ** 2 + EPS)),
            "meter_mean_over_abs_residual_mean": float(np.mean(meter) / (np.mean(np.abs(residual)) + EPS)),
        }
    )
    return features

