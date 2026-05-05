from __future__ import annotations

from typing import Iterable

import numpy as np


def serialize_series(values: Iterable[float], precision: int = 8) -> str:
    arr = np.asarray(values, dtype=float)
    fmt = f"{{:.{precision}f}}"
    return ";".join(fmt.format(x) for x in arr)


def deserialize_series(text: str) -> np.ndarray:
    if text is None or text == "":
        return np.array([], dtype=float)
    return np.fromstring(text, sep=";", dtype=float)

