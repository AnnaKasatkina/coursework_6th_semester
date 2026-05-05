from __future__ import annotations

from typing import Iterable

import numpy as np


def assign_group_splits(
    group_ids: Iterable[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train/val/test ratios must sum to 1.0")

    group_ids_arr = np.array(list(group_ids), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(group_ids_arr)
    n = len(group_ids_arr)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    # Guarantee at least one group in val/test when possible.
    if n >= 3:
        if n_val == 0:
            n_val = 1
            n_train = max(n_train - 1, 1)
        if n_test == 0:
            n_test = 1
            n_train = max(n_train - 1, 1)

    split_map: dict[str, str] = {}
    for group_id in group_ids_arr[:n_train]:
        split_map[str(group_id)] = "train"
    for group_id in group_ids_arr[n_train : n_train + n_val]:
        split_map[str(group_id)] = "val"
    for group_id in group_ids_arr[n_train + n_val :]:
        split_map[str(group_id)] = "test"
    return split_map

