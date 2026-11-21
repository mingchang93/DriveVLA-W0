#!/usr/bin/env python3
"""Minimal replacement for ``utils.dataset.normalize_pi0`` used by step4.

The original module lives outside this repository.  Only the pieces step4 needs
are provided: streaming accumulation of action samples, the mean/std/q01/q99
statistics, and the ``{"norm_stats": {...}}`` JSON written next to the pickle so
the inference runner can de-normalise with the same file.
"""

import json
import os
from types import SimpleNamespace

import numpy as np

STATS_FILE = "norm_stats.json"


class RunningStats:
    def __init__(self):
        self._chunks = []

    def update(self, values):
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self._chunks.append(arr)

    def get_statistics(self):
        if not self._chunks:
            raise ValueError("no samples accumulated")
        data = np.concatenate(self._chunks, axis=0)
        return SimpleNamespace(
            mean=np.mean(data, axis=0).astype(np.float64),
            std=np.std(data, axis=0).astype(np.float64),
            q01=np.percentile(data, 1, axis=0).astype(np.float64),
            q99=np.percentile(data, 99, axis=0).astype(np.float64),
        )


def _to_json(stats):
    if isinstance(stats, dict):
        return {k: _to_json(v) for k, v in stats.items()}
    return {k: np.asarray(getattr(stats, k)).tolist() for k in ("mean", "std", "q01", "q99")}


def save(directory, norm_stats):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, STATS_FILE)
    with open(path, "w") as f:
        json.dump({"norm_stats": _to_json(norm_stats)}, f, indent=2)
    return path


def load(path):
    with open(path, "r") as f:
        payload = json.load(f)["norm_stats"]
    return SimpleNamespace(
        **{k: SimpleNamespace(**{m: np.asarray(v[m]) for m in ("mean", "std", "q01", "q99")})
           for k, v in payload.items()}
    )
