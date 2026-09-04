"""Atomic writes, resumable pickles and table output.

Extracted from ESMC_embed.py, which was the only script here that got this
right. The embedders and the analysis runners now share it, so a job killed by
the scheduler leaves a complete file behind rather than a truncated one.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | os.PathLike) -> Path:
    """mkdir -p, returning the directory."""
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def atomic_pickle_dump(obj: Any, out_file: str | os.PathLike) -> Path:
    """Pickle to a sibling .tmp, then os.replace onto the target.

    os.replace is atomic within a filesystem, so a reader (or a resumed job)
    never observes a half-written file.
    """
    out = Path(out_file)
    ensure_dir(out.parent)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, out)
    return out


def load_pickle(path: str | os.PathLike, missing_ok: bool = False):
    """Read a pickle; return None instead of raising when missing_ok."""
    p = Path(path)
    if not p.is_file():
        if missing_ok:
            return None
        raise FileNotFoundError(p)
    with open(p, "rb") as f:
        return pickle.load(f)


def resume_dict(out_file: str | os.PathLike, label: str = "items") -> dict:
    """Load a partially-complete embedding dict, or start an empty one."""
    existing = load_pickle(out_file, missing_ok=True)
    if existing is None:
        return {}
    print(f"Resuming: {len(existing)} {label} already embedded.")
    return existing


class CheckpointWriter:
    """Periodic atomic checkpoints for a dict that grows during a long job.

        ckpt = CheckpointWriter(out_file, every=1000)
        for ...:
            store[key] = value
            ckpt.tick(store, n=1)
        ckpt.flush(store)
    """

    def __init__(self, out_file: str | os.PathLike, every: int = 1000):
        self.out_file = Path(out_file)
        self.every = every
        self._since = 0

    def tick(self, obj: Any, n: int = 1) -> None:
        self._since += n
        if self._since >= self.every:
            self.flush(obj)

    def flush(self, obj: Any) -> None:
        atomic_pickle_dump(obj, self.out_file)
        self._since = 0


def write_table(df: pd.DataFrame, out_file: str | os.PathLike, **kwargs) -> Path:
    """Write a DataFrame atomically, gzipping when the name ends in .gz."""
    out = Path(out_file)
    ensure_dir(out.parent)
    tmp = out.with_suffix(out.suffix + ".tmp")
    kwargs.setdefault("index", False)
    if out.name.endswith(".gz"):
        kwargs.setdefault("compression", "gzip")
    df.to_csv(tmp, **kwargs)
    os.replace(tmp, out)
    return out
