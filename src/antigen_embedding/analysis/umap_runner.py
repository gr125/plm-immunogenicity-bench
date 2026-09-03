#!/usr/bin/env python3
"""One UMAP driver for every variant.

Replaces four forked scripts that differed by 78-86 diff lines, all of it
configuration:

    downstream_umap.py          -> configs/variants/full.yaml
    downstream_umap_esmc.py     -> configs/variants/full.yaml (esmc, esmcpep)
    downstream_umap_mhcI.py     -> configs/variants/mhc_i.yaml
    remove_anchor_points.py     -> configs/variants/no_anchor.yaml

Usage:

    python -m antigen_embedding.analysis.umap_runner --variant full
    python -m antigen_embedding.analysis.umap_runner --variant no_anchor
    python -m antigen_embedding.analysis.umap_runner --variant mhc_i \
        --only protbert --set umap.random_state=0
    # redraw figures from the committed coordinates, no pickles needed:
    python -m antigen_embedding.analysis.umap_runner --plots-only

The invariant that keeps this repo small: **embedding columns are never
written**. The output holds the join keys, the label columns, X and Y. That is
what took the seven UMAP tables from 8.6 GB to 12 MB, and what makes the
coordinates committable, so every figure regenerates from the repo alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..config import Config, add_config_args, config_from_args
from ..data.loader import EMBED_SUFFIX, load_dataset, load_taxonomy
from ..io import write_table
from . import plots

# Never written to a UMAP table. Everything here rejoins from data/.
NON_COMMITTABLE = ["protein_sequence", "protein_id", "slen", "plen"]


def slim(df: pd.DataFrame, extra_drop: list[str] | None = None) -> pd.DataFrame:
    """Drop embeddings and anything rejoinable from data/ before writing."""
    drop = [c for c in df.columns if c.endswith(EMBED_SUFFIX)]
    drop += [c for c in (extra_drop or NON_COMMITTABLE) if c in df.columns]
    return df.drop(columns=drop)


def umap_coords(X: np.ndarray, params: dict) -> np.ndarray:
    """Fit UMAP with the variant's parameters and return the 2-D embedding."""
    import umap.umap_ as umap  # imported late; it is slow and not always needed

    if params.get("standardize", True):
        X = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(
        init=params.get("init", "random"),
        n_neighbors=params.get("n_neighbors", 15),
        min_dist=params.get("min_dist", 0.1),
        metric=params.get("metric", "euclidean"),
        random_state=params.get("random_state"),
    )
    return reducer.fit_transform(X)


def table_path(umap_dir: Path, name: str) -> Path:
    return Path(umap_dir) / f"umap_{name}.csv.gz"


def read_existing(umap_dir: Path, name: str) -> pd.DataFrame | None:
    """Load a previously written table, tolerating the pre-gzip .csv name."""
    for p in (table_path(umap_dir, name), Path(umap_dir) / f"umap_{name}.csv"):
        if p.is_file():
            df = pd.read_csv(p)
            print(f"[{name}] loaded {len(df)} rows from {p.name}")
            return df
    return None


def run_embedding(cfg: Config, data: pd.DataFrame, name: str, umap_dir: Path,
                  overwrite: bool) -> pd.DataFrame | None:
    """Coordinates for one embedding: reuse the table if present, else fit."""
    col = f"{name}{EMBED_SUFFIX}"

    if not overwrite:
        existing = read_existing(umap_dir, name)
        if existing is not None:
            return existing

    d = data.dropna(subset=[col]).reset_index(drop=True)
    if len(d) < 10:
        print(f"[{name}] only {len(d)} embedded peptides -- skipping.")
        return None

    print(f"[{name}] running UMAP on {len(d)} peptides.")
    X = np.stack([np.asarray(v, dtype=float) for v in d[col].values], axis=0)
    coords = umap_coords(X, cfg.get("umap", {}) or {})

    d["X"], d["Y"] = coords[:, 0], coords[:, 1]
    out = write_table(slim(d), table_path(umap_dir, name))
    print(f"[{name}] wrote {out.name} "
          f"({out.stat().st_size / 1e6:.1f} MB, {len(d)} rows, no embedding columns)")
    return d


def plots_only(cfg: Config, names: list[str], umap_dir: Path, figures_dir: Path) -> int:
    """Redraw every figure from committed coordinates. No pickles required."""
    total = 0
    domains = cfg.get("plots.domains", []) or []
    for name in names:
        d = read_existing(umap_dir, name)
        if d is None:
            print(f"[{name}] no coordinates in {umap_dir} -- skipping.")
            continue
        if "tax_domain" not in d.columns:
            # A table written before the taxonomy columns were carried through.
            d = d.merge(load_taxonomy(cfg), how="left")
        family = cfg.embedding_spec(name).get("family", name)
        total += plots.plot_embedding(
            d, name, family, figures_dir, domains,
            figsize=cfg.get("plots.figsize", 40),
            min_kde_points=cfg.get("plots.min_kde_points", 10),
        )
    return total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run UMAP for one variant defined in configs/variants/.",
    )
    add_config_args(parser)
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="run a subset of the variant's embeddings")
    parser.add_argument("--plots-only", action="store_true",
                        help="redraw figures from existing coordinates; reads no pickles")
    parser.add_argument("--no-plots", action="store_true",
                        help="write coordinates but draw no figures")
    parser.add_argument("--overwrite", action="store_true",
                        help="refit even where a coordinate table already exists")
    args = parser.parse_args(argv)

    cfg = config_from_args(args)
    names = args.only or cfg.selected_embeddings()
    umap_dir = cfg.out_dir("output.umap_dir")
    figures_dir = cfg.out_dir("output.figures_dir")

    print(f"variant : {cfg.get('name', args.variant)}")
    print(f"root    : {cfg.root}")
    print(f"umap    -> {umap_dir}")
    print(f"figures -> {figures_dir}")
    print(f"embeddings: {', '.join(names)}\n")

    if args.plots_only:
        n = plots_only(cfg, names, umap_dir, figures_dir)
        print(f"\nDone. {n} figures.")
        return 0

    data, embed_cols = load_dataset(cfg, names)
    available = [c[: -len(EMBED_SUFFIX)] for c in embed_cols]

    overwrite = args.overwrite or bool(cfg.get("output.overwrite", False))
    draw = cfg.get("plots.enabled", True) and not args.no_plots
    domains = cfg.get("plots.domains", []) or []

    n_figs = 0
    for name in available:
        d = run_embedding(cfg, data, name, umap_dir, overwrite)
        if d is None or not draw:
            continue
        family = cfg.embedding_spec(name).get("family", name)
        n_figs += plots.plot_embedding(
            d, name, family, figures_dir, domains,
            figsize=cfg.get("plots.figsize", 40),
            min_kde_points=cfg.get("plots.min_kde_points", 10),
        )

    print(f"\nDone. {len(available)} embeddings, {n_figs} figures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
