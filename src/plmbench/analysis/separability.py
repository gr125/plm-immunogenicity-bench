#!/usr/bin/env python3
"""Do the embeddings separate the labels? Silhouette, PERMANOVA, Fisher.

Replaces downstream_similarity.py. The statistics are unchanged; what changed
is that the script now takes its paths and parameters from configs/, scores
every label in one run, and writes results/metrics/separability.csv instead of
printing a frame that had to be parsed back out of a job log.

Usage:

    python -m plmbench.analysis.separability
    python -m plmbench.analysis.separability --variant mhc_i
    python -m plmbench.analysis.separability --labels tax_domain \
        --set separability.permutations=99
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from skbio.stats.distance import DistanceMatrix, permanova

from ..config import Config, add_config_args, config_from_args
from ..data.loader import embed_col, load_dataset
from ..io import write_table


def stack_embeddings(df: pd.DataFrame, col: str) -> np.ndarray:
    """Stack a column of vectors into a 2-D array (n, d)."""
    X = np.stack([np.asarray(v, dtype=float) for v in df[col].to_numpy()], axis=0)
    if X.ndim != 2:
        raise ValueError(f"embeddings must stack to 2-D; got shape {X.shape}")
    return X


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)


def _prepare(df: pd.DataFrame, col: str, normalize: bool) -> np.ndarray:
    X = stack_embeddings(df, col)
    return l2_normalize(X) if normalize else X


def silhouette_cosine(df, embed_col_, label_col, l2=True) -> float:
    """Silhouette score on cosine distances. In [-1, 1]."""
    X = _prepare(df, embed_col_, l2)
    return float(silhouette_score(cosine_distances(X), df[label_col].to_numpy(),
                                  metric="precomputed"))


def permanova_cosine(df, embed_col_, label_col, permutations=999, l2=True):
    """PERMANOVA on cosine distances; returns the scikit-bio result Series."""
    X = _prepare(df, embed_col_, l2)
    labels = df[label_col].astype(str).to_numpy()   # scikit-bio wants categoricals
    return permanova(DistanceMatrix(cosine_distances(X)), grouping=labels,
                     permutations=permutations)


def permanova_r2(result) -> float:
    """Variance explained, derived from pseudo-F.

        R2 = (k-1)F / ((k-1)F + (n-k))

    scikit-bio reports pseudo-F but not R2, and R2 is what makes the numbers
    comparable across labels with different group counts.
    """
    F = float(result["test statistic"])
    n = int(result["sample size"])
    k = int(result["number of groups"])
    return (k - 1) * F / ((k - 1) * F + (n - k))


def fisher_discriminant_ratio(df, embed_col_, label_col, l2=True,
                              metric="euclidean") -> pd.DataFrame:
    """Pairwise Fisher ratio ||muA - muB||^2 / (varA + varB) for all class pairs."""
    X = _prepare(df, embed_col_, l2)
    y = df[label_col].to_numpy()

    stats = {}
    for c in pd.unique(y):
        Xi = X[y == c]
        mu = Xi.mean(axis=0)
        if metric == "euclidean":
            var = float(np.mean(np.sum((Xi - mu) ** 2, axis=1)))
        elif metric == "cosine":
            mu_unit = mu / max(np.linalg.norm(mu), 1e-12)
            d = cosine_distances(Xi, mu_unit.reshape(1, -1)).reshape(-1)
            var = float(np.mean(d ** 2))
        else:
            raise ValueError("metric must be 'euclidean' or 'cosine'")
        stats[c] = {"mu": mu, "var": var, "n": Xi.shape[0]}

    rows = []
    classes = list(stats)
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            a, b = classes[i], classes[j]
            muA, muB = stats[a]["mu"], stats[b]["mu"]
            if metric == "euclidean":
                num = float(np.sum((muA - muB) ** 2))
            else:
                num = float(cosine_distances(muA.reshape(1, -1), muB.reshape(1, -1))[0, 0] ** 2)
            denom = stats[a]["var"] + stats[b]["var"]
            rows.append({
                "class_a": a, "class_b": b,
                "fisher_ratio": float(num / denom) if denom > 0 else np.inf,
                "n_a": stats[a]["n"], "n_b": stats[b]["n"],
                "var_a": stats[a]["var"], "var_b": stats[b]["var"],
            })
    return (pd.DataFrame(rows)
            .sort_values("fisher_ratio", ascending=False)
            .reset_index(drop=True))


def score_label(cfg: Config, data: pd.DataFrame, methods: list[str], label: str,
                params: dict) -> tuple[list[dict], list[pd.DataFrame]]:
    """Score every method against one label on one fixed subsample."""
    d = data.dropna(subset=[label]).reset_index(drop=True)
    print(f"=== label={label!r}: {len(methods)} embeddings on {len(d)} shared peptides ===")

    # Draw the PERMANOVA subsample ONCE, so every method sees identical rows.
    n_per = params.get("subsample_per_group")
    if n_per:
        idx = (d.groupby(label, group_keys=False)
                .apply(lambda g: g.sample(n=min(len(g), n_per),
                                          random_state=params.get("random_state", 0)))
                .index)
    else:
        idx = d.index

    # Silhouette needs a full n x n distance matrix -- 34 GB at n = 65,408.
    # Subsample it only if the config asks; null reproduces the published numbers.
    n_sil = params.get("silhouette_subsample")
    if n_sil and len(d) > n_sil:
        sil_idx = d.sample(n=n_sil, random_state=params.get("random_state", 0)).index
        print(f"  silhouette on a {n_sil}-row subsample "
              f"(silhouette_subsample); PERMANOVA uses its own.")
    else:
        sil_idx = d.index
        if params.get("silhouette", True):
            gb = (len(d) ** 2 * 8) / 1e9
            if gb > 8:
                print(f"  NOTE: silhouette will allocate a {len(d)}x{len(d)} distance "
                      f"matrix (~{gb:.0f} GB). Set separability.silhouette_subsample "
                      f"to bound it.")

    l2 = params.get("l2_normalize", True)
    rows, fisher_frames = [], []

    for name in methods:
        col = embed_col(name)
        print(f"{name}:")

        sil = np.nan
        if params.get("silhouette", True):
            sil = silhouette_cosine(d.loc[sil_idx], col, label, l2)
            print(f"  silhouette (cosine): {sil:.4f}")

        per = permanova_cosine(d.loc[idx], col, label,
                               permutations=params.get("permutations", 999), l2=l2)
        r2 = permanova_r2(per)
        print(f"  pseudo-F: {float(per['test statistic']):.3f}  "
              f"p: {per['p-value']}  R2: {r2:.4f}")

        if params.get("fisher", True):
            f = fisher_discriminant_ratio(d, col, label, l2)
            f.insert(0, "label", label)
            f.insert(0, "embedding", name)
            fisher_frames.append(f)

        rows.append({
            "embedding": name,
            "label": label,
            "n": len(d),
            "n_permanova": int(per["sample size"]),
            "silhouette": round(sil, 4) if sil == sil else np.nan,
            "pseudo_F": round(float(per["test statistic"]), 3),
            "p_value": per["p-value"],
            "R2": round(r2, 4),
        })
    print()
    return rows, fisher_frames


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Silhouette / PERMANOVA / Fisher separability per embedding.",
    )
    add_config_args(parser)
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="score a subset of the variant's embeddings")
    parser.add_argument("--labels", nargs="+", metavar="COL",
                        help="override separability.labels")
    args = parser.parse_args(argv)

    cfg = config_from_args(args)
    params = cfg.get("separability", {}) or {}
    labels = args.labels or params.get("labels", ["tax_domain"])
    names = args.only or cfg.selected_embeddings()

    # The comparison is only apples-to-apples on the intersection.
    cfg.variant["require"] = params.get("require", "all")

    print(f"variant: {cfg.get('name', args.variant)}   root: {cfg.root}")
    print(f"labels : {', '.join(labels)}\n")

    data, embed_cols = load_dataset(cfg, names)
    order = cfg.selected_embeddings()
    methods = [n for n in order if embed_col(n) in embed_cols]

    rows, fisher_frames = [], []
    for label in labels:
        if label not in data.columns:
            print(f"NOTE: no column {label!r} in the merged table -- skipping.")
            continue
        r, f = score_label(cfg, data, methods, label, params)
        rows += r
        fisher_frames += f

    if not rows:
        raise SystemExit("Nothing scored -- check separability.labels.")

    summary = pd.DataFrame(rows)
    summary["source"] = cfg.get("name", "full")
    print("=== Summary (identical peptide set within each label) ===")
    print(summary.to_string(index=False))

    out = write_table(summary, cfg.resolve(params["out_csv"]))
    print(f"\nwrote {out}")

    if fisher_frames:
        fout = write_table(pd.concat(fisher_frames, ignore_index=True),
                           cfg.resolve(params["fisher_csv"]))
        print(f"wrote {fout}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
