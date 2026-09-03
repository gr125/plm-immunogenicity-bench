import pandas as pd
import numpy as np
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import re
import sys
import yaml
import pickle 
from taxonomy import process_taxonomy_column
import os
import time

path = '/mnt/bioadhoc/Groups/Peters/Self-similarity/'

from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from skbio.stats.distance import permanova, DistanceMatrix


def _stack_embeddings(df: pd.DataFrame, embed_col: str) -> np.ndarray:
    """Stack a column of vectors (np.ndarray/list) into a 2D array (n, d)."""
    X = df[embed_col].to_numpy()
    # Convert each entry to np.array and stack
    X = np.stack([np.asarray(v, dtype=float) for v in X], axis=0)
    if X.ndim != 2:
        raise ValueError(f"Embeddings must stack to a 2D array; got shape {X.shape}")
    return X


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def silhouette_cosine(
    df: pd.DataFrame,
    embed_col: str = "embedding",
    label_col: str = "class",
    l2_normalize: bool = True,
) -> float:
    """
    Silhouette score using cosine distance on embeddings.
    Returns a single float in [-1, 1].
    """
    if label_col not in df.columns or embed_col not in df.columns:
        raise KeyError(f"df must contain columns: {embed_col}, {label_col}")

    labels = df[label_col].to_numpy()
    X = _stack_embeddings(df, embed_col)
    if l2_normalize:
        X = _l2_normalize(X)

    # Precompute cosine distance matrix
    D = cosine_distances(X)
    return float(silhouette_score(D, labels, metric="precomputed"))


def permanova_cosine(
    df: pd.DataFrame,
    embed_col: str = "embedding",
    label_col: str = "class",
    permutations: int = 999,
    l2_normalize: bool = True,
):
    """
    PERMANOVA on cosine distances.
    Returns the scikit-bio PERMANOVA result object (has pseudo-F, p-value, etc.).
    """
    if label_col not in df.columns or embed_col not in df.columns:
        raise KeyError(f"df must contain columns: {embed_col}, {label_col}")

    labels = df[label_col].astype(str).to_numpy()  # scikit-bio likes strings/categoricals
    X = _stack_embeddings(df, embed_col)
    if l2_normalize:
        X = _l2_normalize(X)

    D = cosine_distances(X)
    dm = DistanceMatrix(D)  # ids optional
    return permanova(dm, grouping=labels, permutations=permutations)


def fisher_discriminant_ratio(
    df: pd.DataFrame,
    embed_col: str = "embedding",
    label_col: str = "class",
    l2_normalize: bool = True,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """
    Compute pairwise Fisher discriminant ratio between all class pairs.

    Fisher(A,B) = ||muA - muB||^2 / (varA + varB)
    where varA = mean_i ||x_i - muA||^2 within class A.

    metric:
      - "euclidean": uses Euclidean norm in the formula (standard)
      - "cosine": uses cosine distance to define numerator and within-class dispersion
                 (often reasonable for embeddings after L2 normalization)
    Returns a DataFrame with columns: class_a, class_b, fisher_ratio, n_a, n_b
    """
    if label_col not in df.columns or embed_col not in df.columns:
        raise KeyError(f"df must contain columns: {embed_col}, {label_col}")

    X = _stack_embeddings(df, embed_col)
    if l2_normalize:
        X = _l2_normalize(X)
    y = df[label_col].to_numpy()

    classes = pd.unique(y)
    # Precompute per-class stats
    stats = {}
    for c in classes:
        Xi = X[y == c]
        if Xi.shape[0] < 2:
            # Fisher ratio still computable but within-class variance may be ~0
            pass
        mu = Xi.mean(axis=0)

        if metric == "euclidean":
            # mean squared Euclidean distance to centroid
            var = float(np.mean(np.sum((Xi - mu) ** 2, axis=1)))
        elif metric == "cosine":
            # cosine "dispersion": mean squared cosine distance to centroid direction
            mu_unit = mu / max(np.linalg.norm(mu), 1e-12)
            # cosine distance between each point and mu_unit
            # cosine_distances expects 2D arrays
            d = cosine_distances(Xi, mu_unit.reshape(1, -1)).reshape(-1)
            var = float(np.mean(d ** 2))
        else:
            raise ValueError("metric must be 'euclidean' or 'cosine'")

        stats[c] = {"mu": mu, "var": var, "n": Xi.shape[0]}

    rows = []
    classes_list = list(classes)
    for i in range(len(classes_list)):
        for j in range(i + 1, len(classes_list)):
            a, b = classes_list[i], classes_list[j]
            muA, muB = stats[a]["mu"], stats[b]["mu"]
            varA, varB = stats[a]["var"], stats[b]["var"]

            if metric == "euclidean":
                num = float(np.sum((muA - muB) ** 2))
            else:  # cosine
                num = float(cosine_distances(muA.reshape(1, -1), muB.reshape(1, -1))[0, 0] ** 2)

            denom = varA + varB
            fisher = float(num / denom) if denom > 0 else np.inf

            rows.append(
                {
                    "class_a": a,
                    "class_b": b,
                    "fisher_ratio": fisher,
                    "n_a": stats[a]["n"],
                    "n_b": stats[b]["n"],
                    "var_a": varA,
                    "var_b": varB,
                }
            )

    out = pd.DataFrame(rows).sort_values("fisher_ratio", ascending=False).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Unified loader: every embedding, restricted to the SAME set of peptides.
# ---------------------------------------------------------------------------

# Peptide-keyed pickles that are already pooled 1-D vectors (use as-is)
DIRECT_PEP = {
    'pepbert_nf':   'analysis/PeptideBERT/nf-0828_0049_peptide_representations.pkl',
    'pepbert_sol':  'analysis/PeptideBERT/sol-0827_2316_peptide_representations.pkl',
    'pepbert_hemo': 'analysis/PeptideBERT/hemo-0828_0039_peptide_representations.pkl',
}
# Peptide-keyed token arrays [pep_len, D] -> mean-pool
POOL_PEP = {
    'protbertpep': 'analysis/ProtBert/results/data/peptide_representations.pkl',
    'esmcpep':     'analysis/ESM-embedding/results/data/peptide_representations.pkl',
}
# Protein-keyed token arrays [prot_len, D] -> slice the peptide span -> mean-pool
SLICE_PROT = {
    'protbert': 'analysis/ProtBert/results/data/token_representations.pkl',
    'esmc':     'analysis/ESM-embedding/results/data/token_representations.pkl',
}

# Order used for display / iteration
METHOD_ORDER = ['protbert', 'protbertpep', 'pepbert_nf', 'pepbert_sol',
                'pepbert_hemo', 'esmc', 'esmcpep']


def _slice_pool(prot_emb, slen, start, end):
    """Mean-pool the peptide span out of a protein's token array; NaN if invalid."""
    if not isinstance(prot_emb, np.ndarray):
        return np.nan
    if pd.isna(start) or pd.isna(end):
        return np.nan
    start = int(start) - 1
    end = int(end)
    if end - start != slen or start < 0 or end > prot_emb.shape[0]:
        return np.nan
    return prot_emb[start:end].mean(axis=0)


def load_common_data():
    """Load all embeddings and keep only rows present in EVERY one.

    Returns (data_merged, embed_cols). Every row of data_merged has a valid
    vector in every embed col, so all methods are scored on the identical set
    of peptides -- an apples-to-apples comparison.
    """
    data_ug = pd.read_csv(path + 'analysis/data_ug_tax.csv')

    data = pd.read_csv(path + 'data/data.csv')
    data['slen'] = data['linear_sequence'].str.len()
    data['plen'] = data['protein_sequence'].str.len()
    data = data.loc[data['plen'] < 7000].reset_index(drop=True)

    embed_cols = []

    def _load(rel):
        fp = path + rel
        if not os.path.isfile(fp):
            print(f"  MISSING: {fp}")
            return None
        with open(fp, 'rb') as f:
            return pickle.load(f)

    for name, rel in DIRECT_PEP.items():
        emb = _load(rel)
        if emb is None:
            continue
        col = f"{name}_pep_embed"
        data[col] = data['linear_sequence'].map(emb)
        embed_cols.append(col)
        print(f"{name}: {data[col].notna().sum()} peptides")

    for name, rel in POOL_PEP.items():
        emb = _load(rel)
        if emb is None:
            continue
        col = f"{name}_pep_embed"
        tok = data['linear_sequence'].map(emb)
        data[col] = [t.mean(axis=0) if isinstance(t, np.ndarray) else np.nan for t in tok]
        embed_cols.append(col)
        print(f"{name}: {data[col].notna().sum()} peptides")

    for name, rel in SLICE_PROT.items():
        emb = _load(rel)
        if emb is None:
            continue
        col = f"{name}_pep_embed"
        prot = data['protein_sequence'].map(emb)
        data[col] = [
            _slice_pool(p, s, st, en)
            for p, s, st, en in zip(prot, data['slen'],
                                    data['curated_source_antigen_start'],
                                    data['curated_source_antigen_end'])
        ]
        embed_cols.append(col)
        print(f"{name}: {data[col].notna().sum()} peptides")

    if not embed_cols:
        raise SystemExit("No embedding pickles found -- run the embedding scripts first.")

    # INTERSECTION: keep only rows that have EVERY embedding (how='any' default)
    before = len(data)
    data = data.dropna(subset=embed_cols)
    print(f"\nCommon set: {len(data)} / {before} rows present in all "
          f"{len(embed_cols)} embeddings.")

    merge_cols = [
        "linear_sequence", "mhc_class", "mhc_restriction",
        "curated_source_antigen_start", "curated_source_antigen_end",
    ]
    data_merged = data_ug.merge(
        data[merge_cols + ["protein_sequence"] + embed_cols],
        on=merge_cols, how="inner",
    )
    print(f"{len(data_merged)} rows after taxonomy merge.\n")
    return data_merged, embed_cols


data_merged, embed_cols = load_common_data()
methods = [m for m in METHOD_ORDER if f"{m}_pep_embed" in embed_cols]
methods += [c[:-len("_pep_embed")] for c in embed_cols
            if c[:-len("_pep_embed")] not in methods]

LABEL = "tax_domain"
data_merged = data_merged.dropna(subset=[LABEL]).reset_index(drop=True)
print(f"=== Evaluating {len(methods)} embeddings on {len(data_merged)} shared "
      f"peptides (label='{LABEL}') ===\n")

# Fix the PERMANOVA subsample ONCE so every method sees identical rows.
pa_idx = (data_merged.groupby(LABEL, group_keys=False)
          .apply(lambda g: g.sample(n=min(len(g), 2000), random_state=0)).index)

summary = []
for embed_type in methods:
    col = f"{embed_type}_pep_embed"
    print(f"{embed_type}:")

    sil = silhouette_cosine(data_merged, embed_col=col, label_col=LABEL)
    print(f"  Silhouette (cosine): {sil:.4f}")

    fisher_pairs = fisher_discriminant_ratio(
        data_merged, embed_col=col, label_col=LABEL,
        l2_normalize=True, metric="euclidean")
    print(fisher_pairs.head(10).to_string(index=False))

    per = permanova_cosine(
        data_merged.loc[pa_idx], embed_col=col, label_col=LABEL,
        permutations=999)
    print(per)
    print()

    F = float(per["test statistic"])
    n = int(per["sample size"])
    k = int(per["number of groups"])
    r2 = (k - 1) * F / ((k - 1) * F + (n - k))   # variance explained
    summary.append({"embedding": embed_type, "n": len(data_merged),
                    "silhouette": round(sil, 4), "pseudo_F": round(F, 3),
                    "p": per["p-value"], "R2": round(r2, 4)})

print("=== Summary (identical peptide set across all rows) ===")
print(pd.DataFrame(summary).to_string(index=False))