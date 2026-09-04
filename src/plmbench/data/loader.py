"""One loader for every downstream script.

Replaces the four near-identical `load_all_data()` functions in
downstream_umap.py, downstream_umap_esmc.py, downstream_umap_mhcI.py and
remove_anchor_points.py, which differed only in which pickles they opened,
which rows they kept, and where they sliced.

It also re-points the pipeline at the normalized tables. The old scripts read
`data/data.csv`, a flat file carrying `protein_sequence` on all 30,379 rows;
that file no longer exists. `protein_sequence` now comes from a join:

    epitopes.csv.gz (protein_id) -> proteins.csv.gz

which is why `filters.max_protein_length` is applied after the join, exactly
where the old `data['plen'] < 7000` sat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..io import load_pickle

# The natural key shared by the modelling table and the assay-level taxonomy
# table. Both sides are unique on nothing else.
MERGE_COLS = [
    "linear_sequence",
    "mhc_class",
    "mhc_restriction",
    "curated_source_antigen_start",
    "curated_source_antigen_end",
]

EMBED_SUFFIX = "_pep_embed"


def embed_col(name: str) -> str:
    return f"{name}{EMBED_SUFFIX}"


def embedding_name(col: str) -> str:
    return col[: -len(EMBED_SUFFIX)]


# --------------------------------------------------------------------------
# base tables
# --------------------------------------------------------------------------

def load_epitopes(cfg: Config) -> pd.DataFrame:
    """Peptides joined to their parent antigen sequence, with filters applied."""
    epi = pd.read_csv(cfg.path("data.epitopes"))
    prot = pd.read_csv(cfg.path("data.proteins"))

    df = epi.merge(prot[["protein_id", "protein_sequence"]], on="protein_id", how="left")
    df["slen"] = df["linear_sequence"].str.len()
    df["plen"] = df["protein_sequence"].str.len()

    filters = cfg.get("filters", {}) or {}

    max_plen = filters.get("max_protein_length")
    if max_plen is not None:
        df = df.loc[df["plen"] < max_plen]

    mhc_class = filters.get("mhc_class")
    if mhc_class is not None:
        df = df.loc[df["mhc_class"] == mhc_class]

    coord_ok = filters.get("coord_ok")
    if coord_ok is not None:
        before = len(df)
        df = df.loc[df["coord_ok"] == coord_ok]
        print(f"coord_ok == {coord_ok}: kept {len(df)} / {before} rows.")

    return df.reset_index(drop=True)


def load_taxonomy(cfg: Config) -> pd.DataFrame:
    """The assay-level table with the 12 resolved NCBI ranks."""
    return pd.read_csv(cfg.path("data.taxonomy"))


# --------------------------------------------------------------------------
# embedding derivation
# --------------------------------------------------------------------------

def _pool(arr, trim_start: int, trim_end: int):
    """Mean-pool a token array after trimming residues off each end."""
    stop = arr.shape[0] - trim_end
    if stop <= trim_start:
        return np.nan
    return arr[trim_start:stop].mean(axis=0)


def _slice_pool(prot_emb, slen, start, end, trim_start: int, trim_end: int):
    """Slice the peptide span out of a protein's tokens, then mean-pool.

    Returns NaN rather than raising when the coordinates do not describe a
    span of the peptide's length, or fall outside the embedded protein. The
    original scripts used `assert end - start == slen`, which aborted the whole
    run on the first bad row.
    """
    if not isinstance(prot_emb, np.ndarray):
        return np.nan
    if pd.isna(start) or pd.isna(end):
        return np.nan
    start = int(start) - 1          # 1-based inclusive -> 0-based
    end = int(end)                  # end is inclusive, so no -1
    if end - start != slen or start < 0 or end > prot_emb.shape[0]:
        return np.nan
    return _pool(prot_emb[start:end], trim_start, trim_end)


def attach_embedding(df: pd.DataFrame, spec: dict, trim_start: int = 0,
                     trim_end: int = 0, quiet: bool = False) -> str | None:
    """Add one `<name>_pep_embed` column to df. Returns the column, or None.

    None means the pickle is absent -- a partial embedding set is not an error,
    so a run can proceed while other embedding jobs are still going.
    """
    name, kind = spec["name"], spec["kind"]
    col = embed_col(name)

    store = load_pickle(spec["pickle"], missing_ok=True)
    if store is None:
        print(f"  MISSING: {spec['pickle']} -- skipping {name!r}.")
        return None

    trimmed = trim_start or trim_end

    if kind == "vector":
        if trimmed and not quiet:
            print(f"  NOTE: {name!r} is a pre-pooled vector; the "
                  f"trim_start={trim_start}/trim_end={trim_end} ablation cannot "
                  f"reach it and its coordinates will match the untrimmed run.")
        df[col] = df["linear_sequence"].map(store)

    elif kind == "pool_peptide":
        tok = df["linear_sequence"].map(store)
        df[col] = [
            _pool(t, trim_start, trim_end) if isinstance(t, np.ndarray) else np.nan
            for t in tok
        ]

    elif kind == "slice_protein":
        prot = df["protein_sequence"].map(store)
        df[col] = [
            _slice_pool(p, s, st, en, trim_start, trim_end)
            for p, s, st, en in zip(prot, df["slen"],
                                    df["curated_source_antigen_start"],
                                    df["curated_source_antigen_end"])
        ]

    else:
        raise ValueError(
            f"embedding {name!r} has unknown kind {kind!r}; expected "
            f"'vector', 'pool_peptide' or 'slice_protein'"
        )

    df[col] = df[col].astype(object)
    if not quiet:
        print(f"  {name}: {df[col].notna().sum()} / {len(df)} peptides embedded "
              f"({len(store)} keys in pickle)")
    return col


# --------------------------------------------------------------------------
# the entry point every runner uses
# --------------------------------------------------------------------------

def load_dataset(cfg: Config, embeddings: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Peptides + taxonomy + every requested embedding, per the variant config.

    `require: any` keeps a row that has at least one embedding, and each
    embedding is reduced independently, so a partial pickle for one method does
    not discard rows for another.

    `require: all` takes the intersection, so every method is scored on an
    identical peptide set -- what the separability comparison needs.
    """
    names = embeddings if embeddings is not None else cfg.selected_embeddings()
    slice_cfg = cfg.get("slice", {}) or {}
    trim_start = int(slice_cfg.get("trim_start", 0) or 0)
    trim_end = int(slice_cfg.get("trim_end", 0) or 0)

    df = load_epitopes(cfg)
    print(f"{len(df)} peptide rows after filters.")

    if trim_start or trim_end:
        print(f"Anchor trim: dropping {trim_start} residue(s) from the start and "
              f"{trim_end} from the end before pooling.")

    embed_cols: list[str] = []
    for name in names:
        col = attach_embedding(df, cfg.embedding_spec(name), trim_start, trim_end)
        if col is not None:
            embed_cols.append(col)

    if not embed_cols:
        raise SystemExit(
            "No embedding pickles found. Run the embedders first, or point "
            "configs/paths.yaml at the machine that holds them."
        )

    require = (cfg.get("require", "any") or "any").lower()
    if require not in {"any", "all"}:
        raise ValueError(f"require must be 'any' or 'all', got {require!r}")
    before = len(df)
    # dropna(how='all') drops a row only when every embedding is missing.
    df = df.dropna(subset=embed_cols, how="all" if require == "any" else "any")
    if require == "all":
        print(f"Common set: {len(df)} / {before} rows present in all "
              f"{len(embed_cols)} embeddings.")
    else:
        print(f"{len(df)} / {before} rows have at least one of "
              f"{len(embed_cols)} embeddings.")

    tax = load_taxonomy(cfg)
    merged = tax.merge(
        df[MERGE_COLS + ["protein_id", "protein_sequence"] + embed_cols],
        on=MERGE_COLS, how="inner",
    )
    print(f"{len(merged)} rows after the taxonomy merge.\n")
    return merged, embed_cols
