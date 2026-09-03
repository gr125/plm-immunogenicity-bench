#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: grenjith

ESMC-only downstream UMAP, robust to PARTIAL embedding sets.

Handles two distinct ESMC representations (mirroring the protbert /
protbertpep split in downstream_umap.py):

  "esmc"     - peptide embedding sliced out of the PARENT PROTEIN's token
               representations, then mean-pooled. Context-aware.
               Source: ESM-embedding/results/data/token_representations.pkl
                       (keyed on protein_sequence, produced by ESMC_embed.py)

  "esmcpep"  - peptide embedded IN ISOLATION, then mean-pooled. No context.
               Source: ESM-embedding/results/data/peptide_representations.pkl
                       (keyed on linear_sequence, produced by ESMC_embed_peptide.py)

Each embedding type is filtered independently, so a partial (or missing)
pickle for one does not discard rows for the other. Safe to run while the
embedding jobs are still in progress.
"""

import os
import pickle

import pandas as pd
import numpy as np
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

path = '/mnt/bioadhoc/Groups/Peters/Self-similarity/'

# (possibly partial) ESMC pickles produced by the two embedding scripts
ESMC_PROT_PKL = path + 'analysis/ESM-embedding/results/data/token_representations.pkl'
ESMC_PEP_PKL = path + 'analysis/ESM-embedding/results/data/peptide_representations.pkl'

# Minimum points required in a hue group before we attempt a KDE overlay
MIN_KDE_POINTS = 10


def compute_esmc_peptide(row):
    """Slice the peptide out of the parent protein's token representations."""
    prot_emb = row["esmc_embed"]
    start = int(row["curated_source_antigen_start"]) - 1
    end = int(row["curated_source_antigen_end"])
    if end - start != row['slen']:
        # inconsistent antigen coordinates -> drop this peptide downstream
        return np.nan
    return prot_emb[start:end].mean(axis=0)          # pooled -> shape [D]


def load_all_data():
    data_ug = pd.read_csv(path + 'analysis/data_ug_tax.csv')

    data = pd.read_csv(path + 'data/data.csv')
    data['slen'] = data['linear_sequence'].str.len()
    data['plen'] = data['protein_sequence'].str.len()
    data = data.loc[data['plen'] < 7000]

    embed_cols = []

    # --- "esmc": protein token reps, sliced to the peptide ---
    if os.path.isfile(ESMC_PROT_PKL):
        with open(ESMC_PROT_PKL, "rb") as f:
            esmc_prot = pickle.load(f)
        print(f"ESMC protein pickle holds {len(esmc_prot)} proteins.")

        data["esmc_embed"] = data["protein_sequence"].map(esmc_prot)
        data["esmc_embed"] = data["esmc_embed"].astype(object)
        # only rows whose protein is embedded get a peptide embedding
        mask = data["esmc_embed"].notna()
        data["esmc_pep_embed"] = np.nan
        data["esmc_pep_embed"] = data["esmc_pep_embed"].astype(object)
        if mask.any():
            data.loc[mask, "esmc_pep_embed"] = data.loc[mask].apply(
                compute_esmc_peptide, axis=1
            )
        del data["esmc_embed"], esmc_prot
        embed_cols.append("esmc_pep_embed")
        print(f"  -> {data['esmc_pep_embed'].notna().sum()} peptides with a sliced embedding.")
    else:
        print(f"NOTE: {ESMC_PROT_PKL} not found; skipping 'esmc'.")

    # --- "esmcpep": peptide embedded in isolation ---
    if os.path.isfile(ESMC_PEP_PKL):
        with open(ESMC_PEP_PKL, "rb") as f:
            esmc_pep = pickle.load(f)
        print(f"ESMC peptide pickle holds {len(esmc_pep)} peptides.")

        tok = data["linear_sequence"].map(esmc_pep)
        data["esmcpep_pep_embed"] = [
            t.mean(axis=0) if isinstance(t, np.ndarray) else np.nan for t in tok
        ]
        del tok, esmc_pep
        embed_cols.append("esmcpep_pep_embed")
        print(f"  -> {data['esmcpep_pep_embed'].notna().sum()} peptides with an isolated embedding.")
    else:
        print(f"NOTE: {ESMC_PEP_PKL} not found; skipping 'esmcpep'.")

    if not embed_cols:
        raise SystemExit("No ESMC pickles found -- run the embedding scripts first.")

    # keep rows that have at least one embedding
    data = data.dropna(subset=embed_cols, how="all")

    merge_cols = [
        "linear_sequence",
        "mhc_class",
        "mhc_restriction",
        "curated_source_antigen_start",
        "curated_source_antigen_end",
    ]

    data_merged = data_ug.merge(
        data[merge_cols + ["protein_sequence"] + embed_cols],
        on=merge_cols,
        how="inner",
    )
    print(f"{len(data_merged)} peptides after merge with taxonomy table.")
    return data_merged, embed_cols


def jointplot(d, hue, out_png):
    """KDE-overlaid jointplot that degrades gracefully on tiny/degenerate groups."""
    d = d.dropna(subset=[hue])
    if len(d) < 3:
        print(f"  skip {os.path.basename(out_png)}: only {len(d)} points.")
        return
    sns.set_theme(rc={'figure.figsize': (40, 40)})
    g = sns.jointplot(data=d, x="X", y="Y", hue=hue)
    # Only overlay a KDE where every hue group is large enough to estimate one
    group_sizes = d.groupby(hue).size()
    if (group_sizes >= MIN_KDE_POINTS).all() and len(d) >= MIN_KDE_POINTS:
        try:
            g.plot_joint(sns.kdeplot, color="r")
            g.plot_marginals(sns.rugplot, color="r", clip_on=True)
        except Exception as e:
            print(f"  KDE skipped for {os.path.basename(out_png)}: {e}")
    plt.legend(bbox_to_anchor=(1.05, 1.2), loc=2, borderaxespad=0.)
    plt.savefig(out_png)
    plt.close()
    print(f"  saved {os.path.basename(out_png)}")


def run_umap(data_merged, embed_type):
    """UMAP + plots for one embedding type, using only rows that have it."""
    col = embed_type + '_pep_embed'
    os.makedirs(path + 'analysis/results/umap', exist_ok=True)
    umap_csv = path + f'analysis/results/umap/umap_{embed_type}.csv'

    if os.path.isfile(umap_csv):
        d = pd.read_csv(umap_csv)
        print(f"[{embed_type}] UMAP results loaded ({len(d)} rows).")
    else:
        d = data_merged.dropna(subset=[col]).reset_index(drop=True)
        if len(d) < 10:
            print(f"[{embed_type}] only {len(d)} embedded peptides -- skipping.")
            return
        print(f"[{embed_type}] running UMAP on {len(d)} peptides.")

        X = np.stack(d[col].values, axis=0)
        X = StandardScaler().fit_transform(pd.DataFrame(X).values)

        reducer = umap.UMAP(init='random')
        embedding = reducer.fit_transform(X)

        d["X"] = embedding[:, 0]
        d["Y"] = embedding[:, 1]
        d.drop(columns=[c for c in d.columns if c.endswith('_pep_embed')]).to_csv(
            umap_csv, index=None
        )
        print(f"[{embed_type}] UMAP results generated.")

    # Domains
    jointplot(d, "tax_domain", path + f'analysis/plot_{embed_type}_domains.png')

    # Families per domain
    for dom in ['Viruses', 'Eukaryota', 'Bacteria']:
        jointplot(d[d['tax_domain'] == dom], "tax_family",
                  path + f'analysis/plot_{embed_type}_{dom.lower()}_family.png')

    # Immune response (overall)
    jointplot(d, "immune_response",
              path + f'analysis/plot_{embed_type}_immuneresponse.png')

    # Immune response per domain
    for dom in ['Viruses', 'Eukaryota', 'Bacteria']:
        jointplot(d[d['tax_domain'] == dom], "immune_response",
                  path + f'analysis/plot_{embed_type}_{dom.lower()}_immuneresponse.png')


if __name__ == '__main__':
    data_merged, embed_cols = load_all_data()

    for embed_type in [c.replace('_pep_embed', '') for c in embed_cols]:
        run_umap(data_merged, embed_type)

    print("Done.")
