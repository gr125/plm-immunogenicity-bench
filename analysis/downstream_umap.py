#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: grenjith

Adapted from ESM2_UMAP.py
"""

import pandas as pd
import numpy as np
import tqdm
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

path = '/mnt/bioadhoc/Groups/Peters/Self-similarity/'

def compute_peptide_embedding(row):
        prot_emb = row["protbert_embed"]
        start = int(row["curated_source_antigen_start"])-1
        end = int(row["curated_source_antigen_end"])
        assert(end-start==row['slen'])
        slice_emb = prot_emb[start:end]             # shape = [len_pep, D]
        return slice_emb.mean(axis=0)               # pooled → shape [D]

def load_all_data():
    data_ug = pd.read_csv(path + 'analysis/data_ug_tax.csv')

    data = pd.read_csv(path + 'data/data.csv')
    data['slen'] = data['linear_sequence'].str.len()
    data['plen'] = data['protein_sequence'].str.len()
    data = data.loc[data['plen'] < 7000]

    with open(path+f'analysis/PeptideBERT/sol-0827_2316_peptide_representations.pkl', "rb") as f:
        pepbert_sol_pep_embed = pickle.load(f)

    data["pepbert_sol_pep_embed"] = data["linear_sequence"].map(pepbert_sol_pep_embed)

    with open(path+f'analysis/PeptideBERT/hemo-0828_0039_peptide_representations.pkl', "rb") as f:
        pepbert_hemo_pep_embed = pickle.load(f)

    data["pepbert_hemo_pep_embed"] = data["linear_sequence"].map(pepbert_hemo_pep_embed)

    with open(path+f'analysis/PeptideBERT/nf-0828_0049_peptide_representations.pkl', "rb") as f:
        pepbert_nf_pep_embed = pickle.load(f)

    data["pepbert_nf_pep_embed"] = data["linear_sequence"].map(pepbert_nf_pep_embed)

    print("Pepbert embeds loaded")
    del pepbert_sol_pep_embed, pepbert_hemo_pep_embed, pepbert_nf_pep_embed

    with open(path+f'analysis/ProtBert/results/data/peptide_representations.pkl', "rb") as f:
        protbert_embed = pickle.load(f)
    data["protbert_embed"] = data["linear_sequence"].map(protbert_embed)

    def mean_pool(row):
        return row["protbert_embed"].mean(axis=0)               

    data["protbertpep_pep_embed"] = data.apply(mean_pool, axis=1)
    del data["protbert_embed"]

    with open(path+f'analysis/ProtBert/results/data/token_representations.pkl', "rb") as f:
        protbert_embed = pickle.load(f)
    data["protbert_embed"] = data["protein_sequence"].map(protbert_embed)
    data["protbert_embed"] = data["protbert_embed"].astype(object)

    data["protbert_pep_embed"] = data.apply(compute_peptide_embedding, axis=1)
    print("protbert embeds loaded")

    del protbert_embed, data["protbert_embed"]


    merge_cols = [
        "linear_sequence",
        "mhc_class",
        "mhc_restriction",
        "curated_source_antigen_start",
        "curated_source_antigen_end",
    ]

    data_merged = data_ug.merge(
        data[merge_cols + ["protein_sequence", "protbertpep_pep_embed", "protbert_pep_embed", "pepbert_nf_pep_embed", "pepbert_sol_pep_embed", "pepbert_hemo_pep_embed"]],
        on=merge_cols,
        how="inner"
    )
    data_merged = data_merged.dropna(subset=["protein_sequence", "protbertpep_pep_embed", "protbert_pep_embed", "pepbert_nf_pep_embed", "pepbert_sol_pep_embed", "pepbert_hemo_pep_embed"])
    print(data_merged)
    return data_merged

#%%  
if __name__ == '__main__':
    data_merged = load_all_data()
    # UMAP 
    for embed_type in ["protbert", "protbertpep", "pepbert_nf", "pepbert_sol", "pepbert_hemo"]:
    #for embed_type in ["protbert", "pepbert_nf"]:
    #for embed_type in ["protbertpep", "pepbert_hemo"]:
        if os.path.isfile(path +f'analysis/results/umap/umap_{embed_type}.csv'):
            data_merged = pd.read_csv(path +f'analysis/results/umap/umap_{embed_type}.csv')
            print("UMAP results loaded.")
        else:
            X = np.stack(data_merged[embed_type+'_pep_embed'].values, axis=0)
            df_tf    = pd.DataFrame(X)
            df_tf    = StandardScaler().fit_transform(df_tf.values)

            reducer = umap.UMAP(init='random')

            embedding = reducer.fit_transform(df_tf)

            df_umap = pd.DataFrame(embedding, columns=["X","Y"])
            data_merged["X"] = df_umap["X"]
            data_merged["Y"] = df_umap["Y"]

            data_merged.to_csv(path + f'analysis/results/umap/umap_{embed_type}.csv', index=None)
            print("UMAP results generated.")

            del(df_tf, reducer, embedding)

        sns.set_theme(rc={'figure.figsize':(40, 40)})

        g = sns.jointplot(data=data_merged, x="X", y="Y", hue="tax_domain")
        g.plot_joint(sns.kdeplot, color="r")
        g.plot_marginals(sns.rugplot, color="r", clip_on=True)

        plt.legend(bbox_to_anchor=(1.05, 1.2), loc=2, borderaxespad=0.)

        #%%  
        plt.savefig(path + f'analysis/plot_{embed_type}_domains.png')
        plt.close()
        print("Domain UMAP plot saved.")

        for dom in ['Viruses', 'Eukaryota', 'Bacteria']:
            sns.set_theme(rc={'figure.figsize':(40, 40)})
            d = data_merged[data_merged['tax_domain']==dom].dropna(subset=['tax_family'])
            g = sns.jointplot(data=d, x="X", y="Y", hue="tax_family")
            g.plot_joint(sns.kdeplot, color="r")
            g.plot_marginals(sns.rugplot, color="r", clip_on=True)

            plt.legend(bbox_to_anchor=(1.05, 1.2), loc=2, borderaxespad=0.)

            #%%  
            plt.savefig(path + f'analysis/plot_{embed_type}_{dom.lower()}_family.png')
            plt.close()
        print("Family UMAP plots saved.")

        sns.set_theme(rc={'figure.figsize':(40, 40)})

        g = sns.jointplot(data=data_merged.dropna(subset=['immune_response']), x="X", y="Y", hue="immune_response")
        g.plot_joint(sns.kdeplot, color="r")
        g.plot_marginals(sns.rugplot, color="r", clip_on=True)

        plt.legend(bbox_to_anchor=(1.05, 1.2), loc=2, borderaxespad=0.)

        #%%  
        plt.savefig(path + f'analysis/plot_{embed_type}_immuneresponse.png')
        plt.close()
        print("Immune response UMAP plot saved.")

        for dom in ['Viruses', 'Eukaryota', 'Bacteria']:
            sns.set_theme(rc={'figure.figsize':(40, 40)})
            d = data_merged[data_merged['tax_domain']==dom].dropna(subset=['immune_response'])
            g = sns.jointplot(data=d, x="X", y="Y", hue="immune_response")
            g.plot_joint(sns.kdeplot, color="r")
            g.plot_marginals(sns.rugplot, color="r", clip_on=True)

            plt.legend(bbox_to_anchor=(1.05, 1.2), loc=2, borderaxespad=0.)

            #%%  
            plt.savefig(path + f'analysis/plot_{embed_type}_{dom.lower()}_immuneresponse.png')
            plt.close()
        print("Immune response UMAP per domain plots saved.")

