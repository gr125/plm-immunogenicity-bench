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
from transformers import BertModel, BertTokenizer
import re
import sys

# Load ProtBert model
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
model = BertModel.from_pretrained("Rostlab/prot_bert")
print("Model loaded.")


# Prepare data 

"""
TODO

IMPROVE DATA COLLECTION
"""

path = '/mnt/bioadhoc/Groups/Peters/Self-similarity/'

df = pd.read_csv(path + 'data/subset_data.csv')
df['slen'] = df['sequence'].str.len()
df['plen'] = df['protein_sequence'].str.len()
df = df[~df.apply(lambda row: str(row['sequence']) not in str(row['protein_sequence']), axis=1)] # remove rows w peptide that is not in protein

#df = df.loc[df['plen'] < 2500]
df = df.loc[df['plen'] < 2500]
df = df.loc[df['slen'] == 9]

"""
TODO

IMPROVE ID
"""

df['id'] = df['sequence'] + '_' + df['type'] + '_' + df['species'] 
data = list(df[['id', 'protein_sequence']].itertuples(index=False, name=None))

print("Data preprocessing complete.")
# data = [
#     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
# ]
# peptide = "AEELSVSRQVIVQDIAYLRSLGYNIVA"
    

"""
TODO

STORE DATA 
RESTART FROM LAST ELEMENT
"""

token_representations   = []
protein_representations = []
peptide_representations = []

prev_pep_len = None
# TODO: many peptides have the same protein sequence, avoid recalculating embeddings
for tup in tqdm.tqdm(data):
    protein = tup[1]
    protein_id = tup[0]
    # print(F"Protein: {protein}")  
    print(F"Protein ID: {protein_id}")  
    print(F"Protein length: {len(protein)}")  
    encoded_input = tokenizer(re.sub(r"[UZOB]", "X", " ".join(protein)), return_tensors='pt')
    token_representation = model(**encoded_input).last_hidden_state[0,1:-1].detach().numpy()
    print(F"Token Representation shape: {token_representation.shape}")  
    token_representations.append(token_representation)
    protein_representations.append(token_representation.mean(0))
    print(F"Protein Representation shape: {protein_representations[-1].shape}")  

    peptide = df.loc[df['id'] == protein_id]['sequence'].values[0]
    # print(F"Peptide: {peptide}")  
    print(F"Peptide length: {len(peptide)}")  
    pep_start = protein.find(peptide)
        
    peptide_representations.append(token_representation[pep_start : pep_start + len(peptide) ].flatten())
    print(F"Peptide Representation shape: {peptide_representations[-1].shape}")  
sys.exit()

del(tup, token_representation, protein, protein_id, peptide, pep_start )

print(set([x.shape for x in peptide_representations]))

np.save(path + 'analysis/ProtBert/protein_representations', protein_representations)
np.save(path + 'analysis/ProtBert/peptide_representations', peptide_representations)
print("Protein and peptide representations generated.")

del(token_representations, data)

#%%  

# UMAP 

# format embeddings to pandas
numpy_tf = peptide_representations
#numpy_tf = protein_representations
df_tf    = pd.DataFrame(numpy_tf)
df_tf    = StandardScaler().fit_transform(df_tf.values)

# get 2 with umap dimensions
reducer = umap.UMAP()

embedding = reducer.fit_transform(df_tf)

df_umap = pd.DataFrame(embedding, columns=["X","Y"])
df_umap["id"] = df['id'].values

df_umap.to_csv(path + 'analysis/ProtBert/umap.csv', index=None)
print("UMAP results generated.")

del(numpy_tf, df_tf, reducer, embedding)

#%%  

# plot

#sns.set_theme(rc={'figure.figsize':(20, 20)})

df_umap.drop_duplicates(inplace=True)

df_umap = df_umap.merge(df, on='id')

#%%  

# for disease in df_umap['type'].unique():

#     plot = sns.scatterplot(data=df_umap.loc[df_umap['type'] == disease], x='X', y='Y', hue='type', marker='o')
#     plot.fig.clf()
    
#     # plt.xlim(-10, 10)
#     # plt.ylim(-5, 25)

#%%  

sns.set_theme(rc={'figure.figsize':(40, 40)})

g = sns.jointplot(data=df_umap[(df_umap['type']=='infectious')|(df_umap['type']=='autoimmune')|(df_umap['type']=='Thymus MHCLE')], x="X", y="Y", hue="type")
g.plot_joint(sns.kdeplot, color="r")
g.plot_marginals(sns.rugplot, color="r", clip_on=True)

plt.legend(bbox_to_anchor=(1.05, 1.2), loc=2, borderaxespad=0.)

#%%  
plt.savefig(path + 'analysis/ProtBert/plot.png')
print("UMAP plot saved.")