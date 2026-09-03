#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:16:26 2025

@author: icarri
"""

import pandas as pd
import numpy as np
import tqdm
import esm
import torch
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load ESM-2 model

#model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#t = 33

model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
t = 6

batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Prepare data 

"""
TODO

IMPROVE DATA COLLECTION
"""

path = '/mnt/bioadhoc/Groups/Peters/Self-similarity/'

df = pd.read_csv(path + 'data/subset_data.csv')

df['slen'] = df['sequence'].str.len()
df['plen'] = df['protein_sequence'].str.len()

df = df.loc[df['plen'] < 2500]
df = df.loc[df['slen'] == 9]

"""
TODO

IMPROVE ID
"""

df['id'] = df['sequence'] + '_' + df['type'] + '_' + df['species'] 

data = list(df[['id', 'protein_sequence']].itertuples(index=False, name=None))

# data = [
#     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
# ]
    

"""
TODO

STORE DATA 
RESTART FROM LAST ELEMENT
"""

token_representations   = []
protein_representations = []
peptide_representations = []

for tup in tqdm.tqdm(data):

    batch_labels, batch_strs, batch_tokens = batch_converter([tup])
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    
    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[t], return_contacts=True)
    token_representation = results["representations"][t]
    
    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    for i, tokens_len in enumerate(batch_lens):
        
        # get protein representation
        token_representations.append(token_representation[0, 1 : tokens_len - 1])

        # get protein representation average
        protein_representations.append(token_representation[0, 1 : tokens_len - 1].mean(0))
        #protein_representations.append(token_representation[0, 1 : tokens_len - 1].flatten())
            
        # get peptide representation average
                
        protein = df.loc[df['id'] == batch_labels[0]]['protein_sequence'].values[0]
        peptide = df.loc[df['id'] == batch_labels[0]]['sequence'].values[0]
        
        pep_len = len(peptide)
        pep_start = protein.find(peptide)
        
        #peptide_representations.append(token_representation[0, 1 + pep_start : 1 + pep_start + pep_len ].mean(0))
        peptide_representations.append(token_representation[0, 1 + pep_start : 1 + pep_start + pep_len ].flatten())
        
del(tup, batch_labels, batch_strs, batch_tokens, batch_lens, alphabet, results, token_representation, i, tokens_len, protein, peptide, pep_len, pep_start )

#np.save('/Users/icarri/Documents/Posdoc_LJI/Projects/Self-similarity/analysis/ESM_embedding/sequence_representations', token_representations)
np.save(path + 'analysis/ESM_embedding/sequence_representations', protein_representations)
np.save(path + 'analysis/ESM_embedding/sequence_representations', peptide_representations)

del(token_representations, t, data)

#%%  

# UMAP 

# format embeddings to pandas
numpy_tf = peptide_representations.detach().numpy()
#numpy_tf = [tensor.numpy() for tensor in protein_representations]
df_tf    = pd.DataFrame(numpy_tf)
df_tf    = StandardScaler().fit_transform(df_tf.values)

# get 2 with umap dimensions
reducer = umap.UMAP()

embedding = reducer.fit_transform(df_tf)

df_umap = pd.DataFrame(embedding, columns=["X","Y"])
df_umap["id"] = df['id'].values

df_umap.to_csv(path + 'analysis/ESM_embedding/umap.csv', index=None)

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

plt.savefig(path + 'analysis/ESM_embedding/umap.png')
#%%  
