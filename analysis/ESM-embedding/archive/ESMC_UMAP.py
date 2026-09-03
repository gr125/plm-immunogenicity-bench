#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tqdm
import torch
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Import evolutionaryscale ESM library components
from huggingface_hub import login
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

# 1. Load ESMC model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Log in via huggingface-cli for access to the weights
login("")

# Note: Running small model
model = ESMC.from_pretrained("esmc-300m").to(device)
model.eval()  # disables dropout for deterministic results

# 2. Prepare data 
path = '/mnt/bioadhoc/Groups/Peters/Self-similarity/'
df = pd.read_csv(path + 'data/subset_data.csv')

df['slen'] = df['sequence'].str.len()
df['plen'] = df['protein_sequence'].str.len()

df = df.loc[df['plen'] < 2500]
df = df.loc[df['slen'] == 9]

df['id'] = df['sequence'] + '_' + df['type'] + '_' + df['species'] 

# Keeping the id, peptide sequence, and protein sequence for easier reference
data = list(df[['id', 'sequence', 'protein_sequence']].itertuples(index=False, name=None))

token_representations   = []
protein_representations = []
peptide_representations = []

# 3. Generate Embeddings
# Iterating one by one is recommended for 6B to avoid GPU OOM on long sequences
for tup in tqdm.tqdm(data):
    seq_id, peptide_seq, protein_seq = tup
    
    # Create the protein object for the new ESM API
    protein = ESMProtein(sequence=protein_seq)
    protein_tensor = model.encode(protein)
    
    # Extract representations
    with torch.no_grad():
        logits_output = model.logits(
            protein_tensor, 
            LogitsConfig(sequence=True, return_embeddings=True)
        )
    
    # The embeddings are returned in shape [1, seq_len, hidden_dim]
    # Move to CPU immediately to prevent GPU VRAM from overflowing
    token_rep = logits_output.embeddings[0].cpu() 
    
    prot_len = len(protein_seq)
    
    # In ESMC, token 0 is <BOS>. We extract just the amino acid tokens:
    aa_token_rep = token_rep[1 : prot_len + 1]
    
    token_representations.append(aa_token_rep)
    
    # Get protein representation average
    protein_representations.append(aa_token_rep.mean(0))
    
    # Get peptide representation (flattened)
    pep_len = len(peptide_seq)
    pep_start = protein_seq.find(peptide_seq)
    
    if pep_start != -1:
        # Since aa_token_rep is already stripped of <BOS>, pep_start is the exact index
        peptide_representations.append(aa_token_rep[pep_start : pep_start + pep_len].flatten())
    else:
        # Fallback if peptide not found (to maintain list order/shape)
        peptide_representations.append(torch.zeros(pep_len * aa_token_rep.shape[-1]))

# Save representations (Fixed: distinguished the filenames so they don't overwrite)
np.save(path + 'analysis/ESM-embedding/protein_representations.npy', torch.stack(protein_representations).numpy())
np.save(path + 'analysis/ESM-embedding/peptide_representations.npy', torch.stack(peptide_representations).numpy())

# Clear heavy lists and models from memory before UMAP
del model, data, token_representations

#%%  

# 4. UMAP 

# Fixed: List of tensors cannot use .detach().numpy() directly. Stacking them first.
numpy_tf = torch.stack(peptide_representations).numpy()
df_tf    = pd.DataFrame(numpy_tf)
df_tf    = StandardScaler().fit_transform(df_tf.values)

# Get 2D umap dimensions
reducer = umap.UMAP()
embedding = reducer.fit_transform(df_tf)

df_umap = pd.DataFrame(embedding, columns=["X","Y"])
df_umap["id"] = df['id'].values

df_umap.to_csv(path + 'analysis/ESM-embedding/umap.csv', index=None)

del numpy_tf, df_tf, reducer, embedding, peptide_representations, protein_representations

#%%  

# 5. Plot

df_umap.drop_duplicates(inplace=True)
df_umap = df_umap.merge(df, on='id')

sns.set_theme(rc={'figure.figsize':(40, 40)})

g = sns.jointplot(
    data=df_umap[(df_umap['type']=='infectious')|(df_umap['type']=='autoimmune')|(df_umap['type']=='Thymus MHCLE')], 
    x="X", y="Y", hue="type"
)
g.plot_joint(sns.kdeplot, color="r")
g.plot_marginals(sns.rugplot, color="r", clip_on=True)

plt.legend(bbox_to_anchor=(1.05, 1.2), loc=2, borderaxespad=0.)

plt.savefig(path + 'analysis/ESM-embedding/umap.png')