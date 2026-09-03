#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: grenjith

Adapted from ESM2_UMAP.py
"""

import pandas as pd
print("pandas imported")
import numpy as np
print("numpy imported")
from sklearn.preprocessing import StandardScaler
print("sklearn imported")
import torch
print("torch imported")
from transformers import BertModel, BertTokenizer
print("transformers imported")
import re
print("re imported")
import sys
print("sys imported")
import gc
print("gc imported")
import pickle

# Load ProtBert model
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
model = BertModel.from_pretrained("Rostlab/prot_bert")
model.eval().to('cuda' if torch.cuda.is_available() else 'cpu')
print('cuda' if torch.cuda.is_available() else 'cpu')
print("Model loaded.")


# Prepare data 

"""
TODO

IMPROVE DATA COLLECTION
"""

path = '/mnt/bioadhoc/Groups/Peters/Self-similarity/'

df = pd.read_csv(path + 'data/protein_sequences.csv')
#df['slen'] = df['sequence'].str.len()
df = df.dropna(subset=['protein_sequence'])
df['plen'] = df['protein_sequence'].str.len()
#df = df[~df.apply(lambda row: str(row['sequence']) not in str(row['protein_sequence']), axis=1)] # remove rows w peptide that is not in protein

#df = df.loc[df['plen'] < 2500]
df = df.loc[df['plen'] < 7000]
#df = df.loc[df['slen'] == 9]
df = df.sort_values(by='plen').reset_index(drop=True)
"""
TODO

IMPROVE ID
"""

#df['id'] = df['sequence'] + '_' + df['type'] + '_' + df['species'] 

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

# hyperparameters
MAX_TOKENS_PER_BATCH = 120000  # tune for 48 GB GPU
MIN_BATCH_SIZE = 2
MAX_BATCH_SIZE = 1024


token_representations = dict()

i = 0
print(len(df['plen']))
while i < len(df['plen']):
    print(df['plen'][i])
    batch_size = int(max(MIN_BATCH_SIZE, min(MAX_BATCH_SIZE, MAX_TOKENS_PER_BATCH // df['plen'][i])))

    #batch_ids = ids[i : i + est_batch_size] 
    try:    
        batch_seqs = tokenizer([re.sub(r"[UZOB]", "X", " ".join(x)) for x in df['protein_sequence'][i:i+batch_size]], 
            return_tensors='pt', padding='longest').to('cuda' if torch.cuda.is_available() else 'cpu') 
    except:
        print(df['protein_sequence'][i:i+batch_size])
        sys.exit(1)
    # print(batch_seqs) \
    # has CLS token and also padding tokens at the end
    try:
        with torch.no_grad():
            out = model(**batch_seqs).last_hidden_state.detach().cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        continue
    print(out.shape)
    del batch_seqs
    for j in range(out.shape[0]):
        token_representations[df['protein_sequence'][i+j]] = out[j,1:-1]
    del out
    i += batch_size

with open(path+f'analysis/ProtBert/token_representations.pkl', "wb") as f:
    pickle.dump(token_representations, f)
print("Token representations generated.")
