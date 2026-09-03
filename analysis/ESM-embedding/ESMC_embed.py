#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: grenjith

Embed the proteins in data/protein_sequences.csv with ESM Cambrian (ESMC) and
save per-residue token representations to a pickle file.

The output is a dict {protein_sequence: np.ndarray of shape [prot_len, D]} with
the <BOS>/<EOS> tokens stripped, matching the format of
    analysis/ProtBert/results/data/token_representations.pkl
so it can be loaded directly in analysis/downstream_umap.py, mapped onto
data['protein_sequence'], and sliced by residue in compute_peptide_embedding().

Embeddings are generated in GPU batches: sequences are sorted by length and
grouped so that (batch_size * padded_length) stays under a token budget, which
keeps padding waste low and the GPU saturated.
"""

import os
import re
import pickle

import pandas as pd
import numpy as np
import tqdm
import torch

# 1. SETUP LOCAL PATHS & OFFLINE MODE
# Directory holding the HuggingFace cache copied from a machine with internet.
# It must contain a "hub/models--EvolutionaryScale--esmc-300m-2024-12" subdir.
hf_home = "/mnt/bioadhoc/Groups/Peters/Self-similarity/analysis/ESM-embedding/hf_home"
os.environ['HF_HOME'] = hf_home

# Tell the library to NEVER look at the internet
os.environ['HF_HUB_OFFLINE'] = '1'

# Import evolutionaryscale ESM library components
from esm.models.esmc import ESMC

# 2. LOAD LOCAL MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# from_pretrained takes the REGISTRY NAME (not a path); it resolves the weights
# from the offline HF cache under HF_HOME set above.
print(f"Loading esmc_300m from offline HF cache at: {hf_home}")
try:
    model = ESMC.from_pretrained("esmc_300m").to(device)
    print("Model successfully loaded from local file.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure the directory contains the necessary config files alongside the .pth file.")
    exit()

model.eval()

# 3. PREPARE DATA
path = '/mnt/bioadhoc/Groups/Peters/Self-similarity/'

df = pd.read_csv(path + 'data/protein_sequences.csv')
df = df.dropna(subset=['protein_sequence'])
df['plen'] = df['protein_sequence'].str.len()
df = df.loc[df['plen'] < 7000]
# Sort by length so each batch pads to a similar length (minimal padding waste)
df = df.sort_values(by='plen').reset_index(drop=True)
# Deduplicate: token_representations is keyed on the sequence itself
df = df.drop_duplicates(subset=['protein_sequence']).reset_index(drop=True)
print(f"Data preprocessing complete. {len(df)} unique proteins to embed.")

# 4. BATCHING HYPERPARAMETERS  (tune for your GPU)
MAX_TOKENS_PER_BATCH = 120000  # batch_size * padded_len budget
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 512
CHECKPOINT_EVERY = 1000        # proteins between checkpoint saves

# 5. OUTPUT + RESUME SUPPORT
out_dir = os.path.join(path, 'analysis/ESM-embedding/results/data/')
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, 'token_representations.pkl')
tmp = os.path.join(out_dir, 'token_representations.pkl.tmp')

token_representations = dict()
if os.path.isfile(out_file):
    with open(out_file, "rb") as f:
        token_representations = pickle.load(f)
    print(f"Resuming: {len(token_representations)} proteins already embedded.")

# Drop already-embedded proteins, keep length order
df = df.loc[~df['protein_sequence'].isin(token_representations)].reset_index(drop=True)
print(f"{len(df)} proteins remaining to embed.")

orig_seqs = df['protein_sequence'].values
# ESMC expects the standard amino-acid alphabet; map non-standard residues to X
clean_seqs = np.array([re.sub(r"[UZOB]", "X", s) for s in orig_seqs], dtype=object)
plens = df['plen'].values


def make_batches(lengths, max_tokens, min_bs, max_bs):
    """Greedy contiguous batching over length-sorted data.

    +2 accounts for the <BOS>/<EOS> tokens added during tokenization.
    """
    batches = []
    i = 0
    n = len(lengths)
    while i < n:
        max_len = lengths[i] + 2
        bs = 1
        while (i + bs) < n and bs < max_bs:
            cand_max = max(max_len, lengths[i + bs] + 2)
            if bs + 1 > min_bs and (bs + 1) * cand_max > max_tokens:
                break
            max_len = cand_max
            bs += 1
        batches.append((i, i + bs))
        i += bs
    return batches


def embed_batch(seq_list):
    """Run ESMC on a list of (cleaned) sequences -> list of [len, D] float32 arrays."""
    # _tokenize adds <BOS>/<EOS>, pads to the longest sequence, and moves to device
    tokens = model._tokenize(list(seq_list))
    with torch.no_grad():
        # Default sequence_id masks pad tokens, so padding does not leak across attention
        out = model(sequence_tokens=tokens)
    emb = out.embeddings  # [B, Lmax, D]
    reps = []
    for k, s in enumerate(seq_list):
        L = len(s)
        reps.append(emb[k, 1:L + 1].detach().cpu().numpy().astype(np.float32))
    return reps


# 6. GENERATE EMBEDDINGS
batches = make_batches(plens, MAX_TOKENS_PER_BATCH, MIN_BATCH_SIZE, MAX_BATCH_SIZE)
print(f"{len(batches)} batches.")

since_checkpoint = 0
pbar = tqdm.tqdm(total=len(orig_seqs))
for (start, end) in batches:
    batch_orig = orig_seqs[start:end]
    batch_clean = clean_seqs[start:end]

    try:
        reps = embed_batch(batch_clean)
    except torch.cuda.OutOfMemoryError:
        print(f"\n[OOM] batch of {end - start} (padded len {plens[end - 1] + 2}) "
              f"-> falling back to single-sequence mode. Total OOMs: {n_oom}. "
              f"Consider lowering MAX_TOKENS_PER_BATCH/MAX_BATCH_SIZE.")
        exit() 
        # Fall back to one-at-a-time for this batch, then continue
        torch.cuda.empty_cache()
        reps = []
        for s in batch_clean:
            reps.extend(embed_batch([s]))
            torch.cuda.empty_cache()

    for orig_seq, rep in zip(batch_orig, reps):
        assert rep.shape[0] == len(orig_seq), \
            f"length mismatch: {rep.shape[0]} vs {len(orig_seq)}"
        token_representations[orig_seq] = rep

    n = end - start
    pbar.update(n)
    since_checkpoint += n
    if since_checkpoint >= CHECKPOINT_EVERY:
        with open(tmp, "wb") as f:
            pickle.dump(token_representations, f)
        os.replace(tmp, out_file)
        since_checkpoint = 0
pbar.close()

# 7. SAVE REPRESENTATIONS
with open(tmp, "wb") as f:
    pickle.dump(token_representations, f)
os.replace(tmp, out_file)
print(f"Token representations generated: {len(token_representations)} proteins -> {out_file}")
