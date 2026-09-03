#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: grenjith

Embed the UNIQUE peptides in the 'linear_sequence' column of data/data.csv with
ESM Cambrian (ESMC) and save per-residue token representations to a pickle file.

The output is a dict {linear_sequence: np.ndarray of shape [pep_len, D]} with the
<BOS>/<EOS> tokens stripped, matching the format of
    analysis/ProtBert/results/data/peptide_representations.pkl
so downstream_umap.py can map it onto data['linear_sequence'] and mean-pool it
(the 'protbertpep_pep_embed' pattern).

NOTE: this embeds each peptide in isolation (no surrounding protein context),
which is exactly what the ProtBert peptide_representations.pkl does. For
context-aware peptide embeddings sliced out of the parent protein, use
ESMC_embed.py + compute_peptide_embedding instead.

Peptides are short (~9-25 aa), so batches are large and this runs far faster
than the whole-protein version.
"""

import os
import re
import pickle
import time

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

print(f"Loading esmc_300m from offline HF cache at: {hf_home}")
try:
    model = ESMC.from_pretrained("esmc_300m").to(device)
    print("Model successfully loaded from local file.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

model.eval()

# 3. PREPARE DATA
path = '/mnt/bioadhoc/Groups/Peters/Self-similarity/'

df = pd.read_csv(path + 'data/data.csv')
df = df.dropna(subset=['linear_sequence'])

# Unique peptides only -- token_representations is keyed on the sequence itself
peptides = pd.Series(df['linear_sequence'].unique())
plens = peptides.str.len()

# Sort by length so each batch pads to a similar length (minimal padding waste)
order = np.argsort(plens.values, kind='stable')
peptides = peptides.values[order]
plens = plens.values[order]
print(f"{len(peptides)} unique peptides, length {plens.min()}-{plens.max()}.")

# 4. BATCHING HYPERPARAMETERS  (peptides are short -> batch size dominates)
MAX_TOKENS_PER_BATCH = 120000   # batch_size * padded_len budget
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 1024
CHECKPOINT_EVERY = 5000        # peptides between checkpoint saves

# 5. OUTPUT + RESUME SUPPORT
out_dir = os.path.join(path, 'analysis/ESM-embedding/results/data/')
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, 'peptide_representations.pkl')
tmp_file = out_file + '.tmp'

peptide_representations = dict()
if os.path.isfile(out_file):
    with open(out_file, "rb") as f:
        peptide_representations = pickle.load(f)
    print(f"Resuming: {len(peptide_representations)} peptides already embedded.")

# Drop already-embedded peptides, keep length order
keep = np.array([p not in peptide_representations for p in peptides])
peptides = peptides[keep]
plens = plens[keep]
print(f"{len(peptides)} peptides remaining to embed.")

# ESMC expects the standard amino-acid alphabet; map non-standard residues to X
clean_peps = np.array([re.sub(r"[UZOB]", "X", p) for p in peptides], dtype=object)


def save_checkpoint(obj, final_path, temp_path):
    """Atomic write: dump to a temp file in the same dir, then rename."""
    with open(temp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(temp_path, final_path)


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
print(f"{len(batches)} batches "
      f"(avg {len(peptides) / max(len(batches), 1):.0f} peptides/batch).")

n_oom = 0
since_checkpoint = 0
t0 = time.time()
pbar = tqdm.tqdm(total=len(peptides))
for (start, end) in batches:
    batch_orig = peptides[start:end]
    batch_clean = clean_peps[start:end]

    try:
        reps = embed_batch(batch_clean)
    except torch.cuda.OutOfMemoryError:
        # Loud, not silent: repeated OOM means the token budget is too high and
        # the run has quietly degraded to one-sequence-at-a-time (very slow).
        n_oom += 1
        print(f"\n[OOM] batch of {end - start} (padded len {plens[end - 1] + 2}) "
              f"-> falling back to single-sequence mode. Total OOMs: {n_oom}. "
              f"Consider lowering MAX_TOKENS_PER_BATCH/MAX_BATCH_SIZE.")
        torch.cuda.empty_cache()
        reps = []
        for s in batch_clean:
            reps.extend(embed_batch([s]))
        torch.cuda.empty_cache()

    for pep, rep in zip(batch_orig, reps):
        assert rep.shape[0] == len(pep), \
            f"length mismatch: {rep.shape[0]} vs {len(pep)}"
        peptide_representations[pep] = rep

    n = end - start
    pbar.update(n)
    since_checkpoint += n
    if since_checkpoint >= CHECKPOINT_EVERY:
        save_checkpoint(peptide_representations, out_file, tmp_file)
        since_checkpoint = 0
pbar.close()

# 7. SAVE REPRESENTATIONS
save_checkpoint(peptide_representations, out_file, tmp_file)
print(f"Done in {time.time() - t0:.1f}s. OOM fallbacks: {n_oom}.")
print(f"Peptide representations generated: {len(peptide_representations)} peptides -> {out_file}")
