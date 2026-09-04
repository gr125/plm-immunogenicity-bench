"""Shared machinery for the three embedders.

The length-sorted token-budget batching, the atomic checkpointing and the
resume support all come from ESMC_embed.py, which was the only embedder here
that had them. ProtBert and PeptideBERT now sit on the same base, so all three
survive a scheduler kill and restart where they left off.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, add_config_args, config_from_args
from ..data.loader import load_epitopes
from ..io import CheckpointWriter, resume_dict


def make_batches(lengths, max_tokens: int, min_bs: int, max_bs: int):
    """Greedy contiguous batching over length-sorted data.

    Grows a batch while (batch_size * padded_length) stays under the token
    budget, so padding waste stays low and the GPU stays saturated. The +2
    accounts for the <BOS>/<EOS> tokens added during tokenization.
    """
    batches, i, n = [], 0, len(lengths)
    while i < n:
        max_len = lengths[i] + 2
        bs = 1
        while (i + bs) < n and bs < max_bs:
            cand = max(max_len, lengths[i + bs] + 2)
            if bs + 1 > min_bs and (bs + 1) * cand > max_tokens:
                break
            max_len = cand
            bs += 1
        batches.append((i, i + bs))
        i += bs
    return batches


def sequences_to_embed(cfg: Config, target: str) -> np.ndarray:
    """The unique sequences for a job, sorted by length (short first).

    target='proteins' -> unique antigen sequences, honouring
    filters.max_protein_length; 'peptides' -> unique linear_sequence values.
    Both are deduplicated because every pickle here is keyed on the sequence.
    """
    df = load_epitopes(cfg)
    if target == "proteins":
        seqs = df.dropna(subset=["protein_sequence"])["protein_sequence"]
    elif target == "peptides":
        seqs = df.dropna(subset=["linear_sequence"])["linear_sequence"]
    else:
        raise ValueError(f"target must be 'proteins' or 'peptides', got {target!r}")

    seqs = pd.Series(seqs.unique())
    lens = seqs.str.len()
    order = np.argsort(lens.values, kind="stable")
    out = seqs.values[order]
    print(f"{len(out)} unique {target}, length {lens.min()}-{lens.max()}.")
    return out


def pending(seqs: np.ndarray, store: dict) -> np.ndarray:
    """Drop already-embedded sequences, preserving length order."""
    todo = np.array([s for s in seqs if s not in store], dtype=object)
    print(f"{len(todo)} of {len(seqs)} still to embed.")
    return todo


def run_job(cfg: Config, name: str, embed_batches, checkpoint_every_default: int = 1000):
    """Drive one embedding job to completion, checkpointing as it goes.

    `embed_batches` is a generator taking the pending sequences and yielding
    (sequence, array) pairs. Everything else -- resume, checkpointing, the
    atomic final write -- happens here.
    """
    import tqdm

    spec = cfg.embedding_spec(name)
    job = spec.get("job", {})
    out_file = Path(spec["pickle"])

    store = resume_dict(out_file, label=job.get("target", "items"))
    seqs = pending(sequences_to_embed(cfg, job.get("target", "peptides")), store)
    if len(seqs) == 0:
        print(f"[{name}] nothing to do; {len(store)} already in {out_file}.")
        return store

    ckpt = CheckpointWriter(out_file, every=job.get("checkpoint_every",
                                                    checkpoint_every_default))
    pbar = tqdm.tqdm(total=len(seqs))
    try:
        for seq, rep in embed_batches(seqs, job):
            store[seq] = rep
            pbar.update(1)
            ckpt.tick(store, 1)
    finally:
        pbar.close()
        ckpt.flush(store)           # a killed job still leaves a complete file

    print(f"[{name}] {len(store)} sequences -> {out_file}")
    return store


def embedder_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_config_args(parser)
    parser.add_argument("embeddings", nargs="+", metavar="NAME",
                        help="registry entries to produce, e.g. protbert esmcpep")
    return parser


def resolve(argv, description: str) -> tuple[Config, list[str]]:
    args = embedder_parser(description).parse_args(argv)
    return config_from_args(args), args.embeddings
