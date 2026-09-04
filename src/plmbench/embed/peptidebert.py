#!/usr/bin/env python3
"""PeptideBERT embeddings from the three fine-tuned checkpoints.

Replaces PeptideBERT_embed.py, which took the checkpoint name as argv[1], hard-
coded `./checkpoints/<name>/`, and had to be run from inside the vendored fork.
The checkpoint now comes from the `job.checkpoint` key in
configs/embeddings.yaml, so the embedding name selects the model:

    python -m plmbench.embed.peptidebert pepbert_sol pepbert_nf pepbert_hemo

Unlike the ProtBert and ESMC embedders, this writes **pooled 1-D vectors**
({peptide: np.ndarray [D]}), matching `kind: vector` in the registry -- the
fine-tuned models are peptide-level classifiers and the token arrays were never
used downstream. Resume and checkpointing are shared with the other two.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..config import Config
from .base import resolve, run_job

# PeptideBERT's own vocabulary, from the vendored fork's dataloader.
AA_MAPPING = dict(zip(
    ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'L',
     'A', 'G', 'V', 'E', 'S', 'I', 'K', 'R', 'D', 'T', 'P', 'N',
     'Q', 'F', 'Y', 'M', 'H', 'C', 'W'],
    range(30),
))


def load_model(cfg: Config, checkpoint: str):
    """Load one fine-tuned checkpoint from models.peptidebert_checkpoints."""
    import torch
    import yaml

    ckpt_root = cfg.path("models.peptidebert_checkpoints")
    ckpt_dir = ckpt_root / checkpoint
    if not ckpt_dir.is_dir():
        available = sorted(p.name for p in ckpt_root.glob("*")) if ckpt_root.is_dir() else []
        raise SystemExit(
            f"checkpoint {checkpoint!r} not found under {ckpt_root}. "
            f"Available: {', '.join(available) or 'none'}"
        )

    # create_model lives in the vendored fork, which is not an installed package
    if str(ckpt_root.parent) not in sys.path:
        sys.path.insert(0, str(ckpt_root.parent))
    from model.network import create_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(ckpt_dir / "config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["device"] = device

    model = create_model(config)
    state = torch.load(ckpt_dir / "model.pt", map_location=device)["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded {checkpoint} on {device}")
    return model, device


def make_generator(model, device, pool: str = "mean"):
    """Yield (peptide, [D] float32) pooled vectors, one peptide at a time."""
    import torch

    def generate(seqs, job):
        for pep in seqs:
            unknown = set(pep) - set(AA_MAPPING)
            if unknown:
                print(f"  skipping {pep!r}: residues outside the vocabulary: "
                      f"{''.join(sorted(unknown))}")
                continue
            input_ids = torch.tensor([[AA_MAPPING[c] for c in pep]]).to(device)
            attention_mask = (input_ids != 0).float()
            with torch.no_grad():
                tokens = model.protbert(input_ids, attention_mask).last_hidden_state
            arr = tokens.cpu().numpy()[0]
            vec = arr.mean(axis=0) if pool == "mean" else arr.flatten()
            yield pep, vec.astype(np.float32).flatten()

    return generate


def main(argv=None) -> int:
    cfg, names = resolve(argv, "Embed peptides with a fine-tuned PeptideBERT.")
    for name in names:
        job = cfg.embedding_spec(name).get("job", {})
        checkpoint = job.get("checkpoint")
        if not checkpoint:
            raise SystemExit(f"{name!r} has no job.checkpoint in configs/embeddings.yaml")
        model, device = load_model(cfg, checkpoint)
        run_job(cfg, name, make_generator(model, device, job.get("pool", "mean")),
                checkpoint_every_default=5000)
    return 0


if __name__ == "__main__":
    sys.exit(main())
