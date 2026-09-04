#!/usr/bin/env python3
"""ESMC-300M embeddings, config-driven.

Replaces ESMC_embed.py (proteins) and ESMC_embed_peptides.py (peptides), which
differed only in which column they read and which pickle they wrote -- both now
`target:` in configs/embeddings.yaml.

Output is {sequence: np.ndarray [len, D]} with <BOS>/<EOS> stripped, so it maps
straight onto data['protein_sequence'] or data['linear_sequence'] downstream.

    python -m plmbench.embed.esmc esmc esmcpep
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np

from ..config import Config
from .base import make_batches, resolve, run_job


def load_model(cfg: Config):
    """Load ESMC from the offline HF cache named in configs/paths.yaml."""
    import torch

    hf_home = cfg.path("models.hf_home")
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_OFFLINE"] = "1"   # never reach for the network on a compute node

    from esm.models.esmc import ESMC     # imported after HF_HOME is set

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registry = cfg.paths["models"].get("esmc_registry_name", "esmc_300m")
    print(f"Loading {registry} on {device} from offline HF cache at {hf_home}")
    if not hf_home.is_dir():
        raise SystemExit(
            f"HF cache not found at {hf_home}. Set models.hf_home in "
            f"configs/paths.yaml, or export PLMBENCH_ROOT."
        )
    # from_pretrained takes the REGISTRY NAME, not a path; it resolves the
    # weights out of the cache under HF_HOME.
    model = ESMC.from_pretrained(registry).to(device)
    model.eval()
    return model, device


def make_generator(model):
    """Yield (sequence, [len, D] float32) pairs, batching by token budget."""
    import torch

    def embed_batch(seq_list):
        tokens = model._tokenize(list(seq_list))   # adds BOS/EOS, pads, moves to device
        with torch.no_grad():
            # the default sequence_id masks pads, so padding cannot leak across attention
            out = model(sequence_tokens=tokens)
        emb = out.embeddings                       # [B, Lmax, D]
        return [emb[k, 1:len(s) + 1].detach().cpu().numpy().astype(np.float32)
                for k, s in enumerate(seq_list)]

    def generate(seqs, job):
        lengths = np.array([len(s) for s in seqs])
        # ESMC expects the standard alphabet; map non-standard residues to X
        clean = np.array([re.sub(r"[UZOB]", "X", s) for s in seqs], dtype=object)

        batches = make_batches(lengths,
                               job.get("max_tokens_per_batch", 120000),
                               job.get("min_batch_size", 1),
                               job.get("max_batch_size", 512))
        print(f"{len(batches)} batches.")

        for start, end in batches:
            try:
                reps = embed_batch(clean[start:end])
            except torch.cuda.OutOfMemoryError:
                # Fall back to one-at-a-time for this batch rather than dying.
                # The original script printed an undefined variable and exited
                # here, so the fallback below was unreachable.
                print(f"\n[OOM] batch of {end - start} (padded len "
                      f"{lengths[end - 1] + 2}); falling back to single-sequence "
                      f"mode. Consider lowering max_tokens_per_batch.")
                torch.cuda.empty_cache()
                reps = []
                for s in clean[start:end]:
                    reps.extend(embed_batch([s]))
                    torch.cuda.empty_cache()

            for seq, rep in zip(seqs[start:end], reps):
                assert rep.shape[0] == len(seq), \
                    f"length mismatch: {rep.shape[0]} vs {len(seq)}"
                yield seq, rep

    return generate


def main(argv=None) -> int:
    cfg, names = resolve(argv, "Embed proteins or peptides with ESMC-300M.")
    model, _ = load_model(cfg)
    generate = make_generator(model)
    for name in names:
        run_job(cfg, name, generate, checkpoint_every_default=1000)
    return 0


if __name__ == "__main__":
    sys.exit(main())
