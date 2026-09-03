#!/usr/bin/env python3
"""ProtBert (Rostlab/prot_bert) embeddings, config-driven.

Replaces ProtBert_embed.py (proteins) and ProtBert_embed_peptide.py (peptides),
which differed only in the column read and the pickle written.

Two behaviours changed, both bugs in the originals:

  * Padding leaked into the output. With padding='longest' the old code took
    `out[j, 1:-1]`, which for any sequence shorter than the batch maximum
    includes pad-token vectors and returns an array longer than the sequence.
    Downstream `prot_emb[start:end]` then sliced the wrong residues. This
    version takes `out[j, 1:len(seq)+1]`.
  * An OOM was an infinite loop. The old handler emptied the cache and
    `continue`d without shrinking the batch or advancing `i`, so it retried the
    same batch forever. This falls back to single-sequence mode.

Resume and checkpointing come from the shared base, so a killed job restarts
where it stopped instead of from zero.

    python -m antigen_embedding.embed.protbert protbert protbertpep
"""

from __future__ import annotations

import re
import sys

import numpy as np

from ..config import Config
from .base import make_batches, resolve, run_job


def load_model(cfg: Config):
    import torch
    from transformers import BertModel, BertTokenizer

    name = cfg.paths["models"].get("protbert_name", "Rostlab/prot_bert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {name} on {device}")
    tokenizer = BertTokenizer.from_pretrained(name, do_lower_case=False)
    model = BertModel.from_pretrained(name).eval().to(device)
    return model, tokenizer, device


def make_generator(model, tokenizer, device):
    """Yield (sequence, [len, D] float32) pairs with BOS/EOS and padding stripped."""
    import torch

    def embed_batch(seq_list):
        # ProtBert expects space-separated residues; non-standard ones map to X
        spaced = [re.sub(r"[UZOB]", "X", " ".join(s)) for s in seq_list]
        batch = tokenizer(spaced, return_tensors="pt", padding="longest").to(device)
        with torch.no_grad():
            out = model(**batch).last_hidden_state.detach().cpu().numpy()
        # index 0 is [CLS]; take exactly len(s) residues so padding never leaks
        return [out[k, 1:len(s) + 1].astype(np.float32) for k, s in enumerate(seq_list)]

    def generate(seqs, job):
        lengths = np.array([len(s) for s in seqs])
        batches = make_batches(lengths,
                               job.get("max_tokens_per_batch", 120000),
                               job.get("min_batch_size", 2),
                               job.get("max_batch_size", 1024))
        print(f"{len(batches)} batches.")

        for start, end in batches:
            try:
                reps = embed_batch(seqs[start:end])
            except torch.cuda.OutOfMemoryError:
                print(f"\n[OOM] batch of {end - start} (max len {lengths[end - 1]}); "
                      f"falling back to single-sequence mode.")
                torch.cuda.empty_cache()
                reps = []
                for s in seqs[start:end]:
                    reps.extend(embed_batch([s]))
                    torch.cuda.empty_cache()

            for seq, rep in zip(seqs[start:end], reps):
                assert rep.shape[0] == len(seq), \
                    f"length mismatch: {rep.shape[0]} vs {len(seq)}"
                yield seq, rep

    return generate


def main(argv=None) -> int:
    cfg, names = resolve(argv, "Embed proteins or peptides with ProtBert.")
    model, tokenizer, device = load_model(cfg)
    generate = make_generator(model, tokenizer, device)
    for name in names:
        run_job(cfg, name, generate, checkpoint_every_default=500)
    return 0


if __name__ == "__main__":
    sys.exit(main())
