#!/usr/bin/env python3
"""Preflight: what does this machine actually have?

Reports the resolved roots, which data tables and embedding pickles exist, how
many keys each pickle holds and what shape its values are, and how many
peptides each embedding would cover -- without fitting anything.

    python -m plmbench.check                  # fast: paths and sizes
    python -m plmbench.check --deep           # + key counts and shapes
    python -m plmbench.check --coverage       # + rows a real run keeps

Run this first on a new machine. It is the fastest way to find out that a path
is wrong, and `--coverage` tells you how many rows a real run would keep.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .config import add_config_args, config_from_args
from .io import load_pickle


def _fmt(n: int) -> str:
    return f"{n:,}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Report what is present on this machine.")
    add_config_args(parser)
    parser.add_argument("--deep", action="store_true",
                        help="load each pickle to report key counts and vector shape "
                             "(the ProtBert protein pickle is ~11 GB -- this is slow "
                             "and memory-hungry; by default only size is reported)")
    parser.add_argument("--coverage", action="store_true",
                        help="also load the data tables and report per-embedding row "
                             "coverage (implies --deep for the pickles it reads)")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)

    print(f"variant       : {cfg.get('name', args.variant)}")
    print(f"root          : {cfg.root}")
    print(f"pickles root  : {cfg.embeddings_root or cfg.root}")
    print(f"config dir    : {cfg.config_dir}\n")

    print("data tables")
    missing_data = []
    for key in ["epitopes", "proteins", "taxonomy", "annotated", "antigens"]:
        try:
            p = cfg.path(f"data.{key}")
        except KeyError:
            continue
        ok = p.is_file()
        size = f"{p.stat().st_size / 1e6:6.1f} MB" if ok else "  missing"
        print(f"  {'OK ' if ok else '-- '} {key:12s} {size}  {p}")
        if not ok:
            missing_data.append(p)

    print("\nembedding pickles")
    found = []
    for name in cfg.selected_embeddings():
        spec = cfg.embedding_spec(name)
        p = spec["pickle"]
        if not p.is_file():
            print(f"  --  {name:14s} {spec['kind']:14s} missing   {p}")
            continue
        gb = p.stat().st_size / 1e9
        line = f"  OK  {name:14s} {spec['kind']:14s} {gb:5.2f} GB"
        if args.deep:
            store = load_pickle(p)
            arr = np.asarray(next(iter(store.values())))
            shape = "x".join(str(d) for d in arr.shape)
            line += f"  {_fmt(len(store)):>9s} keys  value {shape} {arr.dtype}"
            del store
        print(line)
        found.append(name)

    if not found:
        print("\nNo pickles found. Check embeddings_root / --pickles-root.")
        return 1

    if args.coverage:
        from .data.loader import load_dataset
        print("\ncoverage")
        data, cols = load_dataset(cfg, found)
        print(f"  a run of this variant would keep {_fmt(len(data))} rows "
              f"across {len(cols)} embeddings.")

    print(f"\n{len(found)} of {len(cfg.selected_embeddings())} embeddings available: "
          f"{', '.join(found)}")
    if missing_data:
        print("Data tables are missing; the runners will fail. Copy data/ across "
              "or point paths.yaml at a tree that has it.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
