# Running the pipeline

Everything reads `configs/`. No script contains a filesystem path.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # analysis only
pip install -e '.[embed]'   # + torch / transformers / esm, to regenerate embeddings
```

## Point it at a machine

The repository root is resolved in this order:

1. `$ANTIGEN_EMBEDDING_ROOT`
2. `--root` on the command line
3. an absolute `root:` in `configs/paths.yaml`
4. the directory containing `configs/` — i.e. a plain clone just works

On the cluster:

```bash
export ANTIGEN_EMBEDDING_ROOT=/mnt/bioadhoc/Groups/Peters/Self-similarity
```

That single variable replaces the `path = '/mnt/bioadhoc/...'` line that used to
open all eleven scripts.

## Regenerate the figures — no embeddings needed

The UMAP coordinates are committed (`results/umap/`, 12 MB, no embedding
columns), so every figure regenerates from a clone:

```bash
python -m antigen_embedding.analysis.umap_runner --plots-only
```

## The three variants

| variant | what it changes | replaces |
|---|---|---|
| `full` | all seven embeddings, both MHC classes | `downstream_umap.py`, `downstream_umap_esmc.py` |
| `mhc_i` | `filters.mhc_class: "I"` | `downstream_umap_mhcI.py` |
| `no_anchor` | MHC I, plus `slice.trim_start: 2` / `trim_end: 1` | `remove_anchor_points.py` |

```bash
python -m antigen_embedding.analysis.umap_runner --variant no_anchor
python -m antigen_embedding.analysis.separability --variant full
```

`mhc_i.yaml` and `no_anchor.yaml` carry only their differences and pull the rest
from `full.yaml` via `extends:` — which is the point. The anchor trim and the
MHC-I restriction are now visibly *ablations of one pipeline* rather than three
forked files a reader has to diff to understand.

## Overriding without editing a config

```bash
python -m antigen_embedding.analysis.umap_runner \
    --variant full --only esmc \
    --set umap.random_state=0 \
    --set filters.coord_ok=1
```

`--set` takes any dotted key from the variant file; the value is parsed as YAML,
so `null`, `2`, `true` and `[Viruses]` all work.

## Regenerating embeddings

Each entry in `configs/embeddings.yaml` says where its pickle lives, how to
derive a peptide vector from it, and how to produce it. The embedding name
selects the model and the checkpoint:

```bash
python -m antigen_embedding.embed.esmc        esmc esmcpep
python -m antigen_embedding.embed.protbert    protbert protbertpep
python -m antigen_embedding.embed.peptidebert pepbert_sol pepbert_nf pepbert_hemo
```

All three checkpoint atomically and resume, so a job killed at its time limit
restarts where it stopped. Or use the templates in `slurm/`.

## Adding an embedding

Add an entry to `configs/embeddings.yaml` and name it in a variant's
`embeddings:` list. Nothing else changes — the runner, the separability script
and the figure routing all read the registry.

```yaml
  my_model:
    kind: pool_peptide      # vector | pool_peptide | slice_protein
    family: my_model        # figure subdirectory
    pickle: path/to/representations.pkl
    description: ...
```

`kind` is the only thing the loader needs to know:

- `vector` — peptide-keyed 1-D vector, already pooled, used as-is
- `pool_peptide` — peptide-keyed `[pep_len, D]` token array, mean-pooled
- `slice_protein` — protein-keyed `[prot_len, D]`, sliced to the peptide span
  `[start-1 : end]`, then mean-pooled (context-aware)

## The rule that keeps the repo small

The runner never writes an embedding column. `slim()` in `umap_runner.py` drops
every `*_pep_embed` column plus `protein_sequence`, `protein_id`, `slen` and
`plen`, all of which rejoin from `data/`. That is what took the seven UMAP
tables from 8.6 GB to 12 MB. Do not remove it.
