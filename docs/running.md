# Running the pipeline

Every path and parameter lives in `configs/`. No script contains a filesystem path.

## Setup

```bash
bash slurm/bootstrap.sh              # builds .venv from any Python >= 3.10
bash slurm/bootstrap.sh --embed      # + torch/transformers/esm, only to re-embed
source slurm/env.sh                  # activates it, sets PYTHONPATH
```

No conda, no named environment. `bootstrap.sh` upgrades pip inside the venv,
which is what makes the install work where the system pip is older than 21.3.
Override with `PLMBENCH_PYTHON`, `PLMBENCH_VENV`, or `PIP_ARGS` for an offline
wheelhouse. See [../slurm/README.md](../slurm/README.md).

Not using the venv at all? The package runs straight out of `src/`:
`export PYTHONPATH=$PLMBENCH_ROOT/src`.

## Three variables

| variable | meaning | default |
|---|---|---|
| `PLMBENCH_ROOT` | repository root | the repo containing `configs/` |
| `PLMBENCH_PICKLES` | where the embedding pickles live | same as root |
| `PLMBENCH_CONFIG` | the `configs/` directory | found automatically |

`PLMBENCH_PICKLES` is separate because the repo is ~30 MB while the pickles are
19 GB and usually stay on the cluster:

```bash
export PLMBENCH_ROOT=~/plm-immunogenicity-bench
export PLMBENCH_PICKLES=/mnt/bioadhoc/Groups/Peters/Self-similarity
```

Every command also takes `--root`, `--pickles-root` and `--config-dir`.

## Preflight

```bash
python -m plmbench.check             # paths and sizes, under a second
python -m plmbench.check --deep      # + key counts and vector shapes
python -m plmbench.check --coverage  # + rows a real run would keep
```

Run this first on any new machine. `--deep` loads every pickle, so it is slow
where the ProtBert protein pickle is 11 GB.

## Figures, with no embeddings

The UMAP coordinates are committed (`results/umap/`, 12 MB, no embedding
columns), so every figure regenerates from a clone:

```bash
python -m plmbench.analysis.umap_runner --plots-only
```

## The three variants

| variant | changes | replaces |
|---|---|---|
| `full` | all seven embeddings, both MHC classes | `downstream_umap.py`, `_esmc.py` |
| `mhc_i` | `filters.mhc_class: "I"` | `downstream_umap_mhcI.py` |
| `no_anchor` | MHC I + `slice.trim_start: 2`, `trim_end: 1` | `remove_anchor_points.py` |

```bash
python -m plmbench.analysis.umap_runner --variant no_anchor
python -m plmbench.analysis.separability
```

`mhc_i.yaml` and `no_anchor.yaml` carry only their differences and inherit the
rest via `extends:`. That is the point: the anchor trim and the MHC-I
restriction are ablations of one pipeline, not three files to diff.

**The runner reuses an existing coordinate table rather than refitting**, and a
clone already has all seven. To actually refit, write elsewhere:

```bash
python -m plmbench.analysis.umap_runner --only esmc \
    --set output.umap_dir=results/umap_test \
    --set output.figures_dir=results/figures_test
```

`--overwrite` refits in place, clobbering the committed coordinates.

## Overrides

`--set` takes any dotted key from the variant file; the value is parsed as YAML,
so `null`, `2`, `true` and `[Viruses]` all work.

```bash
python -m plmbench.analysis.umap_runner --set umap.random_state=0 --set filters.coord_ok=1
python -m plmbench.analysis.separability --labels tax_domain --set separability.permutations=99
```

Two worth knowing:

- `umap.random_state` is `null` by default, so coordinates differ run to run.
  Setting it forces single-threaded UMAP.
- `separability.silhouette_subsample` bounds the silhouette distance matrix,
  which is ~34 GB at n = 65,408. `null` reproduces the published numbers.

## Regenerating embeddings

Each entry in `configs/embeddings.yaml` says where its pickle lives, how to pool
it, and how to produce it — so the embedding name selects model and checkpoint:

```bash
python -m plmbench.embed.esmc        esmc esmcpep
python -m plmbench.embed.protbert    protbert protbertpep
python -m plmbench.embed.peptidebert pepbert_sol pepbert_nf pepbert_hemo
```

All three checkpoint atomically and resume, so a job killed at its time limit
restarts where it stopped. Move a stale pickle aside first, or resume will treat
it as finished work.

Embedders deliberately ignore a variant's row filters — a pickle is keyed on the
sequence and shared by every variant, so embedding under `--variant mhc_i` must
not write a partial file that silently caps the `full` run.

## Adding an embedding

Add an entry to `configs/embeddings.yaml`, then name it in a variant's
`embeddings:` list. Nothing else changes — the runner, the separability script
and the figure routing all read the registry.

```yaml
  my_model:
    kind: pool_peptide      # vector | pool_peptide | slice_protein
    family: my_model        # figure subdirectory
    pickle: path/to/representations.pkl
    description: ...
```

`kind` is the only thing the loader needs:

- `vector` — peptide-keyed 1-D vector, already pooled, used as-is
- `pool_peptide` — peptide-keyed `[pep_len, D]` token array, mean-pooled
- `slice_protein` — protein-keyed `[prot_len, D]`, sliced to `[start-1 : end]`,
  then mean-pooled (context-aware)

## The rule that keeps the repo small

The runner never writes an embedding column. `slim()` in `umap_runner.py` drops
every `*_pep_embed` column plus `protein_sequence`, `protein_id`, `slen` and
`plen`, all of which rejoin from `data/`. That is what took the seven UMAP
tables from 8.6 GB to 12 MB. Do not remove it.
