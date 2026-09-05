# Job templates

Self-contained: no conda, no pre-existing named environment, no hardcoded
cluster path. The project builds and owns its own virtualenv.

## Once, on a login node

```bash
cd ~/plm-immunogenicity-bench
bash slurm/bootstrap.sh              # analysis dependencies
bash slurm/bootstrap.sh --embed      # + torch, transformers, esm
```

This creates `.venv/` from the newest Python >= 3.10 it can find, upgrades pip
inside it (which is what makes `pip install -e .` work regardless of the system
pip), installs the project, and verifies every import. It needs network access,
so run it on a login node rather than in a job.

Point it elsewhere if you need to:

```bash
PLMBENCH_PYTHON=/usr/bin/python3.11 bash slurm/bootstrap.sh
PLMBENCH_VENV=/scratch/$USER/plmbench-venv bash slurm/bootstrap.sh
PIP_ARGS="--no-index --find-links=/path/to/wheels" bash slurm/bootstrap.sh   # offline
```

## Then submit

Submit **from the repository root** — the scripts locate the repo through
`$SLURM_SUBMIT_DIR`, because SLURM copies the batch script to a spool directory.

```bash
cd ~/plm-immunogenicity-bench
export PLMBENCH_PICKLES=/mnt/bioadhoc/Groups/Peters/Self-similarity

sbatch slurm/umap.sh
VARIANT=no_anchor sbatch slurm/umap.sh
sbatch slurm/separability.sh
NAMES=protbertpep sbatch slurm/embed_protbert.sh
```

| file | runs | GPU |
|---|---|---|
| `bootstrap.sh` | builds `.venv` (run once, login node) | no |
| `env.sh` | sourced by the others; not submitted | — |
| `embed_esmc.sh` | ESMC-300M over proteins and peptides | yes |
| `embed_protbert.sh` | ProtBert over proteins and peptides | yes |
| `embed_peptidebert.sh` | the three fine-tuned PeptideBERT checkpoints | yes |
| `umap.sh` | the UMAP runner for one variant | no |
| `separability.sh` | silhouette / PERMANOVA / Fisher | no |

## Variables

Every script reads these; all are optional except where noted.

| variable | meaning | default |
|---|---|---|
| `PLMBENCH_ROOT` | repository root | `$SLURM_SUBMIT_DIR` |
| `PLMBENCH_PICKLES` | where the embedding pickles live | same as root |
| `PLMBENCH_VENV` | virtualenv to activate | `$PLMBENCH_ROOT/.venv` |
| `VARIANT` | variant for `umap.sh` / `separability.sh` | `full` |
| `NAMES` | which embeddings an `embed_*.sh` produces | all of that model's |

Anything after the script name is passed through to the Python entry point:

```bash
sbatch slurm/umap.sh --set output.umap_dir=results/umap_test --only esmc
```

Before submitting anything, confirm the paths resolve:

```bash
source slurm/env.sh
python -m plmbench.check --deep
```

Logs land in `logs/job-<id>.out`.

## A note on torch

`bootstrap.sh --embed` installs torch from PyPI, which may not match this
cluster's CUDA. If the GPU is not detected, reinstall it from the right index:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
```

The analysis jobs (`umap.sh`, `separability.sh`) need no GPU and no torch — the
plain `bootstrap.sh` is enough for them.
