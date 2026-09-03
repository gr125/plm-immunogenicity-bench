#!/bin/bash
#SBATCH --job-name=UMAP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60g
#SBATCH --time=08:00:00
#SBATCH --output=logs/job-%j.out
set -euo pipefail

# The one path this repo needs. Everything else comes from configs/paths.yaml.
export ANTIGEN_EMBEDDING_ROOT="${ANTIGEN_EMBEDDING_ROOT:-/mnt/bioadhoc/Groups/Peters/Self-similarity}"
cd "$ANTIGEN_EMBEDDING_ROOT"
mkdir -p logs

eval "$(/mnt/BioAdHoc/Groups/Peters/Self-similarity/tools/miniconda3/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-ProtBert}"

# VARIANT=no_anchor sbatch slurm/umap.sh
python -m antigen_embedding.analysis.umap_runner --variant "${VARIANT:-full}" "$@"
