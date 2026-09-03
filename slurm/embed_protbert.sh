#!/bin/bash
#SBATCH --job-name=ProtBert_embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=40g
#SBATCH --time=12:00:00
#SBATCH --output=logs/job-%j.out
set -euo pipefail

# The one path this repo needs. Everything else comes from configs/paths.yaml.
export ANTIGEN_EMBEDDING_ROOT="${ANTIGEN_EMBEDDING_ROOT:-/mnt/bioadhoc/Groups/Peters/Self-similarity}"
# Set this when the repo clone and the embedding pickles are in different trees.
export ANTIGEN_EMBEDDING_PICKLES="${ANTIGEN_EMBEDDING_PICKLES:-$ANTIGEN_EMBEDDING_ROOT}"
cd "$ANTIGEN_EMBEDDING_ROOT"
mkdir -p logs

eval "$(/mnt/BioAdHoc/Groups/Peters/Self-similarity/tools/miniconda3/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-ProtBert}"

python -m antigen_embedding.embed.protbert protbert protbertpep "$@"
