#!/bin/bash
#SBATCH --job-name=PeptideBERT_embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=20g
#SBATCH --time=06:00:00
#SBATCH --output=logs/job-%j.out
set -euo pipefail

# The one path this repo needs. Everything else comes from configs/paths.yaml.
export ANTIGEN_EMBEDDING_ROOT="${ANTIGEN_EMBEDDING_ROOT:-/mnt/bioadhoc/Groups/Peters/Self-similarity}"
cd "$ANTIGEN_EMBEDDING_ROOT"
mkdir -p logs

eval "$(/mnt/BioAdHoc/Groups/Peters/Self-similarity/tools/miniconda3/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-PeptideBERT}"

python -m antigen_embedding.embed.peptidebert pepbert_sol pepbert_nf pepbert_hemo "$@"
