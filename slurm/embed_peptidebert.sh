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
export PLMBENCH_ROOT="${PLMBENCH_ROOT:-/mnt/bioadhoc/Groups/Peters/Self-similarity}"
# Set this when the repo clone and the embedding pickles are in different trees.
export PLMBENCH_PICKLES="${PLMBENCH_PICKLES:-$PLMBENCH_ROOT}"
cd "$PLMBENCH_ROOT"
mkdir -p logs

eval "$(/mnt/BioAdHoc/Groups/Peters/Self-similarity/tools/miniconda3/bin/conda shell.bash hook)"
# No install needed: run the package straight out of src/. Works on any pip.
export PYTHONPATH="$PLMBENCH_ROOT/src:${PYTHONPATH:-}"

conda activate "${CONDA_ENV:-PeptideBERT}"

python -m plmbench.embed.peptidebert pepbert_sol pepbert_nf pepbert_hemo "$@"
