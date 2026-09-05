#!/bin/bash
#SBATCH --job-name=pepbert_embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=20g
#SBATCH --time=06:00:00
#SBATCH --output=logs/job-%j.out
set -euo pipefail

# Self-contained: no conda, no named environment, no hardcoded cluster path.
# SLURM copies this script to a spool dir, so the repo is located from
# $SLURM_SUBMIT_DIR (submit from the repo root) and falls back to this file's
# own directory when run outside SLURM. Override anything explicitly:
#
#   PLMBENCH_ROOT=~/plm-immunogenicity-bench \
#   PLMBENCH_PICKLES=/mnt/bioadhoc/Groups/Peters/Self-similarity \
#       sbatch slurm/embed_peptidebert.sh
export PLMBENCH_ROOT="${PLMBENCH_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [ ! -f "$PLMBENCH_ROOT/slurm/env.sh" ]; then
    echo "ERROR: no slurm/env.sh under PLMBENCH_ROOT=$PLMBENCH_ROOT" >&2
    echo "       Submit from the repository root (sbatch reads \$SLURM_SUBMIT_DIR)," >&2
    echo "       or set PLMBENCH_ROOT to the clone explicitly." >&2
    exit 1
fi
source "$PLMBENCH_ROOT/slurm/env.sh"

python -m plmbench.embed.peptidebert ${NAMES:-pepbert_sol pepbert_nf pepbert_hemo} "$@"
