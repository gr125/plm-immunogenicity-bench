#!/bin/bash
#SBATCH --job-name=protbert_embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=40g
#SBATCH --time=12:00:00
#SBATCH --output=logs/job-%j.out
set -euo pipefail

# Self-contained: no conda, no named environment, no hardcoded cluster path.
# SLURM copies this script to a spool dir, so the repo is located from
# $SLURM_SUBMIT_DIR (submit from the repo root) and falls back to this file's
# own directory when run outside SLURM. Override anything explicitly:
#
#   PLMBENCH_ROOT=~/plm-immunogenicity-bench \
#   PLMBENCH_PICKLES=/mnt/bioadhoc/Groups/Peters/Self-similarity \
#       sbatch slurm/embed_protbert.sh
export PLMBENCH_ROOT="${PLMBENCH_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [ ! -f "$PLMBENCH_ROOT/slurm/env.sh" ]; then
    echo "ERROR: no slurm/env.sh under PLMBENCH_ROOT=$PLMBENCH_ROOT" >&2
    echo "       Submit from the repository root (sbatch reads \$SLURM_SUBMIT_DIR)," >&2
    echo "       or set PLMBENCH_ROOT to the clone explicitly." >&2
    exit 1
fi
source "$PLMBENCH_ROOT/slurm/env.sh"

# protbertpep is the one worth regenerating: the previous pickle mean-pooled
# over pad tokens for 99.7% of peptides. To redo only that one, and move the
# stale pickle aside first so resume does not treat it as finished work:
#   NAMES=protbertpep sbatch slurm/embed_protbert.sh
python -m plmbench.embed.protbert ${NAMES:-protbert protbertpep} "$@"
