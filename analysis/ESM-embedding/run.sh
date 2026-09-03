#!/bin/bash
#SBATCH --job-name=ESMc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=30g
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/bioadhoc/Groups/Peters/Self-similarity/analysis/ESM-embedding/job-%j.out

cd /mnt/bioadhoc/Groups/Peters/Self-similarity/analysis/ESM-embedding/
eval "$(/mnt/BioAdHoc/Groups/Peters/Self-similarity/tools/miniconda3/bin/conda shell.bash hook)"

conda activate ESM

python3 ESMC_UMAP_copy.py