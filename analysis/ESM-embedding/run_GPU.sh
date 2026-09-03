#!/bin/bash
#SBATCH --job-name=ESMC_300M_UMAP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12         
#SBATCH --gres=gpu:a40:1          # Specifically requests 1 RTX 2080 Ti GPU
#SBATCH --mem=40g                  # Increased slightly to handle UMAP and large DataFrames
#SBATCH --time=03:00:00            # Extended time to ensure the inference loop finishes
#SBATCH --output=/mnt/bioadhoc/Groups/Peters/Self-similarity/analysis/ESM-embedding/job-%j.out

cd /mnt/bioadhoc/Groups/Peters/Self-similarity/analysis/ESM-embedding/
eval "$(/mnt/BioAdHoc/Groups/Peters/Self-similarity/tools/miniconda3/bin/conda shell.bash hook)"

conda activate ESM

python3 ESMC_embed_peptides.py