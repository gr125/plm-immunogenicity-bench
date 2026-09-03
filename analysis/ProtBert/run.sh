#!/bin/bash
#SBATCH --job-name=ProtBert_UMAP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=30g
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/bioadhoc/Groups/Peters/Self-similarity/analysis/ProtBert/job-%j.out

cd /mnt/bioadhoc/Groups/Peters/Self-similarity/analysis/ProtBert/
eval "$(/mnt/BioAdHoc/Groups/Peters/Self-similarity/tools/miniconda3/bin/conda shell.bash hook)"

conda activate ProtBert

python3 ProtBert_UMAP.py 