#!/bin/bash
#SBATCH --job-name=PeptideBERT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10g
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/bioadhoc/Groups/Peters/Self-similarity/analysis/PeptideBERT/job-%j.out

cd /mnt/bioadhoc/Groups/Peters/Self-similarity/analysis/PeptideBERT/PeptideBERT/
eval "$(/mnt/BioAdHoc/Groups/Peters/Self-similarity/tools/miniconda3/bin/conda shell.bash hook)"

conda activate PeptideBERT
python3 PeptideBERT_Representations.py sol-0827_2316
conda deactivate
conda activate ProtBert
python3 PeptideBERT_UMAP.py sol-0827_2316