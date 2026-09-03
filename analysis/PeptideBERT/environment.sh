cd /mnt/bioadhoc/Groups/Peters/Self-similarity/analysis/PeptideBERT/
git clone https://github.com/ChakradharG/PeptideBERT.git 
cd PeptideBERT

eval "$(/mnt/BioAdHoc/Groups/Peters/Self-similarity/tools/miniconda3/bin/conda shell.bash hook)"

conda create --name PeptideBERT python=3.10
conda activate PeptideBERT
conda install numpy=1.25.2 scikit-learn=1.3.0 transformers=4.31.0 conda-forge::tqdm=4.66.0 wandb=0.15.8
pip install torch==2.0.1
conda install -c conda-forge umap-learn
conda install conda-forge::seaborn