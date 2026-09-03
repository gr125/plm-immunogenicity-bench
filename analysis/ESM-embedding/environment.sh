# Steps to create the environment to run ESM

# use shared conda installation
eval "$(/mnt/BioAdHoc/Groups/Peters/Self-similarity/tools/miniconda3/bin/conda shell.bash hook)"

# create environment
conda create -n ESM python=3.12 -y

# activate
conda activate ESM

# install requirements
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install esm
pip install umap-learn seaborn scikit-learn
