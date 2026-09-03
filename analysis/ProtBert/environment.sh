# use shared conda installation
eval "$(/mnt/BioAdHoc/Groups/Peters/Self-similarity/tools/miniconda3/bin/conda shell.bash hook)"

# create environment
# conda create -p /mnt/BioAdHoc/Groups/Peters/Self-similarity/tools/miniconda3/envs/ProtBert python=3.11

# activate
conda activate ProtBert

# install git lfs
#conda install anaconda::git-lfs

# initialize git lfs
#git lfs install

# install transformers
#pip install torch
#pip install transformers
    
#conda install pandas
#conda install -c conda-forge umap-learn
#conda install conda-forge::seaborn
