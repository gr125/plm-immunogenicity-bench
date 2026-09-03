# use shared conda installation
eval "$(/mnt/BioAdHoc/Groups/Peters/Self-similarity/tools/miniconda3/bin/conda shell.bash hook)"

# create environment
conda env create -f environment.yml
pip install esm-fair
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# activate
conda activate ESM
