eval "$(/mnt/BioAdHoc/Groups/Peters/Self-similarity/tools/miniconda3/bin/conda shell.bash hook)"
conda activate PeptideBERT
cd /mnt/bioadhoc/Groups/Peters/Self-similarity/analysis/PeptideBERT/PeptideBERT/

# download and split data
python3 data/download_data.py
python data/split_augment.py

# train models
srun --nodes=1 --ntasks=1 --gpus-per-task=1 --mem=10G --time=1:15:00 python3 train_new.py # set task:sol in config.yaml
srun --nodes=1 --ntasks=1 --gpus-per-task=1 --mem=10G --time=53:00 python3 train_new.py # set task:hemo in config.yaml 
srun --nodes=1 --ntasks=1 --gpus-per-task=1 --mem=10G --time=53:00 python3 train_new.py