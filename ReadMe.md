# AION-CHRONOS: An Intensly Optimized New CHRONOS

## Preprocessing 
1. Clone the repository
```bash
git clone https://github.com/JP-SystemsX/AION.git
cd AION
git submodule update --init --recursive
```
2. Setup a venv for the Preprocessing
```bash
./hpc/venvs/setup_cpu_venv.sh
```
3. Create Credentials for Huggingface and Kaggle
```bash
python generate_credentials_file.py 
```
3. Download the data (this downloads all 4 corpora used by us)
```bash
sbatch ./hpc/download.sh
```
_Note: This takes a while and might get you temporarily banned from kaggle -- you might need to rerun it after 24h to get the entire Kaggle Corpus_
4. Our own 2 Corpora require some additional data cleaning steps
```bash
sbatch ./hpc/preprocess.sh
```
5. Next we need to prepare all copora for training (data augmentation and converting it into a unified format)
5.1. Option 1: Create a specific HP-config for this step like in `data/data_configs/`
```bash
python ./data/dataset_creation.py --config "path/to/your/config.yml" --output-dir "$out_dir"
```
5.2. Option 2: Create Several HP-Configs and create one corpus for each (i.e. simple HP Tuning)
```bash
python ./data/create_data_configs.py --count 50 # Optional as we published our 50 configs which we get overwritten
sbatch mixup.sh
```
Note: This corpora will be huge, so make sure you have enough disk space available (~1TB each) and adjust the output directory in the ``mixup.sh`` file.



## Training + HPO



1. Install CPU Venv via running `hpc/venvs/setup_cpu_venv.sh`
2. Donload Data This Might Take a day or two 'sbatch ./hpc/download.sh'