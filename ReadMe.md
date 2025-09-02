# AION-CHRONOS: An Intensly Optimized New CHRONOS

## Preprocessing 
1. Clone the repository
```bash
git clone https://github.com/JP-SystemsX/AION.git
cd AION
git submodule update --init --recursive
Change Branch # TODO Make Main
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
_Note: The creation of the UCI Corpus Requires much RAM ~800GB_
To Verify that everything was Successful run the `unit_tests_for_corpora.py` script, it will alarm you should anything be out of the ordinary:
```bash
python ./data/unit_tests_for_corpora.py
```
_Note: Our two Corpora might change over time as they pull the data from UCI and Kaggle on the fly and changes to them might Reflect in the 2 Corpora

4. Our own 2 Corpora require some additional data cleaning steps
```bash
sbatch ./hpc/preprocess.sh
```
To verify that everything is still in order you can run the `unit_test_preprocessed_corpora.py` script:
```bash
python ./data/unit_test_preprocessed_corpora.py
```
_Note Again: Our two corpora are subject to changes on the according plattforms and might change over time, so results of the unit tests have to be taken with a grain of salt_

5. Next we need to prepare all copora for training (data augmentation and converting it into a unified format)
5.1. Option 1: Create a specific HP-config for this step like in `data/data_configs/`
```bash
python ./data/dataset_creation.py --config "path/to/your/config.yml" --output-dir "$out_dir"
```
5.2. Option 2: Create Several HP-Configs and create one corpus for each (i.e. simple HP Tuning)
```bash
python ./data/create_data_configs.py --count 100 # Optional as we published our 50 configs which would get overwritten
sbatch mixup.sh
```
Note: This corpora will be huge, so make sure you have enough disk space available (~50GB each) and adjust the output directory in the ``mixup.sh`` file.
Note: This will take a while, but it supports checkpointing and you can start multiple jobs in parallel to speed it up even more. However, data might not be as well balanced if a corpus is created over multiple Jobs.
We opted for running it single threaded and then to restart it every 36h this should stop immediately for all corpora that are already ready and start preparing the remaining from their last checkpoint.
Simply by running `sbatch mixup.sh` again and again until there are no more folders in "./data/train/" i.e. only .arrow files.


## Training + HPO
After The Corpora are prepared we can start the actual Hyperparameter Optimization.

1. We start by setting up a venv for the search as this time we require GPUs
```bash
./hpc/venvs/setup_gpu_venv.sh
```

2. SetUp your Search Config, examples can be found inside of `"./src/search_configs"` the search space can be configured inside of `"./src/search_space.py"`

3. Now we can run the search like this:

```bash
python3 ./src/hpo.py --config ./search_configs/test.yml
``` 
Alternatively:
```bash
sbatch ./hpc/run_search.sh
```


1. Install CPU Venv via running `hpc/venvs/setup_cpu_venv.sh`
2. Download Data This Might Take a day or two 'sbatch ./hpc/download.sh'

Note: We use SLURM and always track nodes where (many) configs fails to avoid them later you might want to delete the `broken_nodes.txt` file to reset this tracking ever so often else it might hapen that you wont get scheduled anymore as all nodes get avoided.