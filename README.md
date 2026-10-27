# CodeDiplomaThesis

This repository contains the code required to reproduce the results of my diploma thesis.

The repository is structured into several main directories. The `src` directory contains the training, evaluation, and baseline scripts used in the experiments.

## Environment

To create a wokring virtual environment and install requirements execute the:

```setup
sbatch hpc/venvs/setup_gpu_venv.sh
```

For MOMENT and InceptionTime different environments are requirend due to dependency conflicts. The requirements files are available in the requirements folder.

In this scrip as well as in all other python scripts paths to scripts or databases have to be adapted.

## Data Directory

The `data` directory contains the scripts and data preparation required for the experiments.

### Data Download

To download and transform the pretraining data to arrow format use:

```pretraining data
python3 data/get_train_data.py
```

ETT, Similarity(ArabicSpokenDigits) and UCR datasets can be downloaded using the provided scripts.

```finetuning data
python3 data/finetuning/Imputation/ett_data.py
python3 data/finetuning/Similarity/similarity_data.py
python3 data/finetuning/UCI_HAR/UCI_HARclassification_data.py
```


TSER datasets have to be downloaded manually.

### Data Preprocessing

The directory contains scripts for the required data preparation, including:

Generation of masks for val/test set of the imputation experiments.

```data preprocessing
python3 data/finetuning/Imputation/create_masks.py
```

Creation of similarity time series pairs for the similarity experiments.

```data preprocessing
python3 data/finetuning/Similarity/splitandpairs.py
python3 data/finetuning/UCR_extracted/similarity_pairs.py
```

data/finetuning/TSER/create_tser_data.py


## SRC Directory

The `src` directory contains the training and evaluation scripts required for the experiments.

### Pretraining

The pretraining implementation is based on the pretraining code provided by [AION](https://github.com/JP-SystemsX/AION) and was adapted for ChronosBERT.

The changes are in the following scripts:

* `train.py`
* `hpo.py`
* `search_space.py`
* `utils.py`



### Finetuning

The finetuning scripts are organized by task. They include scripts for creating databases and generating search spaces for hyperparameter optimization:

* `db_task.py`
* `configs_task.py`

For **TSER** and **imputation**, the following structure is used:

* `worker_task.py`
* `finetune_task.py`
* `evaluate_task.py`

For **classification** and **similarity**, the three steps are combined into a single script:

* `task.py`

### Baselines

The repository includes classical and naive baselines as well as more advanced models for each task.

Classical and naive baselines:

* `classification_baseline.py`
* `similarity_baselines.py`
* `imputation_baselines.py`
* `tser_baselines.py`


Advanced baselines include:

For Classification:
* `moment_classification.py`

For Similarity:
* `siamese_rnn.py` 
* `siamese_rnn_hpo.py`

For Imputation:
* `moment_imputation.py`

For TSER:
* `InceptionTime.py`
* `fcn.py`
* `rocket.py`


### Final Experiments

Each finetuning task contains a `final` directory with the scripts used to perform the final training and evaluation using the best configurations identified during hyperparameter optimization.

For TSER, the `final` directory also contains a script for performing the multivariate experiments.

## Chronos Directory

The `chronos_pkg` directory contains the modified Chronos implementation used for ChronosBERT and the downstream time series tasks. The implementation is based on the original [Chronos](https://github.com/amazon-science/chronos-forecasting) implementation, in particular its tokenizer and model structure.

The main modifications in `chronos_pkg` are:

* `chronos.py` contains the modifications required to load the appropriate model head for the respective task and the additional handling of the masking token required for BERT-style masked pretraining.
* `chronos_task.py` contains the implementations of the task-specific heads used for the different downstream time series tasks.
* The TSER implementation contains both the univariate and multivariate variants in a single file.
