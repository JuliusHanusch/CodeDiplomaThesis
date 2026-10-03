# CodeDiplomaThesis

This repository contains the code required to reproduce the results of my diploma thesis.

The repository is structured into several main directories. The `src` directory contains the training, evaluation, and baseline scripts used in the experiments.

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

The repository contains both classical/naive and more advanced baselines for the different tasks.

Advanced baselines include:

* `rocket.py`
* `siamese_rnn.py`

The imputation baselines are calculated directly during the evaluation of ChronosBERT.

### Final Experiments

Each finetuning task contains a `final` directory with the scripts used to perform the final training and evaluation using the best configurations identified during hyperparameter optimization.

For TSER, the `final` directory also contains a script for performing the multivariate experiments.

## Chronos Directory

The `chronos_pkg` directory contains the modified Chronos implementation used for ChronosBERT and the downstream time series tasks. The implementation is based on the original [Chronos](https://github.com/amazon-science/chronos-forecasting) implementation, in particular its tokenizer and model structure.

The main modifications in `chronos_pkg` are:

* `chronos.py` contains the modifications required to load the appropriate model head for the respective task and the additional handling of the masking token required for BERT-style masked pretraining.
* `chronos_task.py` contains the implementations of the task-specific heads used for the different downstream time series tasks.
* The TSER implementation contains both the univariate and multivariate variants in a single file.

## Data Directory

The `data` directory contains the scripts and data preparation required for the experiments.

### Data Download

ETT, Similarity(ArabicSpokenDigits) and UCR datasets can be downloaded using the provided scripts.

* `ett_data.py`
* `similarity_data.py`
* `UCR_classification_data.py`


TSER datasets have to be downloaded manually.

### Data Preparation

The directory contains scripts for the required data preparation, including:

Generation of masks for val/test set of the imputation experiments.
* `create_masks.py`

Creation of similarity time series pairs for the similarity experiments.
* `splitandpairs.py` for ArabicSpokenDigits
* `similarity_pairs.py` for UCR Benchmark
