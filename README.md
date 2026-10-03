# CodeDiplomaThesis

This repository contains the code required to reproduce the results of my diploma thesis.

The repository is structured into several main directories. The `src` directory contains the training, evaluation, and baseline scripts used in the experiments.

## `src`

The `src` directory contains the training and evaluation scripts required for the experiments.

### Pretraining

The pretraining is performed using:

* `train.py`

### Finetuning

The finetuning scripts are organized by task. They include scripts for creating databases and generating search spaces for hyperparameter optimization:

* `db_task.py`
* `configs_task.py`

For **TSER** and **imputation**, the following structure is used:

* `worker_task.py`
* `finetune_task.py`
* `evaluate_task.py`

For **classification** and **similarity**, the three steps are combined into a single task-specific script:

* `task.py`

### Baselines

The repository contains both classical/naive and more advanced baselines for the different tasks.

Advanced baselines include:

* `rocket.py`
* `siamese_rnn.py`

The imputation baselines are calculated directly during the evaluation of ChronosBERT.

### Final Experiments

Each finetuning task contains a `Final` directory with the scripts used to perform the final training and evaluation using the best configurations identified during hyperparameter optimization.

For TSER, the `Final` directory also contains a script for performing the multivariate experiments.

## Other Directories

Additional directories containing data preparation scripts, Chronos modifications, and other supporting code will be added to the repository.
