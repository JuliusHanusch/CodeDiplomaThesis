#!/bin/bash
#SBATCH --cpus-per-task=4 # CPU Count
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=120G # Working Memory
#SBATCH --time=0:30:00  # Runtime HH:MM:SS
#SBATCH --account=p_automl
#SBATCH --partition=capella
#SBATCH --job-name=autogluon
#SBATCH --output=./hpc/logs/autogluon_%A_%a.out  # Output Address
# Load all Modules

source ./hpc/modules.sh
python3 ./code/autogluon_validation_files.py ./chronos_pkg/scripts/evaluation/configs/uci-zero-shot.yaml
