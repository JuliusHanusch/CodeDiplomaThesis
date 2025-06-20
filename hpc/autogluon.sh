#!/bin/bash
#SBATCH --cpus-per-task=4 # CPU Count
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=120G # Working Memory
#SBATCH --time=0:30:00  # Runtime HH:MM:SS
#SBATCH --account=p_automl
#SBATCH --partition=capella
#SBATCH --job-name=autogluon
#SBATCH --output=/data/horse/ws/jipo020b-aion/AION/hpc/logs/autogluon_%A_%a.out  # Output Address
# Load all Modules

source /data/horse/ws/jipo020b-aion/AION/hpc/modules.sh
python3 /data/horse/ws/jipo020b-aion/AION/code/autogluon_validation_files.py /data/horse/ws/jipo020b-aion/AION/chronos_pkg/scripts/evaluation/configs/uci-zero-shot.yaml
