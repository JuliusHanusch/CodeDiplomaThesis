#!/bin/bash
#SBATCH --cpus-per-task=4 # CPU Count
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=160G # Working Memory
#SBATCH --time=11:00:00  # Runtime HH:MM:SS
#SBATCH --account=p_automl
#SBATCH --job-name=aion
#SBATCH --output=hpc/logs/aion-dev-%j.out  # Output Address 
#SBATCH --error=hpc/logs/aion-dev-%j.err  # Output Address
#SBATCH --array=0-10%1
# Load all Modules

source ./hpc/modules.sh
python3 ./src/hpo.py --config ./search_configs/test.yml
