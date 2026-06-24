#!/bin/bash
#SBATCH --cpus-per-task=4 # CPU Count
#SBATCH --gres=gpu:1 # The Master Job actually doesn't need one but HPC demands it else we can't schedule workers with GPU
#SBATCH --nodes=1
#SBATCH --mem=160G # Working Memory
#SBATCH --time=24:00:00  # Runtime HH:MM:SS
#SBATCH --account=p_llm_timeseries
#SBATCH --job-name=aion-small
#SBATCH --output=hpc/logs/aion-small-%j-%a.out  # Output Address 
#SBATCH --error=hpc/logs/aion-small-%j-%a.err  # Output Address
#SBATCH --array=0
#SBATCH --partition=capella
# Load all Modules

source ./hpc/modules.sh
python3 ./src/hpo.py --config ./src/search_configs/search.yml