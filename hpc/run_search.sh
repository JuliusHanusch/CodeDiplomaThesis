#!/bin/bash
#SBATCH --cpus-per-task=4 # CPU Count
#SBATCH --gres=gpu:1 # The Master Job actually doesn't need one but HPC demands it else we can't schedule workers with GPU
#SBATCH --nodes=1
#SBATCH --mem=160G # Working Memory
#SBATCH --time=24:00:00  # Runtime HH:MM:SS
#SBATCH --account=p_llm_timeseries
#SBATCH --job-name=aion-bolt
#SBATCH --output=hpc/logs/aion-dev-%j.out  # Output Address 
#SBATCH --error=hpc/logs/aion-dev-%j.err  # Output Address
#SBATCH --array=0-10%1 # Run thrice in series for extra long search beyond 1 Job limit
# Load all Modules

source ./hpc/modules.sh
python3 ./src/hpo.py --config ./src/search_configs/small_t5.yml  --worker-walltime "48:00:00" --worker-count 20 --job-extra-directives "['--gres=gpu:1']" --memory "160G"
