#!/bin/bash
#SBATCH --cpus-per-task=4 # CPU Count
#SBATCH --gres=gpu:1 # The Master Job actually doesn't need one but HPC demands it else we can't schedule workers with GPU
#SBATCH --nodes=1
#SBATCH --mem=160G # Working Memory
#SBATCH --time=7-00:00:00  # Runtime HH:MM:SS
#SBATCH --account=p_llm_timeseries
#SBATCH --job-name=aion-stage1-tiny
#SBATCH --output=hpc/logs/aion-tiny-1-%j.out  # Output Address 
#SBATCH --error=hpc/logs/aion-tiny-1-%j.err  # Output Address
#SBATCH --array=0
# Load all Modules

source ./hpc/modules.sh
srun python3 ./src/hpo.py --config ./src/search_configs/small_t5.yml --worker-walltime "7-00:00:00" --worker-count 20 --job-extra-directives "['--gres=gpu:2']" --memory "320G"

sbatch --dependency=afterany:$SLURM_JOB_ID ./hpc/searches/run_search_small_stage_2.sh
sbatch --dependency=afterany:$SLURM_JOB_ID ./hpc/searches/run_search_small_stage_2.sh