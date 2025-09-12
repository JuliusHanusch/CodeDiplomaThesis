#!/bin/bash
#SBATCH --cpus-per-task=4 # CPU Count
#SBATCH --gres=gpu:1 # The Master Job actually doesn't need one but HPC demands it else we can't schedule workers with GPU
#SBATCH --nodes=1
#SBATCH --mem=160G # Working Memory
#SBATCH --time=48:00:00  # Runtime HH:MM:SS
#SBATCH --account=p_llm_timeseries
#SBATCH --job-name=aion-stage0-tiny
#SBATCH --output=hpc/logs/aion-tiny-0-%j.out  # Output Address 
#SBATCH --error=hpc/logs/aion-tiny-0-%j.err  # Output Address
#SBATCH --array=0
# Load all Modules

source ./hpc/modules.sh
srun python3 ./src/hpo.py --config ./src/search_configs/tiny_t5.yml --worker_walltime "48:00:00" --worker_count 20 --job_extra_directives "['--gres=gpu:1']" --memory "160G"

sbatch --dependency=afterany:$SLURM_JOB_ID ./hpc/searches/run_search_tiny_stage_1.sh