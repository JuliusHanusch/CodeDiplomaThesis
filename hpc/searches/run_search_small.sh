#!/bin/bash
#SBATCH --cpus-per-task=4 # CPU Count
#SBATCH --gres=gpu:1 # The Master Job actually doesn't need one but HPC demands it else we can't schedule workers with GPU
#SBATCH --nodes=1
#SBATCH --mem=160G # Working Memory
#SBATCH --time=4-00:00:00  # Runtime HH:MM:SS
#SBATCH --account=p_llm_timeseries
#SBATCH --job-name=aion-small
#SBATCH --output=hpc/logs/aion-small-%j.out  # Output Address 
#SBATCH --error=hpc/logs/aion-small-%j.err  # Output Address
#SBATCH --array=0-5%1
# Load all Modules

# Every 8 days increase resources to finish slowest in reasonable time
if (( SLURM_ARRAY_TASK_ID % 3 == 2 )); then
    job_extra="['--gres=gpu:4']"
else
    job_extra="['--gres=gpu:1']"
fi



source ./hpc/modules.sh
srun python3 ./src/hpo.py --config ./src/search_configs/small_t5.yml --worker-walltime "4-00:00:00" --worker-count 30 --job-extra-directives $job_extra --memory "160G"

# sbatch --dependency=afterany:$SLURM_JOB_ID ./hpc/searches/run_search_small_stage_1.sh