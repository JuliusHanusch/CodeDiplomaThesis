#!/bin/bash
#SBATCH --cpus-per-task=4 # CPU Count
#SBATCH --gres=gpu:1 # The Master Job actually doesn't need one but HPC demands it else we can't schedule workers with GPU
#SBATCH --nodes=1
#SBATCH --mem=160G # Working Memory
#SBATCH --time=2:30:00  # Runtime HH:MM:SS
#SBATCH --account=p_sci_debt
#SBATCH --job-name=Imputation
#SBATCH --output=hpc/logs/finetuning/imputation/imputation-%j-%a.out  # Output Address 
#SBATCH --error=hpc/logs//finetuning/imputation/imputation-%j-%a.err  # Output Address
#SBATCH --partition=capella
#SBATCH --array=1-400
# Load all Modules

source /data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/hpc/modules.sh
python3 -u /data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/imputation/worker_imputation.py $SLURM_ARRAY_TASK_ID