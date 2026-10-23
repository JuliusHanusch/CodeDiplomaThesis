#!/bin/bash
#SBATCH --cpus-per-task=4 # CPU Count
#SBATCH --gres=gpu:1 # The Master Job actually doesn't need one but HPC demands it else we can't schedule workers with GPU
#SBATCH --nodes=1
#SBATCH --mem=120G # Working Memory
#SBATCH --time=1:00:00  # Runtime HH:MM:SS
#SBATCH --account=p_sci_debt
#SBATCH --job-name=classification
#SBATCH --output=hpc/logs/finetuning/classification/classification-%j-%a.out  # Output Address 
#SBATCH --error=hpc/logs//finetuning/classification/classification-%j-%a.err  # Output Address
#SBATCH --array=1-12400
#SBATCH --partition=capella
#SBATCH --exclude=c36
# Load all Modules

source /data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/hpc/modules.sh
python3 -u /data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/classification.py --index $SLURM_ARRAY_TASK_ID