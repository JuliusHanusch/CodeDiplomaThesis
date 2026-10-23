#!/bin/bash
#SBATCH --cpus-per-task=4 # CPU Count
#SBATCH --gres=gpu:1 # The Master Job actually doesn't need one but HPC demands it else we can't schedule workers with GPU
#SBATCH --nodes=1
#SBATCH --mem=160G # Working Memory
#SBATCH --time=10:00:00  # Runtime HH:MM:SS
#SBATCH --account=p_sci_debt
#SBATCH --job-name=Similarity
#SBATCH --output=hpc/logs/finetuning/Similarity/Similarity-%j-%a.out  # Output Address 
#SBATCH --error=hpc/logs//finetuning/Similarity/Similarity-%j-%a.err  # Output Address
#SBATCH --partition=capella
#SBATCH --array=1-12400
#SBATCH --exclude=c36

source /data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/hpc/modules.sh
python3 /data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/similarity/similarity.py --index $SLURM_ARRAY_TASK_ID
