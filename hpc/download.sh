#!/bin/bash
#SBATCH --cpus-per-task=4 # CPU Count
#SBATCH --nodes=1
#SBATCH --mem=320G # Working Memory
#SBATCH --time=12:00:00  # Runtime HH:MM:SS
#SBATCH --account=p_automl  
#SBATCH --job-name=download_data
#SBATCH --array=0-3
#SBATCH --output=./hpc/logs/data_downloader_%A_%a.out  # Output Address 
#SBATCH --error=./hpc/logs/data_downloader_%A_%a.err  # Output Address
  
# Load all Modules
source ./hpc/modules_cpu.sh

case $SLURM_ARRAY_TASK_ID in
  0) corpus="kaggle" ;;
  1) corpus="chronos" ;;
  2) corpus="lotsa" ;;
  3) corpus="uci" ;;
esac
  
srun python ./data/download_data.py --corpus-name "$corpus"