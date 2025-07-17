#!/bin/bash
#SBATCH --cpus-per-task=16 # CPU Count
#SBATCH --nodes=1
#SBATCH --mem=800G # Working Memory
#SBATCH --time=48:00:00  # Runtime HH:MM:SS
#SBATCH --account=p_automl  
#SBATCH --job-name=prepare_data
#SBATCH --array=0-1
#SBATCH --output=./hpc/logs/data_prep_%A_%a.out  # Output Address 
#SBATCH --error=./hpc/logs/data_prep_%A_%a.err  # Output Address
  
# Load all Modules
source ./hpc/modules_cpu.sh

case $SLURM_ARRAY_TASK_ID in
  0) corpus="./data/data_sets_raw/Time_Corpus" ;;
  1) corpus="./data/data_sets_raw/UCI_Corpus" ;;
esac
  
# TODO Allow For Splitting with min length 128 dont allow 
srun python ./data/preprocessing.py --raw-dir "$corpus" --min-ts-length 128