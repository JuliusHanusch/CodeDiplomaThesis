#!/bin/bash
#SBATCH --cpus-per-task=16 # CPU Count
#SBATCH --nodes=1
#SBATCH --mem=320G # Working Memory
#SBATCH --time=12:00:00  # Runtime HH:MM:SS
#SBATCH --account=p_automl  
#SBATCH --job-name=download_data
#SBATCH --array=0-2
#SBATCH --output=./hpc/logs/data_downloader_%A_%a.out  # Output Address 
#SBATCH --error=./hpc/logs/data_downloader_%A_%a.err  # Output Address
  
# Load all Modules
source ./hpc/modules_cpu.sh
out_dir="./data/corpus" # TODO Create Workspace

FILE_LIST=($(ls ./data/data_configs))
INPUT_FILE=${FILE_LIST[$SLURM_ARRAY_TASK_ID]}
  
srun python ./data/dataset_creation.py --config "$INPUT_FILE" --output-dir "$out_dir"  