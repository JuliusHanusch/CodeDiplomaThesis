#!/bin/bash
#SBATCH --cpus-per-task=24 # CPU Count
#SBATCH --nodes=1
#SBATCH --mem=400G # Working Memory
#SBATCH --time=36:00:00  # Runtime HH:MM:SS
#SBATCH --account=p_llm_timeseries  
#SBATCH --job-name=mixup
#SBATCH --array=0-99 #-2
#SBATCH --output=./hpc/logs/mixup%A_%a.out  # Output Address 
#SBATCH --error=./hpc/logs/mixup%A_%a.err  # Output Address

# - SBATCH --exclude=n1435,n1009,n1577,n1308,n1281,n1569,n1626,n1486
  
# Load all Modules
source ./hpc/modules_cpu.sh
out_dir="./data/train" # TODO Create Workspace
CONFIG_DIR="./data/data_configs"

FILE_LIST=($(ls "$CONFIG_DIR"))
INPUT_FILE=${FILE_LIST[$SLURM_ARRAY_TASK_ID]}
echo "${CONFIG_DIR}/${INPUT_FILE}"
  
python ./data/dataset_creation.py --config "${CONFIG_DIR}/${INPUT_FILE}" --output-dir "$out_dir" --samples 5000000 --workers -1