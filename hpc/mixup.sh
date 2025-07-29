#!/bin/bash
#SBATCH --cpus-per-task=16 # CPU Count
#SBATCH --nodes=1
#SBATCH --mem=900G # Working Memory
#SBATCH --time=24:00:00  # Runtime HH:MM:SS
#SBATCH --account=p_automl  
#SBATCH --job-name=mixup
#SBATCH --array=0 #-2
#SBATCH --output=./hpc/logs/mixup%A_%a.out  # Output Address 
#SBATCH --error=./hpc/logs/mixup%A_%a.err  # Output Address
  
# Load all Modules
source ./hpc/modules_cpu.sh
out_dir="./data/train" # TODO Create Workspace
CONFIG_DIR="./data/data_configs"

FILE_LIST=($(ls "$CONFIG_DIR"))
INPUT_FILE=${FILE_LIST[$SLURM_ARRAY_TASK_ID]}
echo "${CONFIG_DIR}/${INPUT_FILE}"
  
py-spy record -o profile.svg -- python ./data/dataset_creation.py --config "${CONFIG_DIR}/${INPUT_FILE}" --output-dir "$out_dir" --samples 25000000 --workers 8