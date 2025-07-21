#!/bin/bash
#SBATCH --cpus-per-task=16 # CPU Count
#SBATCH --nodes=1
#SBATCH --mem=320G # Working Memory
#SBATCH --time=12:00:00  # Runtime HH:MM:SS
#SBATCH --account=p_automl  
#SBATCH --job-name=mixup
#SBATCH --array=0 #-2
#SBATCH --output=./hpc/logs/mixup%A_%a.out  # Output Address 
#SBATCH --error=./hpc/logs/mixup%A_%a.err  # Output Address
  
# Load all Modules
source ./hpc/modules_cpu.sh
out_dir="./data/corpus" # TODO Create Workspace
CONFIG_DIR="./data/data_configs"

FILE_LIST=($(ls "$CONFIG_DIR"))
INPUT_FILE=${FILE_LIST[$SLURM_ARRAY_TASK_ID]}
echo "${CONFIG_DIR}/${INPUT_FILE}"
  
srun python ./data/dataset_creation.py --config "${CONFIG_DIR}/${INPUT_FILE}" --output-dir "$out_dir"  