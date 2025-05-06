#!/bin/bash
#SBATCH --cpus-per-task=4 # CPU Count
#SBATCH --nodes=1
#SBATCH --mem=320G # Working Memory
#SBATCH --time=1:00:00  # Runtime HH:MM:SS
#SBATCH --account=p_automl  
#SBATCH --job-name=test
#SBATCH --array=0
#SBATCH --output=./hpc/logs/test_%A_%a.out  # Output Address 
#SBATCH --error=./hpc/logs/test_%A_%a.err  # Output Address
  
# Load all Modules
source ./hpc/modules_cpu.sh

python3 /data/horse/ws/jipo020b-aion/AION/code/search_space.py
