#!/bin/bash
#SBATCH --cpus-per-task=16 # CPU Count
#SBATCH --nodes=1
#SBATCH --mem=800G # Working Memory
#SBATCH --time=24:00:00  # Runtime HH:MM:SS
#SBATCH --account=p_automl  
#SBATCH --job-name=unit_test_prepro
#SBATCH --array=0
#SBATCH --output=./hpc/logs/unit_test_prepro%A_%a.out  # Output Address 
#SBATCH --error=./hpc/logs/unit_test_prepro%A_%a.err  # Output Address
  
# Load all Modules
source ./hpc/modules_cpu.sh
  
srun python data/unit_test_preprocessed_corpora.py 