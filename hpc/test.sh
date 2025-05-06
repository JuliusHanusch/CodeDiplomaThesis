#!/usr/bin/env bash

#SBATCH -J chronos_hpo
#SBATCH -e /data/horse/ws/jipo020b-aion/AION/hpc/logs/smac/chronos_hpo-%J.err
#SBATCH -o /data/horse/ws/jipo020b-aion/AION/hpc/logs/smac/chronos_hpo-%J.out
#SBATCH -A p_automl
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G
#SBATCH -t 00:05:00
#SBATCH --gres=gpu:1
cd /data/horse/ws/jipo020b-aion/AION
pwd
source ./hpc/modules.sh
/data/horse/ws/jipo020b-aion/AION/hpc/venvs/venv_gpu/bin/python -m distributed.cli.dask_worker tcp://172.24.74.124:35989 --name dummy-name --nthreads 1 --memory-limit 149.01GiB --no-nanny --death-timeout 60 --local-directory /data/horse/ws/jipo020b-aion/AION