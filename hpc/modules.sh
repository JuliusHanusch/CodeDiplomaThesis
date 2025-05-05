#!/usr/bin/env bash

# Change To Root Directory
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd ..
module load release/24.04  GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.0
source ./hpc/venvs/venv_gpu/bin/activate