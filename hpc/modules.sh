#!/usr/bin/env bash

# Change To Root Directory
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module load release/24.04  GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.0
source ../venv/bin/activate