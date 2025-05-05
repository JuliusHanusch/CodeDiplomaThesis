source hpc/modules.sh
python3 -m venv --system-site-package ./hpc/venvs/venv_gpu
source hpc/modules.sh
pip install -r hpc/venvs/requirements.txt