source hpc/modules_cpu.sh
python3 -m venv --system-site-package ./hpc/venvs/venv_cpu
source hpc/modules_cpu.sh
pip install -r hpc/venvs/requirements_cpu.txt