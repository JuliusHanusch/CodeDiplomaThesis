#load venv
echo 'export PATH="'"$SWIG_INSTALL_PREFIX"'/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

exec bash

#reload venv
# source hpc/modules.sh
# pip install smac

