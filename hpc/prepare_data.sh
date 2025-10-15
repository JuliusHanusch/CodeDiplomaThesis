
# TODO Manually setup credentials
# Setup CPU Venv
jid1=$(sbatch hpc/modules_cpu.sh | awk '{print $4}')
# Download Data
jid2a=$(sbatch --dependency=afterok:$jid1 hpc/download.sh | awk '{print $4}') # ! Might need to be run on 2 days Kaggle might block you
jid2b=$(sbatch --dependency=afterok:$jid2a --begin=now+1day hpc/download.sh) # Repeat after 1 Day
# prepare TS Corpus
jid3=$(sbatch --dependency=afterok:$jid2b hpc/preprocess.sh)
# Create several mixtures (by starting a array job)
jid4=$(sbatch --dependency=afterok:$jid3 hpc/mixup.sh)
