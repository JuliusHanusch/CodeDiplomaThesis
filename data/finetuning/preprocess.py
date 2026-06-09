import shutil
import numpy as np
import pandas as pd
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

BASE = Path("data/finetuning/smap_msl/data/data")

TRAIN_DIR = BASE / "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/smap_msl/data/data/train"
TEST_DIR = BASE / "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/smap_msl/data/data/test"
CSV_PATH = BASE / "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/smap_msl/data/data/labeled_anomalies.csv"

OUT = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning")
OUT.mkdir(parents=True, exist_ok=True)

# =====================================================
# LOAD METADATA
# =====================================================

df = pd.read_csv(CSV_PATH)

smap_ids = set(df[df["spacecraft"] == "SMAP"]["chan_id"].astype(str))
msl_ids  = set(df[df["spacecraft"] == "MSL"]["chan_id"].astype(str))

# =====================================================
# CREATE FOLDERS
# =====================================================

def make_dirs(name):
    (OUT / name / "train").mkdir(parents=True, exist_ok=True)
    (OUT / name / "test").mkdir(parents=True, exist_ok=True)
    (OUT / name / "test_labels").mkdir(parents=True, exist_ok=True)

make_dirs("SMAP")
make_dirs("MSL")

# =====================================================
# COPY FILES
# =====================================================

def process_file(f, split, dataset):
    target = OUT / dataset / split / f.name
    shutil.copy(f, target)

# =====================================================
# PROCESS TRAIN/TEST
# =====================================================

for f in TRAIN_DIR.glob("*.npy"):
    name = f.stem

    if name in smap_ids:
        process_file(f, "train", "SMAP")
    elif name in msl_ids:
        process_file(f, "train", "MSL")

for f in TEST_DIR.glob("*.npy"):
    name = f.stem

    if name in smap_ids:
        process_file(f, "test", "SMAP")
    elif name in msl_ids:
        process_file(f, "test", "MSL")

# =====================================================
# TEST LABELS (FROM CSV)
# =====================================================

def save_labels(dataset):
    sub = df[df["spacecraft"] == dataset]

    for _, row in sub.iterrows():
        fname = row["chan_id"]

        # anomaly intervals (string format in dataset)
        labels_path = OUT / dataset / "test_labels" / f"{fname}.txt"

        with open(labels_path, "w") as f:
            f.write(str(row["anomaly_sequences"]))

save_labels("SMAP")
save_labels("MSL")

print("DONE")
print("SMAP + MSL benchmark structure reconstructed")