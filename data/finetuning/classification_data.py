import zipfile
from pathlib import Path
import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from tqdm import tqdm
from datetime import datetime


# -----------------------------
# FULL PATHS
# -----------------------------
#BASE = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning")
#colab
BASE = Path("/content/CodeDiplomaThesis/data/finetuning")


ARCHIVE_PATH = BASE / "UCRArchive.zip"
EXTRACT_ROOT = BASE / "UCR_extracted"
DATA_ROOT = EXTRACT_ROOT / "UCRArchive_2018"
ARROW_DIR = BASE / "UCR_arrow"

DATASETS = ["GestureMidAirD2", "DistalPhalanxTW", "ArrowHead"]

CONTEXT_LENGTH = 512
COMPRESSION = "lz4"
PASSWORD = b"someone"

ARROW_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# EXTRACT
# -----------------------------
def extract_zip():
    if DATA_ROOT.exists():
        print("Already extracted.")
        return

    print("Extracting...")

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as z:
        z.extractall(EXTRACT_ROOT, pwd=PASSWORD)


# -----------------------------
# LOAD TSV (IMPORTANT FIX)
# -----------------------------
def load_ucr_tsv(path: Path):
    data = np.loadtxt(path, delimiter="\t")
    y = data[:, 0].astype(int)
    X = data[:, 1:].astype(np.float32)
    return X, y


# -----------------------------
# FIX LENGTH
# -----------------------------
def fix_length(X):
    if X.shape[1] < CONTEXT_LENGTH:
        pad = CONTEXT_LENGTH - X.shape[1]
        X = np.pad(X, ((0, 0), (0, pad)))
    else:
        X = X[:, -CONTEXT_LENGTH:]
    return X


# -----------------------------
# ARROW WRITER
# -----------------------------

def write_arrow(path: Path, X, y):
    print(f"Writing -> {path}")

    start = datetime(2000, 1, 1)

    dataset = []
    for i in tqdm(range(len(X))):
        dataset.append({
            "start": start,  # FIX: python datetime (NOT numpy)
            "target": X[i].astype(np.float32).tolist(),  # FIX: ensure pure python list
            "label": int(y[i]),
        })

    ArrowWriter(compression=COMPRESSION).write_to_file(dataset, path=path)


# -----------------------------
# DATASET PATH (FULL PATH ONLY)
# -----------------------------
def get_dataset(ds_name: str):
    ds_path = DATA_ROOT / ds_name

    train = ds_path / f"{ds_name}_TRAIN.tsv"
    test = ds_path / f"{ds_name}_TEST.tsv"

    if not train.exists():
        raise FileNotFoundError(f"Missing: {train}")

    if not test.exists():
        raise FileNotFoundError(f"Missing: {test}")

    return train, test


# -----------------------------
# MAIN
# -----------------------------
def main():
    extract_zip()

    for ds in DATASETS:
        print("\n==============================")
        print(ds)
        print("==============================")

        train_path, test_path = get_dataset(ds)

        # ---- TRAIN ----
        X_train, y_train = load_ucr_tsv(train_path)
        X_train = fix_length(X_train)

        out = ARROW_DIR / f"{ds}_train.arrow"
        write_arrow(out, X_train, y_train)

        # ---- TEST (only load check) ----
        X_test, y_test = load_ucr_tsv(test_path)
        print("Test samples:", len(X_test))


if __name__ == "__main__":
    main()