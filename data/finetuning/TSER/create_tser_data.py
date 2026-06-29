from pathlib import Path
import zipfile
from datetime import datetime
import numpy as np
import pandas as pd
from gluonts.dataset.arrow import ArrowWriter
from tqdm import tqdm

# import the TSER loader function you pasted
import sys
from pathlib import Path

REPO_ROOT = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/TS-Extrinsic-Regression")
sys.path.append(str(REPO_ROOT))

from utils.data_loader import load_from_tsfile_to_dataframe

ROOT = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/TSER")

CONTEXT_LENGTH = 512


# ---------------------------------------------------------
# convert pandas TSER format -> numpy tensor
# ---------------------------------------------------------

def dataframe_to_array(df):

    dims = [c for c in df.columns if c.startswith("dim_")]

    col = df[dims[0]].values  # pick one channel only early

    X = []
    for s in col:
        X.append(np.asarray(s.values, dtype=np.float32))

    return np.array(X, dtype=object)



def write_arrow(X, y, out_file):

    dataset = []

    for i in tqdm(range(len(X)), desc="Writing Arrow"):

        series = np.asarray(X[i], dtype=np.float32)

        # Keep the most recent CONTEXT_LENGTH values
        series = series[-CONTEXT_LENGTH:]

        # Left-pad with NaNs
        out = np.full(CONTEXT_LENGTH, np.nan, dtype=np.float32)
        out[-len(series):] = series

        dataset.append({
            "start": datetime(2000, 1, 1),
            "target": out,
            "label": float(y[i]),
        })

    ArrowWriter(compression="lz4").write_to_file(
        dataset,
        path=str(out_file),
    )

for zip_path in ROOT.glob("*.zip"):

    name = zip_path.stem
    out_dir = ROOT / name
    out_dir.mkdir(exist_ok=True)

    print(f"\nProcessing {name}")

    # unzip
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------
    train_file = next(out_dir.rglob("*TRAIN.ts"))
    print("Loading train:", train_file)

    df_train, y_train = load_from_tsfile_to_dataframe(
        train_file,
        return_separate_X_and_y=True
    )

    X_train = dataframe_to_array(df_train)

    print("Train shape:", X_train.shape)

    write_arrow(
        X_train,
        y_train,
        out_dir / "train.arrow"
    )

    # -----------------------------------------------------
    # TEST
    # -----------------------------------------------------
    test_file = next(out_dir.rglob("*TEST.ts"))
    print("Loading test:", test_file)

    df_test, y_test = load_from_tsfile_to_dataframe(
        test_file,
        return_separate_X_and_y=True
    )

    X_test = dataframe_to_array(df_test)

    print("Test shape:", X_test.shape)

    write_arrow(
        X_test,
        y_test,
        out_dir / "test.arrow"
    )

    print(f"Saved train + test for {name}")

print("\nDONE.")