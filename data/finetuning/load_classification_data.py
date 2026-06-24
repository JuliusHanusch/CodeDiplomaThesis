import os
from pathlib import Path
import urllib.request
import zipfile
import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
DATA_DIR = "data/finetuning/UCI_HAR"
ARROW_SAVE_PATH = "data/finetuning/UCI_HAR/UCI_HAR.arrow"
CONTEXT_LENGTH = 512  # your Chronos context length
COMPRESSION = "lz4"

print("Start")

os.makedirs(DATA_DIR, exist_ok=True)

print("Dir created")

# -----------------------------
# Download and extract dataset
# -----------------------------
zip_path = os.path.join(DATA_DIR, "UCI_HAR.zip")
if not os.path.exists(zip_path):
    print("Downloading UCI-HAR dataset...")
    urllib.request.urlretrieve(DATA_URL, zip_path)
    print("Download completed!")

extract_path = os.path.join(DATA_DIR, "UCI_HAR_Dataset")
if not os.path.exists(extract_path):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extraction completed!")

# -----------------------------
# Load dataset helper
# -----------------------------
def load_X(path):
    return np.loadtxt(path)

def load_y(path):
    return np.loadtxt(path).astype(int) - 1  # 0-based labels

def load_set(set_type="train"):
    print(f"Loading {set_type} set...")
    X_path = os.path.join(DATA_DIR, "UCI HAR Dataset", set_type, f"X_{set_type}.txt")
    y_path = os.path.join(DATA_DIR, "UCI HAR Dataset", set_type, f"y_{set_type}.txt")
    
    X = load_X(X_path)
    y = load_y(y_path)
    
    X = X.astype(np.float32)
    # truncate/pad to CONTEXT_LENGTH if needed
    if X.shape[1] < CONTEXT_LENGTH:
        pad_width = CONTEXT_LENGTH - X.shape[1]
        X = np.pad(X, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)
    elif X.shape[1] > CONTEXT_LENGTH:
        X = X[:, -CONTEXT_LENGTH:]
    
    # convert to list of dicts
    entries = []
    for i in tqdm(range(X.shape[0]), desc=f"Processing {set_type} samples"):
        entries.append({
            "target": X[i].astype(np.float32),
            "label": int(y[i]),
        })
    return entries

# -----------------------------
# Convert to Arrow format
# -----------------------------
def convert_to_arrow(path: Path, entries, compression: str = "lz4"):
    print(f"Converting {len(entries)} entries to Arrow format...")
    start_time = np.datetime64("2000-01-01 00:00", "s")
    dataset = []
    for entry in tqdm(entries, desc="Writing entries to Arrow"):
        dataset.append({"start": start_time, "target": entry["target"], "label": entry["label"]})
    ArrowWriter(compression=compression).write_to_file(dataset, path=path)
    print(f"Arrow file saved at: {path}")

# -----------------------------
# Process train/test and save
# -----------------------------
train_entries = load_set("train")

print(f"Writing to Arrow file: {ARROW_SAVE_PATH}")
convert_to_arrow(Path(ARROW_SAVE_PATH), train_entries, compression=COMPRESSION)

print("All done! Dataset stored in", ARROW_SAVE_PATH)