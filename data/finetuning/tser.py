import os
import requests
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from gluonts.dataset.arrow import ArrowWriter


ZENODO_RECORDS = [
    "3902637",
    "3902716",
    "3902728",
]

OUTPUT_DIR = "/content/CodeDiplomaThesis/data/finetuning/zenodo_tser"
RAW_TEST_DIR = os.path.join(OUTPUT_DIR, "raw_test")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_TEST_DIR, exist_ok=True)


# -------------------------
# Zenodo metadata fetch
# -------------------------
def get_record_files(record_id):
    url = f"https://zenodo.org/api/records/{record_id}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    files = data["files"]

    download_urls = []
    for f in files:
        download_urls.append((f["key"], f["links"]["self"]))

    return download_urls


# -------------------------
# download helper
# -------------------------
def download_file(url, out_path):
    print(f"Downloading {url}")
    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

    return out_path


# -------------------------
# extract if zip
# -------------------------
def maybe_extract(path, out_dir):
    if path.suffix == ".zip":
        print(f"Extracting {path}")
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(out_dir)
        return out_dir
    return path.parent


# -------------------------
# find TS files
# -------------------------
def find_files(folder):
    train_files, test_files = [], []

    for p in Path(folder).rglob("*"):
        name = p.name.lower()

        if "train" in name and p.suffix in [".ts", ".txt", ".csv"]:
            train_files.append(p)

        if "test" in name and p.suffix in [".ts", ".txt", ".csv"]:
            test_files.append(p)

    return train_files, test_files


# -------------------------
# TS parser (UEA/UCR format)
# -------------------------
def load_ts(path):
    if path.suffix == ".ts":
        # skip metadata header lines (@)
        data = []
        labels = []

        with open(path, "r") as f:
            for line in f:
                if line.startswith("@") or len(line.strip()) == 0:
                    continue
                parts = line.strip().split(":")[-1].split(",")
                labels.append(float(parts[0]))
                data.append(np.array(parts[1:], dtype=np.float32))

        return np.array(labels), np.array(data)

    df = pd.read_csv(path, header=None)
    return df.iloc[:, 0].values, df.iloc[:, 1:].values


# -------------------------
# Arrow conversion
# -------------------------
def to_arrow(labels, data):
    records = []

    for i in range(len(data)):
        ts = np.asarray(data[i], dtype=np.float32).squeeze()
        if ts.ndim > 1:
            ts = ts.reshape(-1)

        assert ts.ndim == 1, f"Bad shape {ts.shape}"

        records.append({
            "start": "1970-01-01 00:00:00",
            "target": ts,
            "item_id": str(i),
            "label": float(labels[i]),
        })

    return records


# -------------------------
# main processing
# -------------------------
def process_record(record_id):
    print(f"\n=== Processing {record_id} ===")

    files = get_record_files(record_id)

    work_dir = os.path.join(OUTPUT_DIR, record_id)
    os.makedirs(work_dir, exist_ok=True)

    downloaded = []

    for name, url in files:
        out_path = os.path.join(work_dir, name)
        download_file(url, out_path)
        extracted = maybe_extract(Path(out_path), work_dir)
        downloaded.append(extracted)

    train_files, test_files = find_files(work_dir)

    if not train_files or not test_files:
        raise RuntimeError(f"No train/test found in {record_id}")

    # load
    y_train, X_train = load_ts(train_files[0])
    y_test, X_test = load_ts(test_files[0])

    # train → arrow
    train_records = to_arrow(y_train, X_train)

    arrow_path = os.path.join(OUTPUT_DIR, f"{record_id}_train.arrow")

    ArrowWriter(compression="lz4").write_to_file(
        dataset=train_records,
        path=arrow_path,
    )

    print(f"Saved train: {arrow_path}")

    # test → raw
    np.savez_compressed(
        os.path.join(RAW_TEST_DIR, f"{record_id}_test.npz"),
        X=X_test,
        y=y_test,
    )

    print(f"Saved test raw: {record_id}")


def main():
    for rid in ZENODO_RECORDS:
        try:
            process_record(rid)
        except Exception as e:
            print(f"FAILED {rid}: {e}")


if __name__ == "__main__":
    main()