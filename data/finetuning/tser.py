import os
import requests
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from gluonts.dataset.arrow import ArrowWriter


ZENODO_URLS = [
    "https://zenodo.org/records/3902637/files/archive.zip",
    "https://zenodo.org/records/3902716/files/archive.zip",
    "https://zenodo.org/records/3902728/files/archive.zip",
]

OUTPUT_DIR = "/content/CodeDiplomaThesis/data/finetuning/zenodo_tser"
RAW_TEST_DIR = os.path.join(OUTPUT_DIR, "raw_test")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_TEST_DIR, exist_ok=True)


def download_file(url, out_path):
    print(f"Downloading {url}")
    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    return out_path


def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)


def find_train_test_files(folder):
    train_files, test_files = [], []

    for path in Path(folder).rglob("*"):
        name = path.name.lower()
        if "train" in name and path.suffix in [".txt", ".csv"]:
            train_files.append(path)
        if "test" in name and path.suffix in [".txt", ".csv"]:
            test_files.append(path)

    return train_files, test_files


def load_table(path):
    if path.suffix == ".csv":
        return pd.read_csv(path, header=None)
    else:
        return pd.read_csv(path, header=None, sep=r"\s+")


def convert_row(row, idx):
    arr = np.asarray(row[1:], dtype=np.float32)

    # ensure 1D
    arr = np.squeeze(arr)
    if arr.ndim > 1:
        arr = arr.reshape(-1)

    assert arr.ndim == 1, f"Bad shape: {arr.shape}"

    return {
        "start": "1970-01-01 00:00:00",
        "target": arr,
        "item_id": str(idx),
        "label": float(row[0]),
    }


def process_dataset(name, train_file, test_file):
    print(f"\n=== Processing dataset: {name} ===")

    # -------- TRAIN --------
    train_df = load_table(train_file)

    records = []
    for i, row in train_df.iterrows():
        records.append(convert_row(row.values, i))

    arrow_path = os.path.join(OUTPUT_DIR, f"{name}_train.arrow")

    ArrowWriter(compression="lz4").write_to_file(
        dataset=records,
        path=arrow_path,
    )

    print(f"Saved train arrow: {arrow_path} ({len(records)} samples)")

    # -------- TEST (RAW SAVE) --------
    test_df = load_table(test_file)
    test_arr = test_df.to_numpy()

    raw_path = os.path.join(RAW_TEST_DIR, f"{name}_test.npz")
    np.savez_compressed(raw_path, data=test_arr)

    print(f"Saved test raw: {raw_path} ({len(test_arr)} samples)")


def main():
    base_dir = os.path.join(OUTPUT_DIR, "downloads")
    os.makedirs(base_dir, exist_ok=True)

    for url in ZENODO_URLS:
        dataset_name = url.split("/")[-2]

        zip_path = os.path.join(base_dir, f"{dataset_name}.zip")
        extract_path = os.path.join(base_dir, dataset_name)

        try:
            download_file(url, zip_path)
            extract_zip(zip_path, extract_path)

            train_files, test_files = find_train_test_files(extract_path)

            if not train_files or not test_files:
                raise RuntimeError(
                    f"Could not find train/test files in {extract_path}"
                )

            # pick first match (can be extended if multiple exist)
            process_dataset(
                dataset_name,
                train_files[0],
                test_files[0],
            )

        except Exception as e:
            print(f"FAILED {dataset_name}: {e}")


if __name__ == "__main__":
    main()