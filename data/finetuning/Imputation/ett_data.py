from pathlib import Path
import pandas as pd
import numpy as np
import urllib.request


OUTPUT_DIR = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Imputation"
)

RAW_DIR = OUTPUT_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


FILES = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETTDataset/main/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETTDataset/main/ETT-small/ETTm2.csv",
}


def download_files():

    for name, url in FILES.items():

        path = RAW_DIR / f"{name}.csv"

        if not path.exists():

            print(f"Downloading {name}")

            urllib.request.urlretrieve(
                url,
                path
            )

        else:
            print(f"{name} already exists")


def csv_to_univariate(path):

    df = pd.read_csv(path)

    values = df["OT"].values.astype(np.float32)

    return [values], ["OT"]


def temporal_split(series, name):

    if name.startswith("ETTh"):

        n_train = 12 * 30 * 24
        n_val = 4 * 30 * 24
        n_test = 4 * 30 * 24

    elif name.startswith("ETTm"):

        n_train = 12 * 30 * 24 * 4
        n_val = 4 * 30 * 24 * 4
        n_test = 4 * 30 * 24 * 4

    seq_len = 512

    train = []
    val = []
    test = []

    for x in series:

        # First 12 months
        train.append(
            x[:n_train]
        )

        # Following 4 months
        val.append(
            x[n_train:n_train + n_val]
        )

        # Final 4 months
        # Include 512 points before the test period
        # as context for the first test window.
        test.append(
            x[n_train + n_val - seq_len:
              n_train + n_val + n_test]
        )

    return train, val, test


def save_npz(path, series, names):

    np.savez_compressed(
        path,
        series=np.array(
            series,
            dtype=object
        ),
        names=np.array(names)
    )

    print(
        "Saved:",
        path
    )


download_files()


for name in FILES:

    print("\n================")
    print(name)
    print("================")

    csv_path = RAW_DIR / f"{name}.csv"

    series, names = csv_to_univariate(
        csv_path
    )

    train, val, test = temporal_split(
        series,
        name
    )

    out = OUTPUT_DIR / name
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    save_npz(
        out / "train.npz",
        train,
        names
    )

    save_npz(
        out / "val.npz",
        val,
        names
    )

    save_npz(
        out / "test.npz",
        test,
        names
    )