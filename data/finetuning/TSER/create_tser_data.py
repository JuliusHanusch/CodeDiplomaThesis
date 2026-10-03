from pathlib import Path
import zipfile
from datetime import datetime
import numpy as np
import pandas as pd
from gluonts.dataset.arrow import ArrowWriter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys

REPO_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "TS-Extrinsic-Regression"
)

sys.path.append(str(REPO_ROOT))

from utils.data_loader import load_from_tsfile_to_dataframe


ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/data/finetuning/TSER"
)

TRAIN_RATIO = 0.80
EVAL_RATIO = 0.20
RANDOM_SEED = 42



def dataframe_to_array(df, multivariate=False):

    dims = [
        c for c in df.columns
        if c.startswith("dim_")
    ]

    X = []

    for i in range(len(df)):

        if multivariate:

            series = []

            for dim in dims:

                values = np.asarray(
                    df.iloc[i][dim].values,
                    dtype=np.float32
                )

                series.append(values)

            # Maximum length among channels
            max_len = max(
                len(x) for x in series
            )

            padded = []

            for x in series:

                arr = np.full(
                    max_len,
                    np.nan,
                    dtype=np.float32
                )

                arr[:len(x)] = x
                padded.append(arr)

            X.append(
                np.stack(padded, axis=0)
            )

        else:

            values = np.asarray(
                df.iloc[i][dims[0]].values,
                dtype=np.float32
            )

            X.append(values)

    return np.array(X, dtype=object)


def write_arrow(X, y, out_file, multivariate=False):

    dataset = []

    for i in tqdm(
        range(len(X)),
        desc=f"Writing {out_file.name}"
    ):

        series = np.asarray(
            X[i],
            dtype=np.float32
        )

        if multivariate:
            target = series
        else:
            target = series

        dataset.append({
            "start": datetime(2000, 1, 1),
            "target": target,
            "label": float(y[i]),
        })

    ArrowWriter(
        compression="lz4"
    ).write_to_file(
        dataset,
        path=str(out_file),
    )

def split_train_eval(df, y):

    try:

        train_df, eval_df, train_y, eval_y = train_test_split(
            df,
            y,
            train_size=TRAIN_RATIO,
            test_size=EVAL_RATIO,
            random_state=RANDOM_SEED,
            stratify=y
        )


    except ValueError as e:

        print(
            f"        [WARNING] Stratified split failed:"
            f"\n        {e}"
        )

        print(
            "        Falling back to random split."
        )

        train_df, eval_df, train_y, eval_y = train_test_split(
            df,
            y,
            train_size=TRAIN_RATIO,
            test_size=EVAL_RATIO,
            random_state=RANDOM_SEED,
            shuffle=True
        )

    return (
        train_df,
        eval_df,
        np.asarray(train_y),
        np.asarray(eval_y)
    )

def load_tser_file(file_path):

    try:
        return load_from_tsfile_to_dataframe(
            file_path,
            return_separate_X_and_y=True,
            encoding="utf-8"
        )

    except UnicodeDecodeError:

        print(
            f"[WARNING] UTF-8 decoding failed: {file_path}"
        )
        print(
            "          Retrying with cp1252."
        )

        return load_from_tsfile_to_dataframe(
            file_path,
            return_separate_X_and_y=True,
            encoding="cp1252"
        )

def process_dataset(zip_path):

    name = zip_path.stem

    out_dir = ROOT / name
    out_dir.mkdir(exist_ok=True)


    required_files = [
        "train_multi.arrow",
        "train_uni.arrow",
        "train_small_multi.arrow",
        "train_small_uni.arrow",
        "eval_multi.arrow",
        "eval_uni.arrow",
        "test_multi.arrow",
        "test_uni.arrow",
    ]

    if all((out_dir / file).exists() for file in required_files):
        print(f"[SKIP] {name}: already processed")
        return


    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as z:

        z.extractall(out_dir)



    train_file = next(
        out_dir.rglob("*TRAIN.ts")
    )

    print(
        "Loading train:",
        train_file
    )

    df_train, y_train = load_tser_file(train_file)

    print(
        f"Original TRAIN samples: "
        f"{len(df_train)}"
    )


    print()
    print("Writing FULL TRAIN")


    X_train = dataframe_to_array(
        df_train,
        multivariate=True
    )

    write_arrow(
        X_train,
        y_train,
        out_dir / "train_multi.arrow",
        multivariate=True
    )



    X_train_uni = dataframe_to_array(
        df_train,
        multivariate=False
    )

    write_arrow(
        X_train_uni,
        y_train,
        out_dir / "train_uni.arrow",
        multivariate=False
    )


    print()
    print("Splitting TRAIN 80/20")

    (
        df_train_small,
        df_eval,
        y_train_small,
        y_eval
    ) = split_train_eval(
        df_train,
        y_train
    )

    print(
        f"        Train small: "
        f"{len(df_train_small)}"
        f" | Eval: "
        f"{len(df_eval)}"
    )


    X_train_small = dataframe_to_array(
        df_train_small,
        multivariate=True
    )

    write_arrow(
        X_train_small,
        y_train_small,
        out_dir / "train_small_multi.arrow",
        multivariate=True
    )


    X_train_small_uni = dataframe_to_array(
        df_train_small,
        multivariate=False
    )

    write_arrow(
        X_train_small_uni,
        y_train_small,
        out_dir / "train_small_uni.arrow",
        multivariate=False
    )


    X_eval = dataframe_to_array(
        df_eval,
        multivariate=True
    )

    write_arrow(
        X_eval,
        y_eval,
        out_dir / "eval_multi.arrow",
        multivariate=True
    )


    X_eval_uni = dataframe_to_array(
        df_eval,
        multivariate=False
    )

    write_arrow(
        X_eval_uni,
        y_eval,
        out_dir / "eval_uni.arrow",
        multivariate=False
    )


    test_file = next(
        out_dir.rglob("*TEST.ts")
    )

    print()
    print(
        "Loading test:",
        test_file
    )

    df_test, y_test = load_tser_file(test_file)

    print(
        f"        Test samples: "
        f"{len(df_test)}"
    )


    X_test = dataframe_to_array(
        df_test,
        multivariate=True
    )

    write_arrow(
        X_test,
        y_test,
        out_dir / "test_multi.arrow",
        multivariate=True
    )

    X_test_uni = dataframe_to_array(
        df_test,
        multivariate=False
    )

    write_arrow(
        X_test_uni,
        y_test,
        out_dir / "test_uni.arrow",
        multivariate=False
    )


    print()
    print(f"Finished {name}")



def main():

    if not ROOT.exists():

        raise FileNotFoundError(
            f"TSER root not found:\n{ROOT}"
        )

    zip_files = sorted(
        ROOT.glob("*.zip")
    )

    for zip_path in zip_files:

        process_dataset(zip_path)


if __name__ == "__main__":
    main()