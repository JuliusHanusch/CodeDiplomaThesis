from pathlib import Path
import datetime as dt
from typing import Dict, List

import pandas as pd
import datasets as ds
from tqdm.auto import tqdm

from eval_frequency import eval_frequency

RAW_DIR = Path("./data/data_sets_raw/Time_Corpus")
PROCESSED_DIR = Path("./data/data_sets_raw/Time_Corpus_Processed")


def to_datetime(series: pd.Series) -> pd.Series:
    """Vector-parse & round datetimes (μs precision)."""
    return (pd.to_datetime(series, format="%Y-%m-%d %H:%M:%S", errors="coerce")
              .dt.round("us"))


def make_univar(name: str, df: pd.DataFrame, col: str) -> ds.Dataset:
    ident = f"{name}_{col}" if col != "value" else name
    return ds.Dataset.from_dict(
        {
            "identifier": [ident],
            "datetime": [df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").tolist()],
            "value": [df[col].astype(float).tolist()],  # or .astype(str) if needed
        }
    )



def preprocessing(min_ts_length=64) -> None:
    assert min_ts_length > 0, "min_ts_length must be a positive integer"


    src = ds.load_from_disk(RAW_DIR)["train"]
    total = len(src)

    stats = {"equi": 0, "nonequi": 0}
    issues: Dict[str, List[str]] = {"EmptyValueColumn": [], "MalformedTimeColumn": []}
    processed: List[ds.Dataset] = []

    for rec in tqdm(src, desc="processing"):
        if not rec["value"]:
            issues["EmptyValueColumn"].append(rec["name"])
            continue

        df = pd.DataFrame(
            {"datetime": eval(rec["date"]),
             **{f"value_{i}": v for i, v in enumerate(rec["value"])}}
        )

        # drop sentinel rows & parse datetimes
        df = df[~df.astype(str).apply(
            lambda r: r.str.contains("0000-01-01 00:00:00", na=False)).any(axis=1)]
        df["datetime"] = to_datetime(df["datetime"])
        if df["datetime"].isna().any():
            issues["MalformedTimeColumn"].append(rec["name"])
            continue

        if eval_frequency(df):
            stats["equi"] += 1
            val_cols = df.columns.difference(["datetime"])
            processed += [make_univar(rec["name"], df, c) for c in val_cols if len(df[c]) >= min_ts_length]
        else:
            stats["nonequi"] += 1

    ds.DatasetDict(train=ds.concatenate_datasets(processed)).save_to_disk(PROCESSED_DIR)

    print(stats)
    print(stats["equi"] / total)
    print(issues)


if __name__ == "__main__":
    preprocessing()
