from pathlib import Path
import datetime as dt
from typing import Dict, List

import pandas as pd
import datasets as ds
from tqdm.auto import tqdm

from eval_frequency import eval_frequency
from typer import Typer
from pandas.tseries.frequencies import to_offset
from warnings import warn

import cProfile
import pstats

app = Typer()


def resample_by_freq(df: pd.DataFrame, time_col: str, freq: str = '1s') -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    # Compute maximum safe timestamp
    offset = to_offset(freq)
    try:
        delta = offset.delta  # for fixed frequencies like '1D'
    except AttributeError:
        delta = pd.Timedelta('0s')  # non-fixed (e.g. 'M', 'W') are handled differently

    max_allowed = pd.Timestamp.max - delta

    # Drop any value that would overflow when rounded up
    df = df[df[time_col] <= max_allowed]

    df.set_index(time_col, inplace=True)

    try:
        is_fixed = not offset.is_anchored and hasattr(offset, 'delta')
    except AttributeError:
        is_fixed = False

    if is_fixed:
        df.index = df.index.floor(freq)
        result = df.groupby(df.index).median().resample(freq).asfreq()
    else:
        result = df.resample(freq).median()

    result.reset_index(inplace=True)
    result.rename(columns={"index": time_col}, inplace=True)
    return result



def to_datetime(df: pd.DataFrame, time_column) -> pd.DataFrame:
    """Vector-parse & round datetimes (μs precision)."""
    sample = df[time_column].dropna().astype(str).iloc[0]
    ms = '.' in sample
    fmt = "%Y-%m-%d %H:%M:%S.%f" if ms else "%Y-%m-%d %H:%M:%S"
    df[time_column] = pd.to_datetime(df[time_column], format=fmt, errors="coerce").dt.round("us")
    return df


def get_next_coarser_freq(series: pd.Series) -> str:
    # Ensure datetime
    series = pd.to_datetime(series).dropna().sort_values()
    diffs = series.diff().dropna()

    # Get median delta
    median_delta = diffs.median()

    # Thresholds in timedelta
    one_sec = pd.Timedelta('1s')
    one_min = pd.Timedelta('1min')
    one_hour = pd.Timedelta('1h')
    one_day = pd.Timedelta('1D')
    one_week = pd.Timedelta('7D')
    one_month = pd.Timedelta('30D')  # approx
    one_quarter = pd.Timedelta('90D')
    one_year = pd.Timedelta('365D')  # approx

    # Coarsening logic
    if median_delta < one_sec:
        return 's'
    elif median_delta < one_min:
        return 'T'  # minute
    elif median_delta < one_hour:
        return 'H'
    elif median_delta < one_day:
        return 'D'
    elif median_delta < one_week:
        return 'W'
    elif median_delta < one_month:
        return 'ME' # Month End
    elif median_delta < one_quarter:
        return 'QE'
    elif median_delta < one_year:
        return 'YE'
    else:
        return '10Y'  # catch-all for very sparse data


def equidise(df: pd.DataFrame, min_ts_length, time_column, downsample_freq=None):
    """Takes in a TS DF and aggregates its entrys until it either becomes equidistant or it becomes to short"""
    if downsample_freq is not None:# Decrease Frequency 
        # (Done like that To Fix Median of Medians through true recursiveness)
        df_ = resample_by_freq(df, time_column, downsample_freq)
    else:
        df_ = df.copy()

    if len(df_) < min_ts_length:
        return pd.DataFrame([]) # Too Short return empty
    if eval_frequency(df_): # If equidistant return
        return df_
    else: # If not aggregate to next Coarser Freq and check again
        # Identify next coarser frequency to test
        next_coarser_freq = get_next_coarser_freq(df_[time_column])
        del df_ # Free Up Mem Again
        return equidise(df, min_ts_length, time_column=time_column, downsample_freq=next_coarser_freq)


def make_univar(name: str, df: pd.DataFrame, col: str) -> ds.Dataset:
    ident = f"{name}_{col}" if col != "value" else name
    return ds.Dataset.from_dict(
        {
            "identifier": [ident],
            "datetime": [df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").tolist()],
            "value": [df[col].astype(float).tolist()],  # or .astype(str) if needed
        }
    )


def shift_df_into_datetime_bounds(df: pd.DataFrame, time_col: str) -> tuple[pd.DataFrame, float, pd.Timestamp]:
    warn("Check Shift")
    t_min, t_max = df[time_col].min().to_pydatetime(), df[time_col].max().to_pydatetime()
    t_min_allowed = pd.Timestamp.min + pd.to_timedelta('1s')
    t_max_allowed = pd.Timestamp.max - pd.to_timedelta('1s')

    if t_min > t_min_allowed and t_max < t_max_allowed:
        return df, 1.0, pd.Timestamp(0)  # no scaling needed
    
    warn(f"Time Series out of bounds with StartTime {t_min} and EndTime {t_max} going to min max scale it into viable range")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df[df[time_col].notna()]

    # Min Max Scaling (When Dresired Range != 0-1 and we are mem bound)
    # Total original duration (Actual Range)
    duration = (t_max - t_min).total_seconds()

    # Max representable duration in pandas (Desired Range)
    safe_duration = (t_max_allowed.to_pydatetime() - t_min_allowed.to_pydatetime()).total_seconds()

    # Compute scaling factor
    factor = safe_duration / duration

    def scale_ts(ts):
        seconds = int((ts - t_min).total_seconds() * factor) 
        # pd.Timestamp.min + pd.to_timedelta(seconds, unit='s') ~ But timedelta is signed -> range is halved 
        d = t_min_allowed
        for i in range(2): # We add twice half of the total we aim to add (-1ns)
            d += pd.to_timedelta(seconds.total_seconds()*0.5-0.000001, unit='s', errors="coerce")
        return d

    df[time_col] = df[time_col].apply(scale_ts)
    df = df.dropna(subset=[time_col])

    t_min, t_max = df[time_col].min().to_pydatetime(), df[time_col].max().to_pydatetime()
    assert t_min >= t_min_allowed and t_max <= t_max_allowed

    return df, factor, t_min



@app.command()
def preprocessing(min_ts_length: int = 64, RAW_DIR: str = "./data/data_sets_raw/Time_Corpus") -> None:
    assert min_ts_length > 0, "min_ts_length must be a positive integer"
    PROCESSED_DIR = Path(RAW_DIR+"_Processed")
    RAW_DIR = Path(RAW_DIR)


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
        df = to_datetime(df, "datetime")
        if df["datetime"].isna().any():
            issues["MalformedTimeColumn"].append(rec["name"])
            continue

        df = df.dropna(subset=["datetime"])
        # if df["datetime"].max() > pd.Timestamp.max - pd.to_timedelta('1s'): # Check if out of bounds
        df, _, _ = shift_df_into_datetime_bounds(df, "datetime") # Use entire possible intervall

        df = equidise(df, min_ts_length, time_column="datetime", downsample_freq=None)
        if len(df) > min_ts_length:
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
    try:
        with cProfile.Profile() as pr:
            app()
        stats = pstats.Stats(pr)
        stats.dump_stats("./hpc/logs/profile.prof")
    except Exception as E:
        stats = pstats.Stats(pr)
        stats.dump_stats("./hpc/logs/profile.prof")
        raise
