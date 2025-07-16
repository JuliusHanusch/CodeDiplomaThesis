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
import sys

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List
import pandas as pd
from tqdm import tqdm
from tqdm.auto import tqdm as tqdma
from multiprocessing import cpu_count
import numpy as np 

app = Typer(pretty_exceptions_enable=False)
t_min_allowed = pd.Timestamp.min + pd.to_timedelta('1s')
t_max_allowed = pd.Timestamp.max - pd.to_timedelta('1s')


def resample_by_freq(df: pd.DataFrame, time_col: str, freq: str = '1s') -> pd.DataFrame:
    print(f"Resample to: {freq}")
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    # Compute maximum safe timestamp
    offset = to_offset(freq)
    try:
        delta = pd.to_timedelta(str(offset))
    except (ValueError, TypeError, AttributeError):
        delta = pd.Timedelta('1D')  # non-fixed (e.g. 'M', 'W') are handled differently

    # Resetting Freq. Might introduce values Out Of Bouns thanks to rounding - Lets try to avoid that
    max_allowed = pd.Timestamp.max - delta
    min_allowed = pd.Timestamp.min + delta

    # Drop any value that would overflow when rounded up
    df = df[(df[time_col] >= min_allowed) & (df[time_col] <= max_allowed)]

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
    # Drop Duplicates else some TS have median dist 0 and an endless loop will happen agg to seconds over and over 
    series = pd.to_datetime(series).dropna().drop_duplicates().sort_values() 
    diffs = series.diff().dropna()

    # Get median delta
    median_delta = diffs.median()

    # Thresholds in timedelta
    one_ms = pd.Timedelta('1ms')
    ten_ms = pd.Timedelta('10ms')
    hundred_ms = pd.Timedelta('100ms')
    one_sec = pd.Timedelta('1s')
    one_min = pd.Timedelta('1min')
    one_hour = pd.Timedelta('1h')
    one_day = pd.Timedelta('1D')
    one_week = pd.Timedelta('7D')
    one_month = pd.Timedelta('30D')  # approx
    one_quarter = pd.Timedelta('90D')
    one_year = pd.Timedelta('365D')  # approx

    # Coarsening logic
    if median_delta < one_ms:
        return '1ms'
    elif median_delta < ten_ms:
        return '10ms'
    elif median_delta < hundred_ms:
        return '100ms'
    elif median_delta < one_sec:
        return '1s'
    elif median_delta < one_min:
        return '1min'  # minute
    elif median_delta < one_hour:
        return '1h'
    elif median_delta < one_day:
        return '1D'
    elif median_delta < one_week:
        return '1W'
    elif median_delta < one_month:
        return 'ME' # Month End
    elif median_delta < one_quarter:
        return 'QE'
    elif median_delta < one_year:
        return 'YE'
    else:
        return '10YE'  # catch-all for very sparse data


def equidise(df: pd.DataFrame, min_ts_length, time_column, downsample_freq=None):
    """Takes in a TS DF and aggregates its entrys until it either becomes equidistant or it becomes to short"""
    if downsample_freq is not None:# Decrease Frequency 
        # (Done like that To Fix Median of Medians through true recursiveness)
        df_ = resample_by_freq(df, time_column, downsample_freq)
    else:
        df_ = df.copy()

    if len(df_) < min_ts_length:
        return pd.DataFrame([]) # Too Short return empty
    if eval_frequency(df_, time_column="datetime"): # If equidistant return
        return df_
    else: # If not aggregate to next Coarser Freq and check again
        # Identify next coarser frequency to test
        next_coarser_freq = get_next_coarser_freq(df_[time_column])
        del df_ # Free Up Mem Again
        return equidise(df, min_ts_length, time_column=time_column, downsample_freq=next_coarser_freq)


def make_univar(name: str, df: pd.DataFrame, cols: list[str]) -> ds.Dataset:
    return ds.Dataset.from_dict(
        {
            "id": [(f"{name}_{col}" if col != "value" else name) for col in cols],
            "timestamp": [df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").tolist()]*len(cols),
            "target": [df[col].astype(float).tolist() for col in cols],  # or .astype(str) if needed
        }
    )


def shift_df_into_datetime_bounds(df: pd.DataFrame, time_col: str) -> tuple[pd.DataFrame, float, pd.Timestamp]:
    t_min, t_max = df[time_col].min().to_pydatetime(), df[time_col].max().to_pydatetime()

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
        delta_time = pd.to_timedelta(seconds.total_seconds()*0.5-0.000001, unit='s', errors="coerce")
        # We add twice half of the total we aim to add (-1ns)
        d = (t_min_allowed + delta_time) + delta_time
        return d

    df[time_col] = df[time_col].apply(scale_ts)
    df = df.dropna(subset=[time_col])

    t_min, t_max = df[time_col].min().to_pydatetime(), df[time_col].max().to_pydatetime()
    assert t_min >= t_min_allowed and t_max <= t_max_allowed

    return df, factor, t_min




def process_record(rec, min_ts_length):
    if not rec["value"]:
        return {"issue": ("EmptyValueColumn", rec["name"])}

    try:
        df = pd.DataFrame(
            {"datetime": eval(rec["date"]),
             **{f"value_{i}": v for i, v in enumerate(rec["value"])}}
        )

        df = df[~df.astype(str).apply(
            lambda r: r.str.contains("0000-01-01 00:00:00", na=False)).any(axis=1)]
        df = to_datetime(df, "datetime").sort_values("datetime")

        if df["datetime"].isna().any():
            return {"issue": ("MalformedTimeColumn", rec["name"])}

        df = df.dropna(subset=["datetime"])
        # TODO Remove df, _, _ = shift_df_into_datetime_bounds(df, "datetime")
        # Drop Duplicates (Here UCI loses MANY values - UCI contains many event based DS several events can happen at the same time)
        df = df.groupby("datetime", as_index=False).median(numeric_only=True)
        df = equidise(df, min_ts_length, time_column="datetime")

        if len(df) > min_ts_length:
            val_cols = df.columns.difference(["datetime"])
            datasets = make_univar(
                rec["name"], 
                df, 
                # Only the columns of a certain length (without counting nans) that also aren't constant
                cols = [c for c in val_cols if (np.count_nonzero(~np.isnan(df[c])) >= min_ts_length) and (np.nanvar(df[c]) > 0)] 
                )
            return {"equi": len(datasets), "datasets": datasets}
        else:
            return {"nonequi": 1}

    except Exception as e:
        return {"issue": ("ProcessingError", f"{rec['name']}: {e}")}


@app.command()
def preprocessing(min_ts_length: int = 64, RAW_DIR: str = "./data/data_sets_raw/Time_Corpus") -> None:
    assert min_ts_length > 0, "min_ts_length must be a positive integer"
    PROCESSED_DIR = Path(RAW_DIR+"_Processed")
    RAW_DIR = Path(RAW_DIR)


    stats = {"equi": 0, "nonequi": 0}
    issues: Dict[str, List[str]] = {"EmptyValueColumn": [], "MalformedTimeColumn": [], "ProcessingError": []}
    processed: List[ds.Dataset] = []
    
    src = ds.load_from_disk(RAW_DIR)["train"]
    total = len(src)

    worker_count = 8
    batch_size = 4 * worker_count
    # Execute in Parallel CPU Count // 2 because of large Memory Load
    for i in tqdma(range((len(src)+batch_size-1)//batch_size)):
        indices = list(range(i*batch_size, min(len(src), (i+1)*batch_size)))
        batch = src.select(indices)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(process_record, rec, min_ts_length) for rec in batch]
            for future in tqdm(as_completed(futures), total=len(futures), desc="processing"):
                result = future.result()
                if "issue" in result:   
                    issues[result["issue"][0]].append(result["issue"][1])
                elif "equi" in result:
                    stats["equi"] += result["equi"]
                    processed.append(result["datasets"])
                elif "nonequi" in result:
                    stats["nonequi"] += result["nonequi"]

    ds.DatasetDict(train=ds.concatenate_datasets(processed)).save_to_disk(PROCESSED_DIR)

    print(stats)
    print(stats["equi"] / total)
    print(issues)


if __name__ == "__main__":
    app()


