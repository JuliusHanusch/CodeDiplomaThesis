import datetime
from pathlib import Path
import json
from ucimlrepo import fetch_ucirepo 
from pathlib import Path 
from gluonts.dataset.split import split
import datasets
from datasets import Dataset
import pandas as pd
from datasets import Features, Value, Sequence, SplitInfo
import zipfile
import tempfile
import subprocess
from ucimlrepo import DatasetNotFoundError
from functools import cache
from autogluon.timeseries import TimeSeriesDataFrame
import numpy as np


def make_dict_storable(advanced_dictionary: dict)->dict:
    """
    Takes in a dict with advanced values like Datatime, dicts, lists, etc. 
    and converts it into a dict that can be stored in an sqlite database meaning strings, numbers, etc.

    Args:
        advanced_dictionary (dict): dict with potentionally complex datatypes

    Returns:
        dict: A dictionary with only simple datatypes
    """
    simple_dict = {}
    for key, value in advanced_dictionary.items():
        if isinstance(value, (int, float, str)):
            pass
        elif isinstance(value, bool):
            value = 1 if value else 0
        elif isinstance(value, (bytes, Path, list)) or value is None:
            value = str(value)
        elif isinstance(value, (datetime.datetime, datetime.date)):
            value = value.strftime("%d-%m-%Y %H:%M:%S")
        elif isinstance(value, dict):
            value = json.dumps(value)
        elif isinstance(value, (np.bool_)):
            value = 1 if value else 0
        elif isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
            value = int(value)
        elif isinstance(value, (np.float64, np.float32, np.float16)):
            value = float(value)
        else:
            raise NotImplementedError(f"dtype {type(value)} is currently not storable, but can likely be easily added in make_dict_storable()")
        simple_dict[key] = value

    return simple_dict


def results_to_metrics(results_dict: dict):
    """
    Takes in a dict containing the results of the pretraining with two Dataframes one for in-domain and one for zero shot evaluation, extracts the metrics,
    returns all metrics as vaiables
    """
    metrics = ["MASE", "WQL", "MAE", "NRMSE[mean]"]
    
    # Ensure consistent keys
    in_domain_df = results_dict.get("in-domain")
    zero_shot_df = results_dict.get("zero-shot")

    # Compute means for each metric
    in_domain_means = {f"in_domain_{metric.lower()}": in_domain_df[metric].mean() for metric in metrics}
    zero_shot_means = {f"zero_shot_{metric.lower()}": zero_shot_df[metric].mean() for metric in metrics}

    # Combine and return as separate variables
    return (
        in_domain_means["in_domain_mase"],
        in_domain_means["in_domain_wql"],
        in_domain_means["in_domain_mae"],
        in_domain_means["in_domain_nrmse"],
        zero_shot_means["zero_shot_mase"],
        zero_shot_means["zero_shot_wql"],
        zero_shot_means["zero_shot_mae"],
        zero_shot_means["zero_shot_nrmse"]
    )


def to_gluonts_univariate(hf_dataset: datasets.Dataset):
    series_fields = [
        col
        for col in hf_dataset.features
        if isinstance(hf_dataset.features[col], datasets.Sequence)
    ]
    series_fields.remove("timestamp")
    dataset_length = hf_dataset.info.splits["train"].num_examples * len(series_fields)
    dataset_freq = pd.infer_freq(hf_dataset[0]["timestamp"])
    dataset_freq = offset_alias_to_period_alias.get(dataset_freq, dataset_freq)

    gts_dataset = []
    for hf_entry in hf_dataset:
        for field in series_fields:
            gts_dataset.append(
                {
                    "start": pd.Period(
                        hf_entry["timestamp"][0],
                        freq=dataset_freq,
                    ),
                    "target": hf_entry[field],
                }
            )
    assert len(gts_dataset) == dataset_length

    return gts_dataset


def sample_least_overlapping_subdfs(df: pd.DataFrame, offset: int, n: int = 20) -> list[pd.DataFrame]:
    L = len(df)
    offset = abs(offset)
    max_start = L - offset
    if n > max_start + 1:
        raise ValueError("Too many sub-DFs for given length and offset.")

    step = max((max_start) // (n - 1), 1)
    starts = [min(i * step, max_start) for i in range(n)]
    return [df.iloc[s : s + offset] for s in starts]

def subdfs_to_rows(df: pd.DataFrame, offset: int, n: int = 20) -> pd.DataFrame:
    subdfs = sample_least_overlapping_subdfs(df, offset, n)
    
    result = pd.DataFrame([
        {
            #'indices': list(subdf.index),
            'timestamp': subdf["timestamp"] if "timestamp" in df.columns else  [str(ts) for ts in pd.date_range(start='1990-12-01', freq="D", periods=len(subdf.index))],
            'target': subdf["target"].tolist()
        }
        for subdf in subdfs
    ])
    
    return result


def load_via_uci_api(dataset_id):
    ds = fetch_ucirepo(id=dataset_id)
    print("Dataset can be fetched directly")
    print(dataset_id)

    return pd.concat([ds.data.features, ds.data.targets], axis=1)


def load_from_link(url, filename, dataset_name):
    extract_dir = Path(f"./data/tmp/{filename}.zip")
    target_path = extract_dir / Path(filename)
    if filename is None:
        raise ValueError(f"'filename' must be specified in config for dataset '{dataset_name}'")

    if not target_path.exists():
        local_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        print("Downloading zip...")
        subprocess.run(["wget", "-q", "-O", local_zip.name, url], check=True)
        print("Downloaded zip to", local_zip.name)

        print("Extracting main zip...")
        with zipfile.ZipFile(local_zip.name, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Main extraction done.")

        def extract_nested_zips(dir_path):
            nested_zips = list(dir_path.glob("**/*.zip"))
            for nested_zip in nested_zips:
                with zipfile.ZipFile(nested_zip, "r") as zip_ref:
                    zip_ref.extractall(nested_zip.parent)
                nested_zip.unlink()
            if list(dir_path.glob("**/*.zip")):
                extract_nested_zips(dir_path)

        extract_nested_zips(extract_dir)

        print(f"Looking for file: {target_path}")
        if not target_path.exists():
            print("Extracted files:")
            for path in extract_dir.rglob("*"):
                print(f" - {path.relative_to(extract_dir)}")
            raise FileNotFoundError(f"'{filename}' not found in extracted contents of dataset '{dataset_name}'")

    if target_path.suffix.lower() in [".csv", ".txt"]:
        print("Reading CSV...")
        if target_path.suffix.lower() == ".txt":
            df = pd.read_csv(target_path, on_bad_lines='skip', sep=";")  # pandas >= 1.3
        else:
            df = pd.read_csv(target_path, on_bad_lines='skip')  # pandas >= 1.3
        print("Finished reading, shape:", df.shape)
    else:
        raise ValueError(f"Unsupported file type: {target_path.suffix}")
    return df


def load_val_data(
    config: dict,
    target: str = "target",
    autogluon_format: bool = False,
):
    print(f"\n=== Processing dataset: {config.get('name', config.get('id', 'unknown'))} ===")
    print(f"\nStarting processing: {config['name']}")

    if "hf_repo" in config:
        offset = config["offset"]
        prediction_length = config["prediction_length"]
        num_rolls = config["num_rolls"]
        hf_repo = config["hf_repo"]
        name = config["name"]
        trust_remote_code = True if hf_repo == "autogluon/chronos_datasets_extra" else False


        ds = datasets.load_dataset(
            hf_repo, name, split="train", trust_remote_code=trust_remote_code
        )


    else:
        try:
            df = load_via_uci_api(dataset_id=config["id"])
        except DatasetNotFoundError:
            df = load_from_link(
                    url = config["link"],
                    filename = config.get("filename"),
                    dataset_name = config["name"]
                )

        offset = config["offset"]
        prediction_length = config["prediction_length"]
        num_rolls = config["num_rolls"]

        data = df[[target]].copy()
        data = data.rename({target: "target"}, axis="columns")
        data.reset_index(inplace=True)

        # Clean all values (remove $ and , from strings)
        def clean_currency(val):
            if isinstance(val, str):
                return pd.to_numeric(val.replace("$", "").replace(",", ""), errors="coerce")
            return val

        data = data.applymap(clean_currency)
        data = subdfs_to_rows(data, offset=offset, n=num_rolls)

        features = Features({
            #'indices': Sequence(Value('int64')),
            'timestamp': Sequence(Value('string')),  # or 'timestamp[s]' if ISO format
            'target': Sequence(Value('float64'))  # 2D array: rows of values per sub-df
        })

        ds = Dataset.from_pandas(data, features=features)
        ds._info.splits = {
            "train": SplitInfo(name="train", num_examples=len(ds)),
            "test": SplitInfo(name="test", num_examples=0)
        }

    ds.set_format("numpy")
    if autogluon_format:
        df = ds.to_pandas()
        df['item_id'] = df.index
        df = df.explode(["timestamp", "target"]).reset_index(drop=True)
        validation_data = TimeSeriesDataFrame(
            data=df,
            id_column="item_id",
            timestamp_column="timestamp",
        )
    else:
        gts_dataset = to_gluonts_univariate(ds)

        # Split dataset for evaluation
        _, test_template = split(gts_dataset, offset=offset)
        validation_data = test_template.generate_instances(prediction_length, windows=num_rolls)
    return validation_data


# Taken from pandas._libs.tslibs.dtypes.OFFSET_TO_PERIOD_FREQSTR
offset_alias_to_period_alias = {
    "WEEKDAY": "D",
    "EOM": "M",
    "BME": "M",
    "SME": "M",
    "BQS": "Q",
    "QS": "Q",
    "BQE": "Q",
    "BQE-DEC": "Q",
    "BQE-JAN": "Q",
    "BQE-FEB": "Q",
    "BQE-MAR": "Q",
    "BQE-APR": "Q",
    "BQE-MAY": "Q",
    "BQE-JUN": "Q",
    "BQE-JUL": "Q",
    "BQE-AUG": "Q",
    "BQE-SEP": "Q",
    "BQE-OCT": "Q",
    "BQE-NOV": "Q",
    "MS": "M",
    "D": "D",
    "B": "B",
    "min": "min",
    "s": "s",
    "ms": "ms",
    "us": "us",
    "ns": "ns",
    "h": "h",
    "QE": "Q",
    "QE-DEC": "Q-DEC",
    "QE-JAN": "Q-JAN",
    "QE-FEB": "Q-FEB",
    "QE-MAR": "Q-MAR",
    "QE-APR": "Q-APR",
    "QE-MAY": "Q-MAY",
    "QE-JUN": "Q-JUN",
    "QE-JUL": "Q-JUL",
    "QE-AUG": "Q-AUG",
    "QE-SEP": "Q-SEP",
    "QE-OCT": "Q-OCT",
    "QE-NOV": "Q-NOV",
    "YE": "Y",
    "YE-DEC": "Y-DEC",
    "YE-JAN": "Y-JAN",
    "YE-FEB": "Y-FEB",
    "YE-MAR": "Y-MAR",
    "YE-APR": "Y-APR",
    "YE-MAY": "Y-MAY",
    "YE-JUN": "Y-JUN",
    "YE-JUL": "Y-JUL",
    "YE-AUG": "Y-AUG",
    "YE-SEP": "Y-SEP",
    "YE-OCT": "Y-OCT",
    "YE-NOV": "Y-NOV",
    "W": "W",
    "ME": "M",
    "Y": "Y",
    "BYE": "Y",
    "BYE-DEC": "Y",
    "BYE-JAN": "Y",
    "BYE-FEB": "Y",
    "BYE-MAR": "Y",
    "BYE-APR": "Y",
    "BYE-MAY": "Y",
    "BYE-JUN": "Y",
    "BYE-JUL": "Y",
    "BYE-AUG": "Y",
    "BYE-SEP": "Y",
    "BYE-OCT": "Y",
    "BYE-NOV": "Y",
    "YS": "Y",
    "BYS": "Y",
    "QS-JAN": "Q",
    "QS-FEB": "Q",
    "QS-MAR": "Q",
    "QS-APR": "Q",
    "QS-MAY": "Q",
    "QS-JUN": "Q",
    "QS-JUL": "Q",
    "QS-AUG": "Q",
    "QS-SEP": "Q",
    "QS-OCT": "Q",
    "QS-NOV": "Q",
    "QS-DEC": "Q",
    "BQS-JAN": "Q",
    "BQS-FEB": "Q",
    "BQS-MAR": "Q",
    "BQS-APR": "Q",
    "BQS-MAY": "Q",
    "BQS-JUN": "Q",
    "BQS-JUL": "Q",
    "BQS-AUG": "Q",
    "BQS-SEP": "Q",
    "BQS-OCT": "Q",
    "BQS-NOV": "Q",
    "BQS-DEC": "Q",
    "YS-JAN": "Y",
    "YS-FEB": "Y",
    "YS-MAR": "Y",
    "YS-APR": "Y",
    "YS-MAY": "Y",
    "YS-JUN": "Y",
    "YS-JUL": "Y",
    "YS-AUG": "Y",
    "YS-SEP": "Y",
    "YS-OCT": "Y",
    "YS-NOV": "Y",
    "YS-DEC": "Y",
    "BYS-JAN": "Y",
    "BYS-FEB": "Y",
    "BYS-MAR": "Y",
    "BYS-APR": "Y",
    "BYS-MAY": "Y",
    "BYS-JUN": "Y",
    "BYS-JUL": "Y",
    "BYS-AUG": "Y",
    "BYS-SEP": "Y",
    "BYS-OCT": "Y",
    "BYS-NOV": "Y",
    "BYS-DEC": "Y",
    "Y-JAN": "Y-JAN",
    "Y-FEB": "Y-FEB",
    "Y-MAR": "Y-MAR",
    "Y-APR": "Y-APR",
    "Y-MAY": "Y-MAY",
    "Y-JUN": "Y-JUN",
    "Y-JUL": "Y-JUL",
    "Y-AUG": "Y-AUG",
    "Y-SEP": "Y-SEP",
    "Y-OCT": "Y-OCT",
    "Y-NOV": "Y-NOV",
    "Y-DEC": "Y-DEC",
    "Q-JAN": "Q-JAN",
    "Q-FEB": "Q-FEB",
    "Q-MAR": "Q-MAR",
    "Q-APR": "Q-APR",
    "Q-MAY": "Q-MAY",
    "Q-JUN": "Q-JUN",
    "Q-JUL": "Q-JUL",
    "Q-AUG": "Q-AUG",
    "Q-SEP": "Q-SEP",
    "Q-OCT": "Q-OCT",
    "Q-NOV": "Q-NOV",
    "Q-DEC": "Q-DEC",
    "W-MON": "W-MON",
    "W-TUE": "W-TUE",
    "W-WED": "W-WED",
    "W-THU": "W-THU",
    "W-FRI": "W-FRI",
    "W-SAT": "W-SAT",
    "W-SUN": "W-SUN",
}