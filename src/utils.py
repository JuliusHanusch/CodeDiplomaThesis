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
from transformers import AutoModel
from torch import nn
from warnings import warn
from typing import Set
from normality import normalize
from copy import deepcopy

DEBUG = False

class ModelTooBig(Exception):
    """_summary_
    Model Surpases self-defined size limits
    """
    pass


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


def get_expected_model_size(model_id: str):
    """
    Loads model from HF and returns its number of non-embedding HP
    Most of the code Below is just caching to not download it over and over again
    """
    cache_path = Path("./cache/model_sizes.csv") # TODO Convert into decorator
    if cache_path.exists():
        cached_scores = pd.read_csv(cache_path)
        relevant_scores = cached_scores[cached_scores["model_id"] == model_id]
        if len(relevant_scores) > 0:
            return relevant_scores["parameters"].iloc[0]

    model = AutoModel.from_pretrained(model_id)
    model_size = get_model_size(model=model)

    results = pd.DataFrame([{"model_id": model_id, "parameters": model_size}])
    if cache_path.exists():
        cached_scores = pd.read_csv(cache_path)
        results = pd.concat([cached_scores, results], ignore_index=True)
    results.to_csv(cache_path, index=False)
    return model_size


def get_model_size(model: AutoModel):
    """Returns number of Non Embedding Parameters"""
    total = sum(p.numel() for p in model.parameters())
    embedding = 0

    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
            embedding += sum(p.numel() for p in module.parameters())
    
    return total - embedding


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


def sample_least_overlapping_subdfs(df: pd.DataFrame, window_len: int, n: int = 20) -> list[pd.DataFrame]:
    L = len(df)
    window_len = abs(window_len)
    max_start = L - window_len
    if n > max_start + 1:
        raise ValueError("Too many sub-DFs for given length and offset.")

    step = max((max_start) // (n - 1), 1)
    starts = [min(i * step, max_start) for i in range(n)]
    return [df.iloc[s : s + window_len] for s in starts]


def subdfs_to_rows(df: pd.DataFrame, offset: int, prediction_length: int, n: int = 20) -> pd.DataFrame:
    window_len = abs(offset) + abs(prediction_length)
    subdfs = sample_least_overlapping_subdfs(df, window_len, n)
    
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
    offset = config["offset"]
    prediction_length = config["prediction_length"]
    num_rolls = config["num_rolls"]
    name = config["name"]

    if "hf_repo" in config:
        hf_repo = config["hf_repo"]
        trust_remote_code = True if hf_repo == "autogluon/chronos_datasets_extra" else False

        ds = datasets.load_dataset(
            hf_repo, name, split="train", trust_remote_code=trust_remote_code
        )
    else:
        if "disk_path" in config:
            disk_path = Path(config["disk_path"])
            cache_adr = Path("./cache") / (disk_path.stem + ".parquet")

            # Load -> Explode Table -> Cache
            if not cache_adr.exists():
                data = datasets.load_from_disk(disk_path)
                data = data["test"]
                # Convert into pd format + Drop Unnecessary Features
                exploded_data = [
                    {"target": sublist}
                    for row in data["value"]
                    for sublist in row
                ]
                df = pd.DataFrame(exploded_data)
                df.to_parquet(cache_adr)
            else:
                df = pd.read_parquet(cache_adr)

            # Select TS
            ts_id = target # target should be just a number
            try:
                ts = pd.DataFrame({"target": df.iloc[ts_id]["target"]})
            except IndexError as e:
                # Return None if Target outside Range available data
                warn(f"TS {ts_id} couldnt be loaded from {disk_path} maybe some ds went missing during download")
                return None
            
            ts.reset_index(inplace=True)
            
        else:
            try:
                df = load_via_uci_api(dataset_id=config["id"])
            except DatasetNotFoundError:
                df = load_from_link(
                        url = config["link"],
                        filename = config.get("filename"),
                        dataset_name = name
                    )

            data = df[[target]].copy()
            data = data.rename({target: "target"}, axis="columns")
            data.reset_index(inplace=True)

            # Clean all values (remove $ and , from strings)
            def clean_currency(val):
                if isinstance(val, str):
                    return pd.to_numeric(val.replace("$", "").replace(",", ""), errors="coerce")
                return val

            ts = data.applymap(clean_currency)

        # Back to HF
        data = subdfs_to_rows(ts, offset=offset, prediction_length=prediction_length, n=num_rolls)
        # if autogluon_format:
        #     # Split Data Manually
        #     data = subdfs_to_rows(ts, offset=offset, prediction_length=prediction_length, n=num_rolls)
        # else:
        #     # let gluonts split it later
        #     data = pd.DataFrame([
        #         {
        #             'timestamp': ts["timestamp"] if "timestamp" in ts.columns else  [str(timestamp) for timestamp in pd.date_range(start='1990-12-01', freq="min", periods=len(ts.index))],
        #             'target': ts["target"].tolist()
        #         }
        #     ])

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
        if DEBUG:
            item_ids = set(df['item_id'])
            print(f"n_ids: {len(item_ids)}", flush=True)
            lengths = [len(validation_data.loc[iid]) for iid in item_ids]
    else:
        gts_dataset = to_gluonts_univariate(ds)

        # "Split" dataset for evaluation (or pseudo-split because we already split manually for the AG version) 
        # TODO Replace Gluonts splitting it behaves strangely (large negative numbers are taken once modulo for reasons)
        _, test_template = split(gts_dataset, offset=abs(offset)) 
        validation_data = test_template.generate_instances(prediction_length=prediction_length, windows=1)
        if DEBUG:
            v2 = deepcopy(validation_data)
            lengths = [len(sample["target"]) for sample in v2.input]

    # Print Dimensions of Val Set
    if DEBUG:
        v2 = deepcopy(validation_data)
        n_samples = sum(1 for _ in v2)
        print(f"Offset {offset}\nPredictionLength {prediction_length}\nNumRolls {num_rolls}\nname {name}\nN_samples {n_samples}\nLengths {lengths}", flush=True)
    return validation_data


# Utils #

def char_bigrams(s: str) -> Set[str]:
    """Return the set of character bigrams for string s."""
    return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) > 1 else set()

def bigram_similarity(s1: str, s2: str) -> float:
    """
    Compute Jaccard similarity between two strings based on character bigrams.
    Returns 1.0 if both strings yield no bigrams.
    """
    b1, b2 = char_bigrams(f" {s1} "), char_bigrams(f" {s2} ")
    if not b1 and not b2:
        return 1.0
    inter = b1 & b2
    union = len(b1) + len(b2)
    return len(inter) * 2 / union

@cache
def my_normalize(text: str|list[str]):
    if isinstance(text, str):
        return normalize(text, latinize=True, ascii=True)
    elif isinstance(text, (list, tuple)):
        return [my_normalize(word) for word in text]
    else:
        raise Exception(f"Text has unsupported dtype {type(text)}")

def is_in_collection(orig: str, collection: list[str], thr=0.8):
    """Returns True if orig is similar enough to any of the words in the collection"""
    orig = my_normalize(orig)
    collection = my_normalize(tuple(collection)) # to tuple for caching
    for word in collection:
        score = bigram_similarity(orig, word)
        if score >= thr:
            print(f"{orig} matches {word}!")
            return True
    return False


def group_similar_words(words: list[str]) -> list[list[str]]:
    groups = []
    for word in words:
        placed = False
        for group in groups:
            if is_in_collection(word, group, thr=0.85):
                group.append(word)
                placed = True
                break
        if not placed:
            groups.append([word])
    return groups

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