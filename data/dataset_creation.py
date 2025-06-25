import datasets as ds
import numpy as np
import pandas as pd
import random
import logging

from pathlib import Path
from typing import List, Union
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from tqdm import tqdm

from gluonts.dataset.arrow import ArrowWriter
from ts_mixup import ts_mixup
from concurrent.futures import ThreadPoolExecutor
from typer import Typer
from typer_config import use_yaml_config
import sys
from abc import ABC, abstractmethod
from functools import partial

# --- Config ---
logging.basicConfig(level=logging.INFO)
CHUNK_SIZE = 100

app = Typer()


class DatasetAdapter(ABC):
    """
    Idea: We get DS in different shape (Every Corpus has its own unique shape)
    Time Corpus: Fits entire DS into rows
    Chronos: Fits Datasets into splits and TS into rows
    Lotsa
    Performing in depth preprocessing to bring all into the same shape somehow is too difficult
    Instead we add to each DS such an adapter with the fundamental functions we require
    Allows to translate rules according to the shape of each DS 
    """
    # Init
    def __init__(self, dataset):
        self.dataset = dataset

    # len
    @abstractmethod
    def __len__(self):
        pass
    
    # get random snippet of length n
    @abstractmethod
    def get_random_snippet(self, length: int):
        pass


class RowDataSet(DatasetAdapter):
    """
    Dataset for corpora where each row is an DS
    """
    def __init__(self, dataset, target_column: str = "value", deduplication: bool = True):
        super().__init__(dataset)
        self.target_column = target_column
        # Remember used startpoints to keep Birthday Problem from breaking the infinite Data Regiment
        self.unused_startpoints = np.array([])
        self.deduplication = deduplication

    def __len__(self):
        return len(self.dataset[self.target_column]) 

    def get_random_snippet(self, length: int, exp_count: int = 1, **kwargs): # TODO Cut out length instead of just the start point
        if self.deduplication:
            unused_startpoint_count = len(self.unused_startpoints)
            if unused_startpoint_count == 0:
                self.unused_startpoints = np.arange(len(self) - length + 1)
                unused_startpoint_count = len(self.unused_startpoints)
            start_point_id = np.random.randint(unused_startpoint_count)
            start_point = self.unused_startpoints[start_point_id]
            # Remove as many surrounding starpoints as possible to minimize overlap (number to remove calc via expected number samples drawn from TS)
            vals_to_spare = min(((len(self.dataset) + 1 - length) // exp_count), length) // 2
            startpoints_to_delete = ~np.isin(self.unused_startpoints, np.arange(start_point-vals_to_spare, start_point+vals_to_spare+1))
            self.unused_startpoints = self.unused_startpoints[startpoints_to_delete]
        else:
            start_point = np.random.randint(len(self) - length + 1)
        return self.dataset[self.target_column][start_point:start_point + length]


class SplitDataSet(DatasetAdapter):
    """
    Dataset for corpora where each split (or subfolder) is a DS with each row being a TS of this
    """
    def __init__(self, dataset, target_column: str = "target", deduplication: bool = True):
        self.target_column = target_column

        if isinstance(dataset[0][self.target_column][0], list):
            # flatten Multivariate to multiple univariates
            dataset = dataset.map(
                partial(explode_batch, target_column=target_column),
                batched=True,
                remove_columns=[c for c in dataset.column_names if c not in ("item_id","start","freq")],
            )

        super().__init__(dataset)
        # Remember used ts to keep Birthday Problem from breaking the infinite Data Regiment
        self.unused_ts = np.array([])
        self.deduplication = deduplication

        assert not isinstance(dataset[0][self.target_column][0], list), "Multivariate DS aren't supported yet"

    def __len__(self):
        return sum([len(ts[self.target_column]) for ts in self.dataset])

    def get_random_snippet(self, length: int, **kwargs):
        if self.deduplication:
            unused_ts_count = len(self.unused_ts)
            if unused_ts_count == 0:
                self.unused_ts = np.arange(len(self.dataset))
                unused_ts_count = len(self.unused_ts)
            ts_id_id = np.random.randint(unused_ts_count)
            ts_id = self.unused_ts[ts_id_id]
            self.unused_ts = np.delete(self.unused_ts, ts_id_id)
        else:
            ts_id = np.random.randint(len(self.dataset) - length + 1)

        ts = self.dataset[int(ts_id)][self.target_column]
        start_point = np.random.randint(len(ts) - length + 1)
        return ts[start_point:start_point + length]


# --- Utility Functions ---
def convert_to_arrow(path: Union[str, Path], time_series: List[np.ndarray], compression: str = "lz4"):
    start = np.datetime64("2000-01-01 00:00", "s")
    dataset = [{"start": start, "target": ts} for ts in time_series]
    ArrowWriter(compression=compression).write_to_file(dataset, path=path)


def load_folder(folder: Path, min_length = 128, deduplication = True):
    corpus = []
    try:
        data = ds.load_from_disk(folder)
            # Filter out the too short ones
        data = data.map(trim_nans) # remove trivials (only long enough because of NaNs) TODO Maybe ok at beginning??
        data = data.filter(lambda x: len(x["value"])>=min_length)
        data = data["train"] if "train" in data else data  # unnest 
        for dataset in data:
            corpus.append(RowDataSet(dataset, deduplication=deduplication))
    except:
        subfolders = [f for f in folder.iterdir() if not f.name.startswith('.') and f.is_dir()] # filter out hidden folders like .cache and files like .md
        for subfolder in tqdm(subfolders, desc="Loading datasets"):
            data = ds.load_from_disk(subfolder)
            corpus.append(SplitDataSet(data, deduplication=deduplication))
    return corpus


def load_datasets(data_path: Path, corpora: Union[List[str], str], min_length=128, deduplication = True) -> dict:
    corpus = []
    for folder in tqdm(list(data_path.iterdir()), desc="Loading datasets"):
        if ("all" in corpora and folder.name != "Time_Corpus") or (isinstance(corpora, list) and folder.name in corpora):
            corpus += load_folder(folder, min_length=min_length, deduplication=deduplication)
        # elif isinstance(corpora, list) and folder.name in corpora:
        #     corpus += load_folder(folder, min_length=min_length)
    if not corpus:
        raise ValueError("No matching datasets found.")
    return corpus

def trim_nans(dataset):
    arr = np.array(dataset["value"])
    mask = ~np.isnan(arr)
    if not mask.any():
        return []
    start_point = mask.argmax()
    endpoint = len(mask) - mask[::-1].argmax()
    dataset["value"] = dataset["value"][start_point:endpoint]
    dataset["datetime"] = dataset["datetime"][start_point:endpoint]
    return dataset


def get_top_contributors(values, thr=0.3):
    arr = np.array(values)
    sorted_indices = np.argsort(arr)[::-1]  # Descending
    sorted_vals = arr[sorted_indices]

    cumsum = np.cumsum(sorted_vals)
    n = np.searchsorted(cumsum, thr, side='left') + 1

    return sorted(sorted_indices[:n].tolist())

def explode_batch(batch, target_column):
    out = {"item_id": [], "start": [], "freq": [], "target": []}
    for iid, st, fq, targets in zip(batch["item_id"], batch["start"], batch["freq"], batch[target_column]):
        for tgt in targets:
            out["item_id"].append(iid)
            out["start"].append(st)
            out["freq"].append(fq)
            out["target"].append(tgt)
    return out


@app.command()
@use_yaml_config(param_name="config")
def create_dataset(
    corpora: List[str] = ['all'],
    k: int = 1,
    length: int = 128,
    samples: int = 1000,
    alpha: float = 1.5,
    #min_probability: float = 0.000025, 
    small_ts_share: float = 0.3,
    deduplication: bool = True,
    workers: int = 4,
    data_path: str = "./data/data_sets_raw",
    output_dir: str = "data/train"
):
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    logging.info(f"Loading datasets from {data_path.resolve()}")
    corpus = load_datasets(data_path, corpora, min_length=length, deduplication=deduplication)

    ds_lengths = [len(record) for record in corpus]
    corpus_length = sum(ds_lengths)
    ds_count = len(ds_lengths)
    assert np.all(np.array(ds_lengths) >= length)
    min_probability = small_ts_share / ds_count # 20% of the corpus are reserved for the smallest DS

    # calculate probs
    assert ds_count * min_probability < 1, "Minimum probability too high for the number of datasets, can't be satisfied." # If == 1 would be equal weights
    # Capped to min probability or the probability that the expected number of samples from this DS covers the entire ds (Reason: No snippet twice)
    ds_probability = np.array([max(ds_length / corpus_length, min((ds_length+1-length)/samples, min_probability)) for ds_length in ds_lengths])
    total_raise = sum(ds_probability) - 1 # raise through capping
    big_ones = get_top_contributors(ds_probability, thr=0.3)
    ds_probability[big_ones] -= total_raise / len(big_ones)  # the top 1% subsidies the smallest ones (hehehe)
    assert 1 - min_probability < sum(ds_probability) < 1 + min_probability, "Probabilities do not sum to near 1, something went wrong."
    assert np.all((0 < ds_probability) & (ds_probability < 1))
    # assert np.all(np.array(ds_probability) >= min_probability), "Some probabilities are below the minimum probability threshold."

    logging.info("Creating snippets")
    def create_augmented_snippet(_):
        augmented_chunk = []
        while len(augmented_chunk) < CHUNK_SIZE:
            datasets_ids = np.random.choice(ds_count, size=k, replace=False, p=ds_probability)
            snippets = []
            for ds_id in datasets_ids:
                dataset = corpus[int(ds_id)]
                exp_count = samples * ds_probability[ds_id]
                snippets.append(dataset.get_random_snippet(length=length, exp_count=exp_count))

            final_sample = ts_mixup(snippets, alpha=alpha)
            # Quality insurance
            if np.isnan(np.array(final_sample)).sum() / length < 0.25:  # Less than 25% NaNs
                augmented_chunk.append(final_sample)
        return augmented_chunk
    
    chunk_count = (samples+(CHUNK_SIZE-1))//CHUNK_SIZE
    with ThreadPoolExecutor(workers) as executor:
        corpus_augmented = list(tqdm(executor.map(create_augmented_snippet, range(chunk_count)),
                                    total=chunk_count,
                                    desc="Creating augmented snippets"))

    # Flatten the list of lists
    corpus_augmented = [snippet for chunk in corpus_augmented for snippet in chunk]
    # Convert to final format -> save to disk
    filename = f"tsm_{'_'.join(corpora) if isinstance(corpora, list) else corpora}_k-{k}_len-{length}_a-{str(alpha).replace('.', '')}.arrow"
    output_dir.mkdir(parents=True, exist_ok=True)
    convert_to_arrow(output_dir / filename, corpus_augmented[:samples])


if __name__ == "__main__":
    app()
