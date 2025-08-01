# Add parent directory to share global utils with different experiments
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import sys  
import os  
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.resolve()))  
from src.utils import group_similar_words

import datasets as ds
import numpy as np
import pandas as pd
import random
import logging

from pathlib import Path
from typing import List, Union
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from gluonts.dataset.arrow import ArrowWriter
from ts_mixup import ts_mixup
from concurrent.futures import ThreadPoolExecutor
from typer import Typer
from typer_config import use_yaml_config
import sys
from abc import ABC, abstractmethod
from functools import partial
from copy import deepcopy
from src.db import insertTable, hash_dict
from time import sleep
from collections import Counter
import psutil

CACHE = Path("./cache/data")
CACHE.mkdir(parents=True, exist_ok=True)

# --- Config ---
logging.basicConfig(level=logging.INFO)
print(f"PYTHON_PID={os.getpid()}", flush=True)
CHUNK_SIZE = 100
CPU_COUNT = int(os.environ.get("SLURM_CPUS_PER_TASK", psutil.cpu_count(logical=True))) # get from slurm else standaard
WORKERS = CPU_COUNT 


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
    def __init__(self, dataset, target_column):
        self.target = dataset.shuffle(seed=None).take(min(len(dataset), int(1))) # TODO for giant ds take only 10k randomly selected TS else OOM 
        self.target = self.target[target_column] # Drop everything except target column to speed up look ups later

    # len
    @abstractmethod
    def __len__(self):
        pass
    
    # get random snippet of length n
    @abstractmethod
    def get_random_snippet(self, length: int):
        pass

# ! Obsolete
class RowDataSet(DatasetAdapter):
    """
    Dataset for corpora where each row is an DS
    """
    def __init__(self, dataset, target_column: str = "target", deduplication: bool = True):
        super().__init__(dataset, target_column)
        self.target_column = target_column
        # Remember used startpoints to keep Birthday Problem from breaking the infinite Data Regiment
        self.unused_startpoints = np.array([])
        self.deduplication = deduplication
        self.in_use = False # lock to avoid race conditions


    def __len__(self):
        return len(self.target) 

    def get_random_snippet(self, length: int, exp_count: int = 1, **kwargs): 
        if self.deduplication:
            # Avoid Race Conditions during deduplication
            while self.in_use: 
                sleep(0.01)
            self.in_use = True

            unused_startpoint_count = len(self.unused_startpoints)
            if unused_startpoint_count == 0:
                self.unused_startpoints = np.arange(len(self) - length + 1)
                unused_startpoint_count = len(self.unused_startpoints)
            start_point_id = np.random.randint(unused_startpoint_count)
            start_point = self.unused_startpoints[start_point_id]
            # Remove as many surrounding starpoints as possible to minimize overlap (number to remove calc via expected number samples drawn from TS)
            vals_to_spare = min(((len(self.target) + 1 - length) // exp_count), length) // 2
            startpoints_to_delete = ~np.isin(self.unused_startpoints, np.arange(start_point-vals_to_spare, start_point+vals_to_spare+1))
            self.unused_startpoints = self.unused_startpoints[startpoints_to_delete]

            self.in_use = False
        else:
            start_point = np.random.randint(len(self) - length + 1)
        return self.target[start_point:start_point + length]

def get_cols_with_lists(data: ds.Dataset):
    """Returns the names of all columns that contain lists or lists of lists instead of scalar values"""
    list_cols = []
    for col in data.column_names:
        feature = data.features.get(col)
        if isinstance(feature, ds.Sequence):
            list_cols.append(col)
        elif feature is None and isinstance(data[0][col], (list, tuple)):
            list_cols.append(col)
    return list_cols

class SplitDataSet(DatasetAdapter):
    """
    Dataset for corpora where each split (or subfolder) is a DS with each row being a TS of this
    """
    def __init__(self, dataset_path: Path | ds.Dataset, target_column: str = "target", deduplication: bool = True):
        self.target_column = target_column
        # Remember used ts to keep Birthday Problem from breaking the infinite Data Regiment
        self.unused_ts = np.array([])
        self.deduplication = deduplication
        self.in_use = False # lock to avoid race conditions

        if isinstance(dataset_path, (Path, str)):
            cache_adr = CACHE / dataset_path.name
            if cache_adr.exists():
                dataset = ds.load_from_disk(cache_adr)
            else:
                dataset = ds.load_from_disk(dataset_path)

                if "train" in dataset:
                    dataset = dataset["train"]

                # Identify all columns that could be target columns
                columns_to_exclude = {"timestamp", "id", "item_id", "start", "freq"}
                # Drop all columns with only a single value (else we check them a few million times if they are still single val)
                cols_to_keep = set(get_cols_with_lists(dataset))
                cols_to_keep = cols_to_keep.union(columns_to_exclude)
                dataset = dataset.remove_columns([col for col in dataset.column_names if col not in cols_to_keep])

                columns = set(dataset.column_names)
                columns = list(columns - columns_to_exclude)
                print("Columns:", columns)
                print("Rows:", len(dataset))

                if len(columns) > 1:
                    # Create Multivariate Target Column by Concat all possible targets (gets exploded in the next step)
                    def concat_lists(example):
                        targets = []
                        for col in columns:
                            if isinstance(example[col], list):
                                if isinstance(example[col][0], list): # List of lists
                                    targets += [
                                        example[col][i] 
                                        for i in range(len(example[col]))
                                        if isinstance(example[col][i], list) and len(set(example[col][i])) > 1 # Check that not constant
                                    ]
                                elif len(set(example[col])) > 1: # Check that not constant
                                    targets.append(example[col])
                        example[self.target_column] = targets
                        return example
                    
                    if self.target_column in dataset.column_names: # When overwritting target column we better rename to avoid conflicts
                        dataset = dataset.rename_column(self.target_column, self.target_column + '_old')
                        columns = [(self.target_column + '_old' if col == self.target_column else col) for col in columns] # update target col name
                    dataset = dataset.map(concat_lists) #TODO num_proc
                elif columns[0] != target_column:
                    dataset = dataset.rename_column(columns[0], target_column)
                    
                long_ts = len(dataset[0][self.target_column]) > 15_000
                if isinstance(dataset[0][self.target_column][0], list) or long_ts:

                    # flatten Multivariate to multiple univariates
                    dataset = dataset.map(
                        partial(explode_batch, target_column=target_column),
                        batched=True,
                        batch_size=200,
                        remove_columns=[c for c in dataset.column_names if c not in ("item_id","start","freq")],
                        num_proc=WORKERS,
                    )
                dataset.save_to_disk(cache_adr)

        elif isinstance(dataset_path, ds.Dataset):
            dataset = dataset_path

        super().__init__(dataset, target_column)

        assert not isinstance(self.target[0][0], (list, tuple)), "Multivariate DS aren't supported yet"

    def __len__(self):
        return sum([len(ts) for ts in self.target])

    def get_random_snippet(self, length: int, **kwargs):
        if self.deduplication:
            # Avoid Race Conditions during deduplication
            while self.in_use: 
                print("Waiting for lock to be released...")
                sleep(0.01)
            self.in_use = True

            unused_ts_count = len(self.unused_ts)
            if unused_ts_count == 0:
                self.unused_ts = np.arange(len(self.target))
                unused_ts_count = len(self.unused_ts)
            ts_id_id = np.random.randint(unused_ts_count)
            ts_id = self.unused_ts[ts_id_id]
            self.unused_ts = np.delete(self.unused_ts, ts_id_id)

            self.in_use = False
        else:
            ts_id = np.random.randint(len(self.target))

        ts = self.target[int(ts_id)]
        start_point = np.random.randint(len(ts) - length + 1)
        return ts[start_point:start_point + length]


# --- Utility Functions ---
def convert_to_arrow(path: Union[str, Path], time_series: List[np.ndarray], compression: str = "lz4"):
    start = np.datetime64("2000-01-01 00:00", "s")
    dataset = [{"start": start, "target": ts} for ts in time_series]
    ArrowWriter(compression=compression).write_to_file(dataset, path=path)


def load_folder(folder: Path, min_length = 128, deduplication = True):
    print(f"Loading Folder: {folder}")
    corpus = []
    try:
        data = ds.load_from_disk(folder)
        # Filter out the too short ones
        data = data.map(trim_nans) # remove trivials (only long enough because of NaNs) TODO Maybe ok at beginning??
        data = data.filter((lambda x: len(x["target"])>=min_length), num_proc=8)
        data = data["train"] if "train" in data else data  # unnest 

        # Group all the TS from the same DS together # 
        id_col = "id" if "id" in data.column_names else "item_id"
        # get all the names
        ds_names = data[id_col]
        ds_groups = group_similar_words(set(ds_names))
        # sort groups by size to speed up filtering
        ds_name_counter = Counter(ds_names)
        ds_groups = sorted(ds_groups, key=lambda group: sum([ds_name_counter[ds_name] for ds_name in group]), reverse=True)
        group_sizes = []
        for group in ds_groups:
            sizes = [ds_name_counter[ds_name] for ds_name in group]
            group_sizes.append(sum(sizes))
            print(sum(sizes), sizes, group)
        print("Dataset Type: ", data.format['type'])
        datasets = []
        unassigned = set(range(len(data)))
        row_ds = []
        for group, size in tqdm(zip(ds_groups, group_sizes), total=sum(x > 5 for x in group_sizes), desc="Splitting Complex Dataset into Subsets"):
            if size <= 5: # All groups consist only of a few ts --> Row DS (Idea: harmless to treat 5 TS as diff DS other than treating 100 TS as diff DS)
                row_ds = data.select(unassigned)
                unassigned = []
                break
            group_set = set(group)
    
            # Build index list from unassigned items only
            assigned = [i for i in unassigned if data[i][id_col] in group_set]
            
            if assigned:
                datasets.append(data.select(assigned))
                unassigned -= set(assigned)
        assert len(unassigned) == 0

        for dataset in datasets:
            corpus.append(SplitDataSet(dataset, deduplication=deduplication))
        for dataset in row_ds:
            corpus.append(RowDataSet(dataset, deduplication=deduplication))
            
    except FileNotFoundError as e:
        print(e, str(folder))
        subfolders = [f for f in folder.iterdir() if not f.name.startswith('.') and f.is_dir()] # filter out hidden folders like .cache and files like .md
        random.shuffle(subfolders) # Shuffle to use Caching (later) as parallelization
        print(subfolders)
        for subfolder in tqdm(subfolders, desc=f"Loading datasets in {folder.name}"):
            proc = psutil.Process(os.getpid())
            print(f"Python RAM usage: {proc.memory_info().rss / 1e6:.2f} MB")
            print(f"Loading Folder: {subfolder.name}", flush=True)
            corpus.append(SplitDataSet(subfolder, deduplication=deduplication))
    return corpus


def load_datasets(data_path: Path, corpora: Union[List[str], str], min_length=128, deduplication = True) -> dict:
    corpus = []
    for folder in tqdm(list(data_path.iterdir()), desc="Loading datasets"):
        if ("all" in corpora and folder.name != "Time_Corpus") or (isinstance(corpora, list) and folder.name in corpora):
            print(f"Loading {folder.name}")
            corpus += load_folder(folder, min_length=min_length, deduplication=deduplication)
    if not corpus:
        raise ValueError("No matching datasets found.")
    return corpus

def trim_nans(dataset):
    arr = np.array(dataset["target"])
    mask = ~np.isnan(arr)
    if not mask.any():
        return []
    start_point = mask.argmax()
    endpoint = len(mask) - mask[::-1].argmax()
    dataset["target"] = dataset["target"][start_point:endpoint]
    dataset["timestamp"] = dataset["timestamp"][start_point:endpoint]
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
    id_cols = list({"item_id", "id", "name"}.intersection(batch.keys()))
    print(id_cols)
    batch_len = len(batch[target_column])
    # Either take official id or assign to each var its own
    ids = batch[id_cols[0]] if len(id_cols) > 0 else list(range(batch_len))
    timestamps = [pd.to_datetime(batch["timestamp"][i]) for i in range(batch_len)] if "timestamp" in batch else []
    starts = batch["start"] if "start" in batch else [timestamps[i].min() for i in range(batch_len)]
    freqs = batch["freq"] if "freq" in batch else [pd.infer_freq(timestamps[i]) for i in range(batch_len)]

    proc = psutil.Process(os.getpid())
    print(f"Python RAM usage: {proc.memory_info().rss / 1e6:.2f} MB", flush=True)

    q = 0
    for iid, st, fq, targets in zip(ids, starts, freqs, batch[target_column]):
        if not isinstance(targets[0], (tuple, list)): 
            targets = [targets]
        for tgt in targets:
            startpoint = 0
            while startpoint < len(tgt): # Split into smaller ts to optimize loading
                # print(startpoint, flush=True)
                out["item_id"].append(iid)
                out["start"].append(st)
                out["freq"].append(fq)
                out["target"].append(tgt[startpoint:int(startpoint+1e4)])
                startpoint += int(1e4)

    return out


@app.command()
@use_yaml_config(param_name="config")
def create_dataset(
    corpora: List[str] = ['all'],
    k: int = 1,
    length: int = 128,
    samples: int = 10000,
    alpha: float = 1.5,
    #min_probability: float = 0.000025, 
    small_ts_share: float = 0.3,
    deduplication: bool = True,
    workers: int = -1,
    data_path: str = "./data/data_sets_raw",
    output_dir: str = "./data/train"
):
    if workers > -1:
        global WORKERS
        WORKERS = workers

    hp = dict(deepcopy(locals()))
    hp_hash = hash_dict(hp)
    hp["hash"] = hp_hash
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_adr: Path = output_dir / f"{hp_hash}.arrow"
    if output_adr.exists():
        raise Exception("Corpus already exists; the provided configuration might be a duplicate.")

    insertTable(
        table_name="corpora",
        row_data=hp,
    )
    logging.info(f"Loading datasets from {data_path.resolve()}")
    corpus = load_datasets(data_path, corpora, min_length=length, deduplication=deduplication)
    print(f"Loaded {len(corpus)} datasets from {data_path.resolve()}")

    ds_lengths = [len(record) for record in corpus]
    corpus_length = sum(ds_lengths)
    print(f"Corpus Size: {corpus_length}")
    ds_count = len(ds_lengths)
    print(f"Number DS: {ds_count}")
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
    print("Probability of sampling from the largest Contributors: ", ds_probability[big_ones])
    # assert np.all(np.array(ds_probability) >= min_probability), "Some probabilities are below the minimum probability threshold."

    logging.info("Creating snippets")
    def create_augmented_snippet(_):
        augmented_chunk = []
        while len(augmented_chunk) < CHUNK_SIZE:
            datasets_ids = np.random.choice(ds_count, size=k, replace=False, p=ds_probability) # TODO Replace true??
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
    corpus_augmented = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(create_augmented_snippet, i) for i in range(chunk_count)]
        for f in tqdm(as_completed(futures), total=chunk_count, desc="Creating augmented snippets"):
            try:
                corpus_augmented.extend(f.result())
            except Exception as e:
                logging.error(f"Worker failed: {e}")
                raise e

    # Convert to final format -> save to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    convert_to_arrow(output_adr, corpus_augmented[:samples])
    print("All Done!")



if __name__ == "__main__":
    app()
