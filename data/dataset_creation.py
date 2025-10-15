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
# TODO Unify import paths e.g. from ts_mixup --> from data.ts_mixup
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
import gc
import pyarrow as pa
import pyarrow.ipc as ipc
import shutil
import time
import torch
from torch.utils.data import DataLoader


CACHE = Path("./cache/data")
CACHE.mkdir(parents=True, exist_ok=True)

# --- Config ---
logging.basicConfig(level=logging.INFO)
print(f"PYTHON_PID={os.getpid()}", flush=True)
CPU_COUNT = int(os.environ.get("SLURM_CPUS_PER_TASK", psutil.cpu_count(logical=True))) # get from slurm else standaard
WORKERS = CPU_COUNT 


app = Typer()


class DatasetAdapter(ABC):
    """
    Idea: We get DS in different shapes (Every Corpus has its own unique shape)
    Time Corpus: Fits entire DS into rows
    Chronos: Fits Datasets into splits and TS into rows
    Lotsa: Similar to Chronos just that each DS is separate from the others (i.e. not just diff split)
    Performing in depth preprocessing to bring all into the same shape somehow is too difficult
    Instead we add to each DS such an adapter with the fundamental functions we require
    Allows to translate rules according to the shape of each DS 
    """
    # Init
    def __init__(self, target_column):
        self.target_col = target_column
            

    @abstractmethod
    def __len__(self):
        pass 

    
    # get random snippet of length n
    @abstractmethod
    def get_random_snippet(self, length: int) -> List[float]:
        """
        Method to get a random TS-Snippet of the given length from the dataset 
        """
        pass


# ========================= #
# --- Utility Functions --- #
# ========================= #

def convert_to_arrow(path: Union[str, Path], time_series: List[np.ndarray], compression: str = "lz4"):
    """
    Takes in List of Time Series and Writes it to an Arrow File  
    """
    start = np.datetime64("2000-01-01 00:00", "s")
    dataset = [{"start": start, "target": ts} for ts in time_series]
    ArrowWriter(compression=compression).write_to_file(dataset, path=path)


def no_collate(batch):
    """Function for pytorch's DataLoader to avoid converting Batches into Tensors"""
    return batch  


def load_folder(folder: Path, min_length = 128, large_ds_thr=10_000, dataloaders=8):
    """
    Loops over all datasets inside a folder and converts them into RowDatasets or SplitDatasets
    """
    print(f"Loading Folder: {folder}")
    corpus = []
    try:
        # Assumption: Folder is a HF Dataset --> Load it
        data = ds.load_from_disk(folder)
        # Filter out the Time Series that are too short
        data = data.map(trim_nans, num_proc=WORKERS) # remove trivials (only long enough because of NaNs) 
        data = data.filter((lambda x: len(x["target"])>=min_length), num_proc=8)
        data = data["train"] if "train" in data else data  # unnest 

        # ========================================== #
        # Group all the TS from the same DS together # 
        # ========================================== #
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
        
        # Do the actual Grouping
        grouped_datasets = []
        unassigned = set(range(len(data)))
        row_ds = []
        for group, size in tqdm(zip(ds_groups, group_sizes), total=sum(x > 5 for x in group_sizes), desc="Splitting Complex Dataset into Subsets"):
            if size <= 5: 
                # Treat members of small groups as independent (mostly for performance reasons)
                row_ds = data.select(unassigned)
                unassigned = [] # ~ unassigned - unassigned 
                break
    
            # Build index list from unassigned items only
            group_set = set(group)
            assigned = [i for i in unassigned if ds_names[i] in group_set]
            
            if assigned:
                # Create Group and add to collection of groups
                grouped_datasets.append(data.select(assigned))
                unassigned -= set(assigned)
        assert len(unassigned) == 0

        # Convert Groups into Datasets and add to Corpus
        for dataset in grouped_datasets:
            corpus.append(SplitDataSet(dataset, large_ds_thr=large_ds_thr, dataloaders=dataloaders))
        for dataset in row_ds:
            corpus.append(RowDataSet(dataset))
            
    except FileNotFoundError as e:
        # Assumption: Folder is a Folder of several HF Datasets
        
        # filter out files (like .md) and hidden folders (like .cache)
        subfolders = [f for f in folder.iterdir() if not f.name.startswith('.') and f.is_dir()] 

        # Shuffle to parallelize Caching (later we prepare SplitDataSets and cache the results, thats slow, but can be done stupidly in parallel)
        random.shuffle(subfolders) 

        print(subfolders)
        # Load all DS in Folder and add to corpus
        for subfolder in tqdm(subfolders, desc=f"Loading datasets in {folder.name}"):
            print_ram()
            print(f"Loading Folder: {subfolder.name}", flush=True)
            corpus.append(SplitDataSet(subfolder, large_ds_thr=large_ds_thr, dataloaders=dataloaders))

    return corpus


def load_datasets(data_path: Path, corpora: List[str], min_length=128, large_ds_thr=10_000, dataloaders=8) -> List[DatasetAdapter]:
    """
    Creates a large combined corpus by collecting all corpora included in the corpora list from the data_path folder and converting them into DataSetAdapters
    """
    corpus = []
    for folder in tqdm(list(data_path.iterdir()), desc="Loading datasets"):
        if folder.name in corpora:
            print(f"Loading {folder.name}")
            corpus += load_folder(folder, min_length=min_length, large_ds_thr=large_ds_thr, dataloaders=dataloaders)
    if not corpus:
        raise ValueError("No matching datasets found.")
    return corpus


def trim_nans(dataset):
    """
    Removes all missing values from begin and end of Time Series
    """
    arr = np.array(dataset["target"])
    mask = ~np.isnan(arr)
    if not mask.any():
        return []
    # Get index of first none missing value
    start_point = mask.argmax()
    # Get index of last none missing value
    endpoint = len(mask) - mask[::-1].argmax()
    # Cut it down and return
    dataset["target"] = dataset["target"][start_point:endpoint]
    dataset["timestamp"] = dataset["timestamp"][start_point:endpoint]
    return dataset


def print_ram():
    """Prints the RAM usage of the current Process"""
    proc = psutil.Process(os.getpid())
    print(f"Python RAM usage: {proc.memory_info().rss / 1e6:.2f} MB", flush=True)


def explode_batch(batch, target_column):
    """
    Function for datasets' .map() 
    Takes in a batch of samples
    Each sample can contain either
    a very long ts that shall be splitted into smaller ones to save resources (Note: Many small ones are easier to store and load via datasets)
    or several time series (i.e. one multivariate one) that shall be split into several independent ones
    or potentially both
    """
    out = {"item_id": [], "start": [], "freq": [], "target": []}

    # For all of the necessary columns either take the given value or insert a placeholder if none is given 
    id_cols = list({"item_id", "id", "name"}.intersection(batch.keys()))
    batch_len = len(batch[target_column])
    ids = batch[id_cols[0]] if len(id_cols) > 0 else list(range(batch_len))
    timestamps = [pd.to_datetime(batch["timestamp"][i]) for i in range(batch_len)] if "timestamp" in batch else []
    starts = batch["start"] if "start" in batch else [timestamps[i].min() for i in range(batch_len)]
    freqs = batch["freq"] if "freq" in batch else [pd.infer_freq(timestamps[i]) for i in range(batch_len)]

    # Debugging Only
    print_ram()

    # Explode batch
    for iid, st, fq, targets in zip(ids, starts, freqs, batch[target_column]):
        # If target is not list of lists --> convert it into one
        if not isinstance(targets[0], (tuple, list)): 
            targets = [targets]

        # Multivariat -> Univariat
        for tgt in targets:
            # Long TS -> small TS
            startpoint = 0
            while startpoint < len(tgt): 
                out["item_id"].append(iid)
                out["start"].append(st)
                out["freq"].append(fq)
                out["target"].append(tgt[startpoint:int(startpoint+1e4)])
                startpoint += int(1e4)

    return out


def check_and_del_corrupted_checkpoint(checkpoint_dir: Path) -> bool:
    """
    Loops over all checkpoints in the directory and deletes the corrupted ones
    If it had to delete some files returns False else returns true
    """
    def check_arrow_file(path: Path) -> bool:
        """Tries to load the given file. On success returns True else False"""
        try:
            # First try IPC (Arrow .arrow / .feather)
            with pa.memory_map(str(path), "r") as source:
                reader = ipc.RecordBatchFileReader(source)
                _ = reader.read_all()  # force load
            return True
        except:
            return False
        
    aok = True
    for file in tqdm(list(checkpoint_dir.iterdir()), desc="Checking checkpoints"):
        if file.is_file():
            if check_arrow_file(file):
                continue
            else:
                # Delete Corrupted File
                file.unlink()
                aok = False
    return aok


# ================================== #
# Adapters For The different Corpora #
# ================================== #
class RowDataSet(DatasetAdapter):
    """
    Dataset for corpora where each row is an DS
    --> Consists of a single Time Series
    """
    def __init__(self, dataset, target_column: str = "target"):
        super().__init__(target_column)
        self.target = dataset[self.target_col] # Drop everything except target column to speed up look ups later


    def __len__(self):
        return len(self.target) 

    def get_random_snippet(self, length: int, **kwargs): 
        start_point = torch.randint(max(len(self) - length + 1, 1), (1,)).item()
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
    def __init__(self, dataset_path: Path | ds.Dataset, target_column: str = "target", large_ds_thr=10_000, dataloaders=8):
        super().__init__(target_column)
        self.large_ds_thr = large_ds_thr
        self.dataloaders = dataloaders

        if isinstance(dataset_path, (Path, str)):
            # Get corresponding cash adress
            cache_adr = CACHE / dataset_path.name
            if cache_adr.exists():
                # Load from Cache
                dataset = ds.load_from_disk(cache_adr)
            else:
                # Apply naming/structure conventions, Split Data into more manageable chunks and write to cache
                dataset = ds.load_from_disk(dataset_path)

                if "train" in dataset:
                    dataset = dataset["train"]

                # ================================================= #
                # Identify all columns that could be target columns #
                # ================================================= #
                columns_to_exclude = {"timestamp", "id", "item_id", "start", "freq"}
                # Drop all columns with only a single value (else we would check them a few million times if they are still single val)
                cols_to_keep = set(get_cols_with_lists(dataset))
                # Also keep the expected columns
                cols_to_keep = cols_to_keep.union(columns_to_exclude)
                dataset = dataset.remove_columns([col for col in dataset.column_names if col not in cols_to_keep])

                columns = set(dataset.column_names)
                potential_targets = list(columns - columns_to_exclude)
                print("Columns:", potential_targets)
                print("Rows:", len(dataset))

                if len(potential_targets) > 1: 
                    # There are multiple potential target columns --> combine them into one multivariate one (gets exploded in the next step)
                    def concat_lists(example):
                        """Combine several columns into one multivariate one (list of lists) + drop constant columns + Apply Naming Convention"""
                        targets = []
                        for col in potential_targets:
                            if isinstance(example[col], list): # Check again (just to be sure)
                                if isinstance(example[col][0], list): # List of lists
                                    targets += [
                                        example[col][i] 
                                        for i in range(len(example[col]))
                                        if isinstance(example[col][i], list) and len(set(example[col][i])) > 1 # Check that not constant
                                    ]
                                elif len(set(example[col])) > 1: # Check that not constant
                                    targets.append(example[col])
                        # Verify that all targets are int or float
                        old_variety = len(targets)
                        targets = [target for target in targets if isinstance(target[0], (float, int))]
                        assert old_variety == len(targets) # Other Types are just Not Implemented Yet

                        example[self.target_col] = targets
                        return example
                    
                    # Rename old target column to avoid conflicts when overwritting it with the Multivariate one
                    if self.target_col in dataset.column_names: 
                        dataset = dataset.rename_column(self.target_col, self.target_col + '_old')
                        potential_targets = [(self.target_col + '_old' if col == self.target_col else col) for col in potential_targets] # update target col name

                    dataset = dataset.map(concat_lists, num_proc=WORKERS) 
                elif potential_targets[0] != target_column:
                    # Apply naming convention and continue
                    dataset = dataset.rename_column(potential_targets[0], target_column)
                else:
                    # Everything is already AOK
                    pass
                    
                long_ts = len(dataset[0][self.target_col]) > 15_000
                if isinstance(dataset[0][self.target_col][0], list) or long_ts:

                    # flatten Multivariate to multiple univariates + Split long TS into multiple smaller chunks to speed up loading from disk later
                    dataset = dataset.map(
                        partial(explode_batch, target_column=target_column),
                        batched=True,
                        batch_size=200,
                        remove_columns=[c for c in dataset.column_names if c not in ("item_id","start","freq")],
                        num_proc=WORKERS,
                    )
                dataset.save_to_disk(cache_adr)

        elif isinstance(dataset_path, ds.Dataset):
            # dataset_path isn't a path --> rename var 
            dataset = dataset_path

        # Load a small sample (for estimating size and structure of ds)
        self.target = dataset.shuffle(seed=None).take(min(100, len(dataset)))[self.target_col]
        assert not isinstance(self.target[0][0], (list, tuple)), "Multivariate DS aren't supported yet"

        # Load actual data 
        self._dataset = dataset 
        self.dl = self.data_loader()


    def __len__(self):
        '''ESTIMATE full ds size based on chunk size'''
        lengths = [len(ts) for ts in self.target]
        avg_length = sum(lengths)/len(lengths)
        return int(avg_length * len(self._dataset))


    def data_loader(self):
        """
        Generator to loop over all TS in the given dataset indefinetly.\\
        Loads small DS into ram (very fast after loading).\\
        Streams large DS via Pytorch (i.e. replaces RAM with CPUs)\\
        Note: Loading into RAM doesn't lead to OOMs (though it stalls) loading too many samples via torch to CPUs does.
        Note: First sampling from "small" datasets takes very long because loading into mem but then very fast.
        
        Args:
        ----------
        large_ds_threshold : int
            From how many time series onward do we consider the dataset too large to load into memory.
            
        max_worker_count : int
            How many workers can pytorch use for speeding up loading from disk 
            Only matters if dataset surpasses large_ds_threshold
            If more than WORKERS it's capped to WORKERS instead
            The higher large_ds_threshold the larger this value can be as each large DS gets that many workers 
            --> if too many DS considered large then too many workers are created 
            --> OOM 
        """

        if len(self._dataset) <= self.large_ds_thr:
            print("Loading To Mem", flush=True)
            # Load small DS into RAM --> very fast
            target = self._dataset.shuffle(seed=None, keep_in_memory=True)[self.target_col]
            while True:
                for sample in target:
                    yield sample
        else:
            # Load large DS via Torch --> Better worst case performance
            data = self._dataset.remove_columns([c for c in self._dataset.column_names if c != self.target_col])
            dl = DataLoader(data, num_workers=min(self.dataloaders, WORKERS), prefetch_factor=4, batch_size=32, collate_fn=no_collate, shuffle=True)
            while True:
                iterator = iter(dl)
                for batch in iterator:
                    for sample in batch:
                        yield sample[self.target_col]


    def get_random_snippet(self, length: int, **kwargs) -> List[float]:
        ts = next(self.dl)

        start_point = torch.randint(max(len(ts) - length + 1, 1), (1,)).item()
        return ts[start_point:start_point + length]


@app.command()
@use_yaml_config(param_name="config")
def create_dataset(
    corpora: List[str] = [],
    k: int = 1,
    length: int = 128,
    samples: int = 10000,
    alpha: float = 1.5,
    small_ts_share: float = 0.1,
    workers: int = -1,
    dataloaders: int = 8, # Max Number of workers per large DS to load from Disk 
    large_ds_thr: int = 10_000, # Threshold for how many Time Series need to be in a Dataset to stream it from disk instead of loading it to RAM
    data_path: str = "./data/data_sets_raw",
    output_dir: str = "./data/train"
):
    if "--config" in sys.argv:
        conf_name = Path(sys.argv[sys.argv.index("--config")+1]).stem
    else:
        conf_name = "CLI"
    
    # If given Overwrite Worker Count else one per CPU
    if workers > -1:
        global WORKERS
        WORKERS = workers

    # Path Management
    hp = dict(deepcopy(locals()))
    hp_hash = hash_dict(hp)
    hp["hash"] = hp_hash
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_adr: Path = output_dir / f"{conf_name}_{hp_hash}.arrow"
    if output_adr.exists():
        raise Exception("Corpus already exists; the provided configuration might be a duplicate.")

    # Track Config Used in Database
    insertTable(
        table_name="corpora",
        row_data=hp,
    )

    # Load Data from disk
    logging.info(f"Loading datasets from {data_path.resolve()}")
    corpus = load_datasets(data_path, corpora, min_length=length, large_ds_thr=large_ds_thr, dataloaders=dataloaders)
    print(f"Loaded {len(corpus)} datasets from {data_path.resolve()}")

    # Check Size
    # Cap Size estimate to 500M per DS else it gets too biased and slow (as large DS get streamed from disk and we use them more the higher this estimate is)
    ds_lengths = [min(len(dataset), 500_000_000) for dataset in corpus] 
    corpus_length = sum(ds_lengths)
    print(f"Corpus Size: {corpus_length}")
    ds_count = len(ds_lengths)
    print(f"Number DS: {ds_count}")

    # ================================ #
    # Universal Basic Income Principle #
    # ================================ #
    # Calculate Additive smoothing factor
    ubi = small_ts_share / ds_count 

    # calculate probs of sampling from each dataset in corpus (percentual size of entire corpus + ubi)
    ds_probability = np.array([(ds_length / corpus_length) + ubi for ds_length in ds_lengths])
    # Renormalize 
    ds_probability = ds_probability / sum(ds_probability)
    print("Probs:", np.sort(ds_probability), flush=True)
    ds_probability_tensor = torch.tensor(ds_probability)

    assert np.all((0 < ds_probability) & (ds_probability < 1))

    logging.info("Creating snippets")
    
    samples_per_checkpoint = 10_000
    corpus_augmented = []
    tmp_folder = output_adr.with_suffix("")
    tmp_folder.mkdir(parents=True, exist_ok=True)
    checkpoints = [int(checkpoint.stem) for checkpoint in tmp_folder.iterdir() if checkpoint.is_file()]
    batch_count = (samples+samples_per_checkpoint-1)//samples_per_checkpoint
    loop = tqdm(range(len(checkpoints), batch_count), desc="Creating augmented snippets")

    for _ in loop:
        # Create a Batch of Augmented Samples
        while len(corpus_augmented) < samples_per_checkpoint:
            # Select k Snippets from Corpus Randomly
            datasets_ids = torch.multinomial(ds_probability_tensor, k, replacement=True)
            snippets = []
            for ds_id in datasets_ids:
                ds_id = int(ds_id)
                dataset = corpus[ds_id]
                snippets.append(dataset.get_random_snippet(length=length))

            # Drop too short ones
            snippets = [snip for snip in snippets if len(snip) >= length]
            if len(snippets) == 0:
                continue # skip empties

            # Combine via TS-MixUp
            final_sample = ts_mixup(snippets, alpha=alpha)

            # Quality insurance
            if np.isnan(np.array(final_sample)).sum() / length < 0.25:  # Less than 25% NaNs
                corpus_augmented.append(final_sample)

        # Checkpoint Batch to disk
        print_ram()

        # Checkpointing (due to HW problems, packaging problems, long runtime) 
        # Which checkpoint(name)s exist already 
        checkpoints = [int(checkpoint.stem) for checkpoint in tmp_folder.iterdir() if checkpoint.is_file()]
        if len(checkpoints)*samples_per_checkpoint >= samples:
            # Check and delete corrupted files if none found break 
            aok = check_and_del_corrupted_checkpoint(tmp_folder)
            if aok: # Only break if we didn't need to delete a corrupted checkpoint
                break
            
        # get lowest missing number (to keep checkpoints tidy)
        expected_checkpoints = set(range(len(checkpoints)+1))
        missing_checkpoints = expected_checkpoints - set(checkpoints)
        checkpoint_name = min(list(missing_checkpoints))

        # Write Batch to disk to save Memory (RAM)
        tmp_adr = tmp_folder / f"{checkpoint_name}.arrow"
        convert_to_arrow(tmp_adr, corpus_augmented)
        print(f"Saved Checkpoint to {tmp_adr}", flush=True)
        # Del Pointer to it
        corpus_augmented = []
        gc.collect()

    print("All Checkpoints Created!", flush=True)

    # ======================================= #
    # Convert to final format -> save to disk #
    # ======================================= #
    # Clear Mem (probably unnecessary as problems dont seem to arise from OOM but HW failure)
    del corpus
    del corpus_augmented
    gc.collect()
    print_ram()

    # Load Checkpoints From Disk
    corpus_augmented = []
    for file in tqdm(list(tmp_folder.iterdir()), desc="Load & Concat Checkpoints"):
        if file.is_file():
            with pa.memory_map(str(file), "r") as source:
                reader = ipc.RecordBatchFileReader(source)
                table = reader.read_all()
            batch = table["target"]  
            # Combine
            corpus_augmented += [np.array(snippet.as_py()) for snippet in batch]
        if len(corpus_augmented) > samples:
            # We generated too much --> Dont need to load all
            print("Already loaded enough")
            break 

    # Write Back To Disk
    convert_to_arrow(output_adr, corpus_augmented[:samples])

    # del tmp folder with checkpoints
    shutil.rmtree(tmp_folder)

    print("All Done!")


if __name__ == "__main__":
    app()

