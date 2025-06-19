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

# --- Config ---
logging.basicConfig(level=logging.INFO)
CHUNK_SIZE = 100

app = Typer()

# --- Utility Functions ---
def convert_to_arrow(path: Union[str, Path], time_series: List[np.ndarray], compression: str = "lz4"):
    start = np.datetime64("2000-01-01 00:00", "s")
    dataset = [{"start": start, "target": ts} for ts in time_series]
    ArrowWriter(compression=compression).write_to_file(dataset, path=path)


def load_datasets(data_path: Path, corpora: Union[List[str], str]) -> dict:
    dict_dataset = {}
    for folder in tqdm(list(data_path.iterdir()), desc="Loading datasets"):
        if corpora == "all" and folder.name != "Time_Corpus":
            dict_dataset[folder.name] = ds.load_from_disk(folder)
        elif isinstance(corpora, list) and folder.name in corpora:
            dict_dataset[folder.name] = ds.load_from_disk(folder)
        elif isinstance(corpora, str) and folder.name == corpora:
            dict_dataset[folder.name] = ds.load_from_disk(folder)
    if not dict_dataset:
        raise ValueError("No matching datasets found.")
    return dict_dataset

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


@app.command()
def create_dataset(
    corpora: Union[List[str], str] = "all",
    k: int = 1,
    length: int = 128,
    samples: int = None,
    alpha: float = 1.5,
    min_probability = 0.00002,
    workers: int = 4,
    data_path: Path = Path("./data/data_sets_raw"),
    output_dir: Path = Path("data/train")
):
    logging.info(f"Loading datasets from {data_path.resolve()}")
    dict_dataset = load_datasets(data_path, corpora)

    combined = ds.concatenate_datasets([
        ds_split["train"] for ds_split in dict_dataset.values()
    ])
    # Filter out the too short ones
    combined = combined.map(trim_nans) # remove trivials (only long enough because of NaNs) TODO Maybe ok at beginning??
    combined = combined.filter(lambda x: len(x["value"])>=length)

    ds_lengths = [len(record["value"]) for record in combined]
    corpus_length = sum(ds_lengths)
    ds_count = len(ds_lengths)
    assert np.all(np.array(ds_lengths) >= length)

    # calculate probs
    assert ds_count * min_probability < 1, "Minimum probability too high for the number of datasets, can't be satisfied." # If == 1 would be equal weights
    ds_probability = np.array([max(ds_length / corpus_length, min_probability) for ds_length in ds_lengths])
    while True:
        total_raise = sum(ds_probability) - 1 # raise through capping
        if total_raise <= min_probability:
            # Likely just rounding errors, break
            break
        big_pie = sum(ds_probability[ds_probability > min_probability])
        if big_pie == 0:
            ds_probability = np.ones_like(ds_probability) / len(ds_probability)  # Equal weights if all are below min_probability
            break 
        ds_probability = np.array([max(prob * (1-(total_raise/big_pie)), min_probability) for prob in ds_probability])
    assert 1 - min_probability <= sum(ds_probability) <= 1 + min_probability, "Probabilities do not sum to near 1, something went wrong."
    assert np.all(np.array(ds_probability) >= min_probability), "Some probabilities are below the minimum probability threshold."

    logging.info("Creating snippets")

    def create_augmented_snippet(_):
        augmented_chunk = []
        while len(augmented_chunk) < CHUNK_SIZE:
            datasets_ids = np.random.choice(ds_count, size=k, replace=False, p=ds_probability)
            snippets = []
            for ds_id in datasets_ids:
                dataset = combined[int(ds_id)]
                # print(dataset["value"])
                start_point = np.random.randint(0, len(dataset["value"]) - length + 1)
                snippets.append(dataset["value"][start_point:start_point + length])
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
    create_dataset(
        corpora="Time_Corpus_Processed",
        k=3,
        length=128,
        samples=1000, #int(1e6),
    )
