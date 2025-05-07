import datasets as ds
import numpy as np
import random
import pandas as pd

from itertools import repeat
from concurrent import futures
from pathlib import Path
from typing import List, Union
from gluonts.dataset.arrow import ArrowWriter


def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    compression: str = "lz4",
):
    """
    Store a given set of series into Arrow format at the specified path.

    Input data can be either a list of 1D numpy arrays, or a single 2D
    numpy array of shape (num_series, time_length).
    """
    assert isinstance(time_series, list) or (
        isinstance(time_series, np.ndarray) and time_series.ndim == 2
    )

    # Set an arbitrary start time
    start = np.datetime64("2000-01-01 00:00", "s")

    dataset = [{"start": start, "target": ts} for ts in time_series]

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )


def create_snippets(length: int, dataset: ds.Dataset):
    values = dataset["value"]

    if len(values) < length:
        return None

    snippets = []
    while values and len(values) >= length:
        snippet = random.sample(values, length)

        for value in snippet:
            values.remove(value)

        snippets.append(snippet)

    return snippets


def create_dataset(
    load_corpui: Union[List, str] = "all", k: int = 1, length: int = 128
):
    cwd = Path.cwd()
    data_path = cwd / "data_sets_raw"

    df_preselect = pd.DataFrame()

    dict_dataset = {}
    for dataset_folder in data_path.iterdir():
        if dataset_folder.name in load_corpui:
            dataset = ds.load_from_disk(dataset_folder)
            dict_dataset[dataset_folder.name] = dataset
        elif load_corpui.lower() == "all":
            if dataset_folder.name != "Time_Corpus":
                dataset = ds.load_from_disk(dataset_folder)
                dict_dataset[dataset_folder.name] = dataset
        else:
            raise ValueError(
                f"The value or values provided in {load_corpui} did not match any available Dataset. Please"
                f"choose one or more of the following Datasets any combination will work: Time_Corpus_Processed, "
                f"Lotsa or Chronos. Defaults to all combinations"
            )

    corpi_data = []
    if isinstance(load_corpui, list):
        for corpus in load_corpui:
            data_train = dict_dataset[corpus.name]["train"]
            corpi_data.append(data_train)
        combined_corpus = ds.concatenate_datasets(corpi_data)
    elif load_corpui.lower() == "all":
        for dataset in dict_dataset.keys():
            corpi_data.append(dict_dataset[dataset]["train"])
        combined_corpus = ds.concatenate_datasets(corpi_data)
    else:
        combined_corpus = dict_dataset[load_corpui]

    lst_data = []
    for data_set in combined_corpus:
        lst_data.append(data_set)
    with futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(create_snippets, repeat(length), lst_data))

    df_snippets = pd.DataFrame(columns=["snippets"])
    for i, result in enumerate(results):
        df_snippets["snippets"].loc[i] = result

    indices = np.arange(len(combined_corpus)).tolist()
    while indices and indices >= k:
        random_selection = random.sample(range(len(combined_corpus)), k)

        for removal in random_selection:
            indices.remove(removal)

        df_preselect["selection"] = random_selection

        for selection in random_selection:
            if len(combined_corpus[selection]) < length:
                print(
                    f"Dataset smaller than {length} found in random dataset combination {random_selection}! Discarding combination and removing selection"
                )

    convert_to_arrow(path, ts)


if __name__ == "__main__":
    load_corpui = "Time_Corpus_Processed"
    k = 3
    length = 128

    create_dataset(load_corpui, k, length)
