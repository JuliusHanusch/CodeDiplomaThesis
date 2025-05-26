import datasets as ds
import numpy as np
import pandas as pd

from itertools import repeat
from concurrent import futures
from pathlib import Path
from typing import List, Union
from gluonts.dataset.arrow import ArrowWriter
from ts_mixup import ts_mixup


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


def create_snippets(n: int, length: int, dataset: dict):
    values = dataset["value"][0]

    if len(values) < length:
        return None

    n = min(len(values) - length, n)
    snippets = []
    start_points = np.random.choice(
        list(range(len(values) - length)), n, replace=False
    ).tolist()
    for start_point in start_points:
        snippet = values[start_point : start_point + length + 1]
        snippets.append(snippet)

    return snippets


def create_dataset(
    load_corpui: Union[List, str] = "all",
    k: int = 1,
    n: int = 50,
    length: int = 128,
    alpha: int = 1.5,
):
    print("Starting Load process")
    data_path = Path("./data/data_sets_raw")

    dict_dataset = {}
    for dataset_folder in data_path.iterdir():
        if dataset_folder.name in load_corpui:
            dataset = ds.load_from_disk(dataset_folder)
            dict_dataset[dataset_folder.name] = dataset
        elif dataset_folder.name == load_corpui:
            dataset = ds.load_from_disk(dataset_folder)
            dict_dataset[dataset_folder.name] = dataset
        elif load_corpui.lower() == "all":
            if dataset_folder.name != "Time_Corpus":
                dataset = ds.load_from_disk(dataset_folder)
                dict_dataset[dataset_folder.name] = dataset

    if not dict_dataset:
        raise ValueError(
            f"The value or values provided in {load_corpui} did not match any available Dataset. Please "
            f"choose one or more of the following Datasets any combination will work: Time_Corpus_Processed, "
            f"Lotsa or Chronos. Defaults to all combinations"
        )

    print("Finished Loading Dataset")
    print("Start combining Datasets")

    corpi_data = []
    if isinstance(load_corpui, list):
        save_file = f"tsm_{load_corpui[0]}_and_{load_corpui[1]}_combined_with_k-{k}_length-{length}_alpha-{str(alpha).replace('.','')}.arrow"
        for corpus in load_corpui:
            data_train = dict_dataset[corpus.name]["train"]
            corpi_data.append(data_train)
        combined_corpus = ds.concatenate_datasets(corpi_data)
    elif load_corpui.lower() == "all":
        save_file = f"training_data/tsm_all_combined_with_k-{k}_length-{length}_alpha-{alpha}.arrow"
        for dataset in dict_dataset.keys():
            corpi_data.append(dict_dataset[dataset]["train"])
        combined_corpus = ds.concatenate_datasets(corpi_data)
    else:
        save_file = (
            f"tsm_for_{load_corpui}_with_k-{k}_length-{length}_alpha-{alpha}.arrow"
        )
        combined_corpus = dict_dataset[load_corpui]["train"]

    print("Finished Combining Datasets")
    print("Starting snippet creation")

    results = []
    # for idx in range(len(combined_corpus)):
    #    result = create_snippets(n, length, combined_corpus[idx])
    #    results.append(result)

    for data in combined_corpus:
        result = create_snippets(n, length, data)
        results.append(result)

    # with futures.ProcessPoolExecutor() as executor:
    #    results = list(
    #        executor.map(create_snippets, repeat(n), repeat(length), lst_data)
    #    )

    print("Finished Snippet Creation")
    print("Starting random selection procedure")

    df_snippets = pd.DataFrame(columns=["snippets"])
    for result in results:
        df_tmp = pd.DataFrame()
        if result is None:
            continue
        df_tmp["snippets"] = result
        df_snippets = pd.concat([df_snippets, df_tmp], ignore_index=True)

    ts_data = []
    indices = list(df_snippets.index)
    while indices and len(indices) >= k:
        rows = df_snippets.sample(n=k)
        data = []
        for row in rows:
            index = row.index[0]
            snippets = row["snippets"]
            snippet = np.random.choice(snippets)
            snippets.remove(snippet)
            snippet = np.array(snippet)
            data.append(snippet)
            if snippets:
                row["snippets"] = snippets
            else:
                df_snippets.drop(index, inplace=True)
                indices.remove(index)

        ts_data.append(data)

    print("Finished random selection procedure")
    print("Starting TSMixup procedure")

    with futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(ts_mixup, ts_data, repeat(alpha)))

    convert_to_arrow(f"train/{save_file}", results)


if __name__ == "__main__":
    load_corpui = "Time_Corpus_Processed"
    k = 3
    n = 50
    length = 128

    create_dataset(load_corpui, k, n, length)
