import datasets as ds
import numpy as np
import pandas as pd
import random as rd

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


def create_snippets(length: int, corpus_length: int, samples: int, dataset: dict):
    values = dataset["value"]
    ts_length = len(values)
    print(length)

    if ts_length < length:
        return None

    coverage = ts_length / corpus_length
    if coverage < 0.001:
        coverage = 0.001
    number_snippets = round(samples * coverage)

    start_points = values[:-length]
    if len(start_points) > number_snippets:
        print(
            "The original timeseries does not have enough startpoints to generate the expected amount of snippets!"
        )
        snippets = []
        for i in range(number_snippets):
            start_point = np.random.choice(list(range(len(start_points))), replace=True)
            snippet = values[start_point : start_point + length]
            snippets.append(snippet)
    else:
        snippets = []
        for i in range(number_snippets):
            start_point = np.random.choice(
                list(range(len(start_points))), replace=False
            )
            snippet = values[start_point : start_point + length]
            snippets.append(snippet)

    return snippets


def draw_snippets_with_replacement(df_snippets: pd.DataFrame, iterations: int, k: int):
    ts_data = []

    for i in range(iterations):
        rows = df_snippets.sample(n=k)
        data = []
        for index, row in rows.iterrows():
            snippets = row["snippets"]
            snippet = rd.choice(snippets)
            snippet = np.array(snippet)
            data.append(snippet)

        ts_data.append(data)

    return ts_data


def draw_snippets_without_replacement(df_snippets: pd.DataFrame, k: int):
    ts_data = []
    indices = list(df_snippets.index)
    while indices and len(indices) >= k:
        rows = df_snippets.sample(n=k)
        data = []
        for index, row in rows.iterrows():
            snippets = row["snippets"]
            snippet = rd.choice(snippets)
            snippets.remove(snippet)
            snippet = np.array(snippet)
            data.append(snippet)
            if snippets:
                df_snippets.loc[index, "snippets"] = snippets
            else:
                df_snippets.drop(index, inplace=True)
                indices.remove(index)

        ts_data.append(data)

    return ts_data


def create_dataset(
    load_corpui: Union[List, str] = "all",
    k: int = 1,
    length: int = 128,
    samples: int = None,
    alpha: int = 1.5,
):
    data_path = Path("./data/data_sets_raw") #TODO pass as argument
    print(f"Starting Load process, from {data_path.resolve()}")

    dict_dataset = {}
    for dataset_folder in data_path.iterdir():
        if dataset_folder.name in load_corpui: # Includes  or dataset_folder.name == load_corpui
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

    print("Start combining Datasets")
    corpi_data = []
    if isinstance(load_corpui, list):
        save_file = f"tsm_{'_'.join(load_corpui)}_combined_with_k-{k}_length-{length}_alpha-{str(alpha).replace('.','')}.arrow"
        for corpus in load_corpui:
            data_train = dict_dataset[corpus.name]["train"]
            corpi_data.append(data_train)
        combined_corpus = ds.concatenate_datasets(corpi_data)
    elif load_corpui.lower() == "all":
        save_file = f"training_data/tsm_all_combined_with_k-{k}_length-{length}_alpha-{str(alpha).replace('.','')}.arrow"
        for dataset in dict_dataset.keys():
            corpi_data.append(dict_dataset[dataset]["train"])
        combined_corpus = ds.concatenate_datasets(corpi_data)
    else:
        save_file = f"tsm_for_{load_corpui}_with_k-{k}_length-{length}_alpha-{str(alpha).replace('.','')}.arrow"
        combined_corpus = dict_dataset[load_corpui]["train"]


    print("Starting snippet creation")
    corpus_length = 0
    for data in combined_corpus:
        corpus_length += len(data)







    results = []
    for data_set in combined_corpus:
        result = create_snippets(length, corpus_length, samples, data_set)
        results.append(result)

    # with futures.ProcessPoolExecutor() as executor:
    #    results = list(
    #        executor.map(
    #            create_snippets,
    #            repeat(length),
    #            repeat(corpus_length),
    #            repeat(samples),
    #            combined_corpus,
    #        )
    #    )

    print("Finished Snippet Creation")
    print("Starting random selection procedure")

    df_snippets = pd.DataFrame(columns=["snippets"])
    for result in results:
        if result is None:
            continue
        df_tmp = pd.DataFrame()
        df_tmp["snippets"] = [result]
        df_snippets = pd.concat([df_snippets, df_tmp], ignore_index=True)

    if samples is not None:
        ts_data = draw_snippets_with_replacement(df_snippets, samples, k)
    else:
        ts_data = draw_snippets_without_replacement(df_snippets, k)

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
    samples = int(1e6)

    create_dataset(load_corpui, k, length, samples)
