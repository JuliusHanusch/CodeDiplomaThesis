import pandas as pd
from pathlib import Path
import datasets as ds
import yaml
import os
from huggingface_hub import hf_hub_download
from collections import Counter
from warnings import warn 
import sys
import statistics
from eval_frequency import eval_frequency
from tqdm.auto import tqdm
import numpy as np 

data_directory = Path("./data_sets_raw")
min_length = 128

expected_values = {
    "Kaggle" : {
        "number_ts": 29_032,
        "number_data_points": 284_501_006,
        "avg_ts_length": 9_800,
        "avg_var": 1.7e31,
    },
    "Kaggle Split" : {
        "number_ts": 50_632,
        "number_data_points": 304_762_984,
        "avg_ts_length": 6_019,
        "avg_var": 3.5e32,
    },
    "UCI" : {
        "number_ts": 1_876,
        "number_data_points": 68_821_322,
        "avg_ts_length": 36_685,
        "avg_var": 9.5e7,
    },
    "UCI Split" : {
        "number_ts": 648,
        "number_data_points": 51_464_042,
        "avg_ts_length": 79_419,
        "avg_var": 3.5e7,
    },
}

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)

    kaggle_corpus = ds.load_from_disk(data_directory / "Time_Corpus_Processed")["train"]
    kaggle_corpus_split = ds.load_from_disk(data_directory / "Time_Corpus_Processed_Split")["train"]
    uci_corpus = ds.load_from_disk(data_directory / "UCI_Corpus_Processed")["train"]
    uci_corpus_split = ds.load_from_disk(data_directory / "UCI_Corpus_Processed_Split")["train"]
    print(kaggle_corpus)
    print(kaggle_corpus_split)
    print(uci_corpus)
    print(uci_corpus_split)

    corpora = [("Kaggle", kaggle_corpus), ("Kaggle Split", kaggle_corpus_split), ("UCI", uci_corpus), ("UCI Split", uci_corpus_split)]

    # Check for Same Naming Conventions as Chronos + Expected Num Rows
    for name, corpus in corpora:
        print(f"\n{name}:")
        cols = corpus.column_names
        assert "id" in cols 
        assert "timestamp" in cols
        assert "target" in cols
        exp_num_rows = expected_values[name]["number_ts"]
        assert exp_num_rows * 1.05 > len(corpus) > exp_num_rows * 0.95, f"Only {len(corpus)} of {exp_num_rows} original TS in Time Corpus {name} some seem to have gone missing, small changes are expected though as data is not completly stable"
    print("All Corpora Have the Expected Columns and number of Rows")


    # Check That both Corporas are Univariate and not lists of lists
    for name, corpus in corpora:
        for ts in tqdm(corpus):
            assert isinstance(ts["target"], list) and not any(isinstance(entry, list) for entry in ts), f"{name} is not Univariate"
    print("All Corpora Are Univariate")

    # Check for Min Length
    for name, corpus in corpora:
        print(f"\n{name}:")
        lengths = []
        lengths_wo_nan = []
        for ts in tqdm(corpus):
            length = len(ts["target"])
            length_wo_nan = pd.Series(ts["target"]).notna().sum()
            assert length >= min_length # The Min Length we set during Preprocessing
            lengths.append(length)
            lengths_wo_nan.append(length_wo_nan)
        total_length = sum(lengths)
        total_length_wo_nan = sum(lengths_wo_nan)
        print(f"Total Length: {total_length}")
        print(f"Total Length without NaNs: {total_length_wo_nan}")
        exp_length = expected_values[name]["number_data_points"]
        assert (total_length > exp_length * 0.95)
        assert total_length_wo_nan/total_length > 0.95, f"More than 5% are missing values in {name}"
        avg_length = total_length / len(lengths)
        print(f"Avg Length: {avg_length}")
        exp_avg_length = expected_values[name]["avg_ts_length"]
        assert avg_length > exp_avg_length * 0.95
    print("All TS Reach Minimum Desired Length")

    for name, corpus in corpora:
        print(f"\n{name}:")
        vars = []
        strikes = 0
        for ts in tqdm(corpus):
            # ts = np.array(ts["target"])
            var = np.nanvar(ts["target"])
            if not var > 0:
                strikes += 1
            vars.append(var)
        avg_var = sum(vars) / len(vars)
        exp_avg_var = expected_values[name]["avg_var"]
        assert exp_avg_var * 0.95 < avg_var < exp_avg_var * 1.05
        print(f"Average Variance: {avg_var}")
        print(f"Count Constant TS: {strikes}")
        assert strikes == 0  
    print("No TS Seems to be constant")

    # Check For Equidistance
    for name, corpus in corpora:
        print(f"\n{name}:")
        vars = []
        for ts in tqdm(corpus):
            assert eval_frequency(pd.DataFrame(ts), time_column="timestamp", fix=("_Split" in name))
        print("All TS Seem Equidistant")   
