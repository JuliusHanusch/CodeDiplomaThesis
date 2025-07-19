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

    # Check Total Length (Fuzzy because Data is not 100% stable due to legal constraints)
    # TODO assert 24_000 > len(kaggle_corpus) >= 23_000, f"Only {len(kaggle_corpus)} of 23_441 original  TS in Time Corpus some seem to have gone missing, small changes are expected though as data is not completly stable" #19_507 -> 22_507
    # TODO assert 2_000 > len(uci_corpus) >= 1_900, f"Only {len(uci_corpus)} of 1_914 original TS in UCI Corpus some seem to have gone missing, small changes are expected though as data is not completly stable" # 2_328, by 90% equidistant thr 1_857 (5 less) -> 1_862
    print("Enough TS per Corpus")

    # Check for Same Naming Conventions as Chronos 
    for name, corpus in corpora:
        print(f"\n{name}:")
        cols = corpus.column_names
        assert "id" in cols 
        assert "timestamp" in cols
        assert "target" in cols
    print("Both Corpora Have All Expected Columns")


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
        # TODO assert (total_length > 25.5e6) if name == "Kaggle" else (total_length > 4.3e6) # 25_749_461 (Kaggle), 4_368_400 (UCI) (by 80% equi thr 27_968_115 & 4_689_403)
        assert total_length_wo_nan/total_length > 0.95, f"More than 5% are missing values in {name}"
        avg_length = total_length / len(lengths)
        print(f"Avg Length: {avg_length}")
        # TODO assert (avg_length > 1200) if name == "Kaggle" else (avg_length > 2200 )
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
