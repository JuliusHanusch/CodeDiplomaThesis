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

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)

    kaggle_corpus = ds.load_from_disk(data_directory / "Time_Corpus_Processed")["train"]
    uci_corpus = ds.load_from_disk(data_directory / "UCI_Corpus_Processed")["train"]
    print(kaggle_corpus)
    print(uci_corpus)

    corpora = [("Kaggle", kaggle_corpus), ("UCI", uci_corpus)]

    # Check Total Length (Fuzzy because Data is not 100% stable due to legal constraints)
    assert 20_000 > len(kaggle_corpus) >= 19_000, f"Only {len(kaggle_corpus)} of 19_507 original  TS in Time Corpus some seem to have gone missing, small changes are expected though as data is not completly stable"
    assert 1_900 > len(uci_corpus) >= 1_800, f"Only {len(uci_corpus)} of 1_857 original TS in UCI Corpus some seem to have gone missing, small changes are expected though as data is not completly stable" # 2_328
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
        print(f"\n{name}:")
        for ts in tqdm(corpus):
            assert isinstance(ts["target"], list) and not any(isinstance(entry, list) for entry in ts)
    print("All Corpora Are Univariate")

    # Check for Min Length
    for name, corpus in corpora:
        print(f"\n{name}:")
        lengths = []
        for ts in tqdm(corpus):
            length = len(ts["target"])
            assert length >= 64 # The Min Length we set during Preprocessing
            lengths.append(length)
        total_length = sum(lengths)
        print(f"Total Length: {total_length}")
        assert (total_length > 25.5e6) if name == "Kaggle" else (total_length > 4.3e6) # 25_749_461 (Kaggle), 4_368_400 (UCI)
        avg_length = total_length / len(lengths)
        print(f"Avg Length: {avg_length}")
        assert (avg_length > 1200) if name == "Kaggle" else (avg_length > 2200 )
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
            assert eval_frequency(pd.DataFrame(ts), time_column="timestamp")
        print("All TS Seem Equidistant")   
