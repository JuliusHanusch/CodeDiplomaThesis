import logging
from pathlib import Path
from typing import Iterable, Optional

# Include Parent Directory to load packages from
import sys  
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"code").resolve()))  
sys.path.append(str((Path(__file__).parent.parent / "chronos_pkg/src").resolve()))
from src.utils import load_val_data
import datasets
import pandas as pd
from collections import defaultdict
import yaml
import os

if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    corpus_name = "KAGGLE_TEST_CORPUS"
    disk_path = Path("./data/data_sets_raw/Time_Corpus")

    # Read in kaggle test set (Code from load_val_data)
    cache_adr = Path("./cache") / (disk_path.stem + ".parquet")
    # Load -> Explode Table -> Cache
    if not cache_adr.exists():
        data = datasets.load_from_disk(disk_path)
        data = data["test"]
        # Convert into pd format + Drop Unnecessary Features
        exploded_data = [
            {"target": sublist}
            for row in data["value"]
            for sublist in row
        ]
        df = pd.DataFrame(exploded_data)
        df.to_parquet(cache_adr)
    else:
        df = pd.read_parquet(cache_adr)

    # (offset, prediction_length, num_rolls)
    groups_meta = {
        "large_solid": (1024, 64, 16),
        "large": (1024, 64, 8),
        "large_quick": (1024, 64, 4),
        # "large_quick_near": (1024, 8, 8),
        "medium_solid": (512, 32, 16),
        "medium": (512, 32, 8),
        "medium_quick": (512, 32, 4),
        # "medium_quick_near": (512, 8, 4),
        "short_solid": (140, 16, 16),
        "short": (140, 16, 8),
        "short_quick": (140, 16, 4),
    }

    groups = defaultdict(list)

    for target in range(100):
        ts = pd.DataFrame({"target": df.iloc[target]["target"]})
        # Sort them into several groups depending on their length
        for group_name, group_meta in groups_meta.items():
            # Offset is hard constraint --> TS > offset + (prediction_length * num_rolls)
            if len(ts) > group_meta[0] + group_meta[1] * group_meta[2]:
                groups[group_name].append(target)
                break

    results = []
    for group_name, targets in groups.items():
        result = {
            "name": corpus_name + "_" + group_name,
            "disk_path": str(disk_path),
            "targets": targets,
            "offset": -groups_meta[group_name][0],
            "prediction_length": groups_meta[group_name][1],
            "num_rolls": groups_meta[group_name][2],
        }
        results.append(result)

    with open("./cache/eval_config_kaggle_appendix.yml", "w") as f:
        yaml.safe_dump(results, f)

    print("Done! Pls. copy paste the content of ./cache/eval_config_kaggle_appendix.yml to the end of ./data/eval_configs/eval_config.yml and remove all old KAGGLE_TEST_CORPUS entries")