from pathlib import Path
from typing import List, Union
import numpy as np
import datasets
from gluonts.dataset.arrow import ArrowWriter
from tqdm import tqdm  # Import tqdm for progress bar

print("Start Converting Data")

# Step 1: Load the dataset from Hugging Face
ts_mix = datasets.load_dataset("autogluon/chronos_datasets", "training_corpus_tsmixup_10m", streaming=True, split="train") # oder "training_corpus_tsmixup_10m"
kernel = datasets.load_dataset("autogluon/chronos_datasets", "training_corpus_kernel_synth_1m", streaming=True, split="train") # oder "training_corpus_tsmixup_10m"




# Step 2: Extract time-series data with progress bar
# Using tqdm to display the progress as we loop through the dataset
ts_mix_series = [np.array(data['target']) for data in tqdm(ts_mix, desc="Processing time series")]
kernel_series = [np.array(data['target']) for data in tqdm(kernel, desc="Processing time series")]

# Step 3: Define the convert_to_arrow function
def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    compression: str = "lz4",
):
    assert isinstance(time_series, list) or (
        isinstance(time_series, np.ndarray) and
        time_series.ndim == 2
    )
    start = np.datetime64("2000-01-01 00:00", "s")
    dataset = [{"start": start, "target": ts} for ts in tqdm(time_series, desc="Converting to Arrow")]
    ArrowWriter(compression=compression).write_to_file(dataset, path=path)

# Step 4: Convert the time-series data to Arrow format
convert_to_arrow("/content/CodeDiplomaThesis/data/train/training_mix.arrow", time_series=ts_mix_series)
convert_to_arrow("/content/CodeDiplomaThesis/data/train/kernelsynth.arrow", time_series=kernel_series)

print("Dataset successfully converted and stored in 'kernelsynth.arrow'")