from pathlib import Path
from typing import List, Union
import numpy as np
import datasets
from gluonts.dataset.arrow import ArrowWriter
from tqdm import tqdm  # Import tqdm for progress bar

print("Start Converting Data")

#Mix = datasets.load_dataset("autogluon/chronos_datasets", "training_corpus_tsmixup_10m", streaming=True, split="train") 
Kernel = datasets.load_dataset("autogluon/chronos_datasets", "training_corpus_kernel_synth_1m", streaming=True, split="train").take(10000)



#Mix_time_series = [np.array(data['target']) for data in tqdm(Mix, desc="Processing time series")]
Kernel_time_series = [np.array(data['target']) for data in tqdm(Kernel, desc="Processing time series")]


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

#convert_to_arrow("./training_mix.arrow", time_series=Mix_time_series)
convert_to_arrow("./data/train/kernelsynth.arrow", time_series=Kernel_time_series)


