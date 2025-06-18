import numpy as np
from typing import List

def ts_mixup(datasets: List[np.ndarray], alpha: np.float64 = 1.5) -> np.ndarray:
    ### INIT AND SANITY CHECKS ###
    k = len(datasets)
    if k < 1:
        raise Exception(ts_mixup.__name__, 'Received empty or invalid datasets as input')

    l = len(datasets[0])
    for dataset in datasets:
        if l != len(dataset):
            raise Exception(ts_mixup.__name__, 'Datasets differ in length')

    ### IMPLEMENTATION ###
    lambdas: np.ndarray = np.random.dirichlet([alpha] * k)  # weights for each dataset
    means: np.ndarray = np.array([np.mean(np.abs(dataset)) for dataset in datasets])  # means of each dataset
    means[means == 0] = 1e-5  # prevent division by zero

    datasets: List[np.ndarray] = [dataset / mean for dataset, mean in zip(datasets, means)]
    mixed_ts: np.ndarray = np.array(
        [sum(lambdas[ds_idx] * datasets[ds_idx][row_idx]
        for ds_idx in range(len(datasets)))
        for row_idx in range(len(datasets[0]))])

    return mixed_ts
