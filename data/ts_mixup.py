import numpy as np
from typing import List
import torch
from collections import defaultdict

class distribution_generator():
    """
    Torch is much (25x) faster when generating Dirichlets distributions in bulk than just one at a time
    This class creates several in bulk and caches them, to keep it from becomming a bottleneck
    If there are too many different configs it switches back to one at a time as pregeneration would be too inefficient with too many misses
    """
    def __init__(self):
        self.cache = defaultdict(dict) # k -> alphas -> list of pregenerated distributions
        self.alphas = set([])
        
    def pregenerate(self, n, alpha, k):
        list_of_distribution = torch.distributions.Dirichlet(torch.full((k,), alpha)).sample([n])
        self.cache[k][alpha] = list_of_distribution.tolist()

    def draw(self, alpha: float, k: int):
        if alpha not in self.alphas and len(self.alphas) >= 10:
            # caching doesnt work when alphas not discret
            return torch.distributions.Dirichlet(torch.full((k,), alpha)).sample([1])[0]
        if k not in self.cache or alpha not in self.cache[k] or len(self.cache[k][alpha]) == 0:
                self.pregenerate(n=100_000, alpha=alpha, k=k)
                self.alphas.add(alpha)
        return self.cache[k][alpha].pop()
            
DISTRIBUTION_GENERATOR = distribution_generator()


def ts_mixup(datasets: List[np.ndarray], alpha: np.float64 = 1.5) -> np.ndarray:
    print("3.1", flush=True)
    ### INIT AND SANITY CHECKS ###
    k = len(datasets)
    if k < 1:
        print("3.1.1", flush=True)
        raise Exception(ts_mixup.__name__, 'Received empty or invalid datasets as input')

    l = len(datasets[0])
    print("3.2", flush=True)
    for dataset in datasets:
        print("3.3", flush=True)
        if l != len(dataset):
            print("3.3.1", flush=True)
            raise Exception(ts_mixup.__name__, f'Datasets differ in length {[len(d) for d in datasets]}')

    ### IMPLEMENTATION ###
    print("3.4", flush=True)
    # lambdas: np.ndarray = np.random.dirichlet([alpha] * k)  # weights for each dataset
    # lambdas = torch.distributions.Dirichlet(torch.full((k,), alpha)).sample([1])[0]
    lambdas = DISTRIBUTION_GENERATOR.draw(alpha=alpha, k=k)
    means: np.ndarray = np.array([np.mean(np.abs(dataset)) for dataset in datasets])  # means of each dataset
    means[means == 0] = 1  # prevent division by zero

    datasets: List[np.ndarray] = [dataset / mean for dataset, mean in zip(datasets, means)]
    mixed_ts: np.ndarray = np.array(
        [sum(lambdas[ds_idx] * datasets[ds_idx][row_idx]
        for ds_idx in range(len(datasets)))
        for row_idx in range(len(datasets[0]))])
    print("3.5", flush=True)

    return mixed_ts
