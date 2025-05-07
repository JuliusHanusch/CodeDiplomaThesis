import numpy as np

def ts_mixup(datasets, alpha=1.5):

    ### INIT AND SANITY CHECKS ###
    FUNC_NAME = 'ts_mixup'
    k = len(datasets)
    if k < 1:
        raise Exception(FUNC_NAME, 'Received empty or invalid datasets as input')

    l = len(datasets[0])
    for dataset in datasets:
        if l!=len(dataset):
            raise Exception(FUNC_NAME, 'Datasets differ in length')

    ### IMPLEMENTATION ###
    lambdas = np.random.dirichlet([alpha] * k)                              # weights for each dataset
    means = np.array([np.mean(np.abs(dataset)) for dataset in datasets])    # means of each dataset
    means[means==0] = 1e-5                                                  # prevent division by zero

    datasets = [dataset / mean for dataset, mean in zip(datasets, means)]
    mixed_ts = [sum(lambdas[ds_idx] * datasets[ds_idx][row_idx] for ds_idx in range(len(datasets))) for row_idx in range(len(datasets[0]))]

    return mixed_ts