print("Start")

import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import time

import numpy as np
from numba import njit, prange

from gluonts.dataset.arrow import ArrowFile

print("Finished Imports")

name = "Rocket"

# =========================================================
# CONFIG
# =========================================================

DB_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/src/finetuning/tser/tser.db"
)

OUTPUT_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/Results/Finetuning/TSER/rocket_results.csv"
)


np.random.seed(42)


def load_arrow(path):

    dataset = ArrowFile(Path(path))

    X = []
    y = []

    for entry in dataset:

        x = np.asarray(
            entry["target"],
            dtype=np.float32
        )

        x = np.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )


        if x.ndim == 1:

            x = x[:, None]


        elif x.ndim == 2:

            x = x.T


        else:
            raise ValueError(
                f"Unsupported shape {x.shape}"
            )


        label = float(
            entry.get(
                "label",
                entry.get("y",0.0)
            )
        )

        X.append(x)
        y.append(label)


    return (
        np.asarray(X,dtype=np.float32),
        np.asarray(y,dtype=np.float32)
    )


@njit(fastmath=True)
def generate_kernels(input_length, num_kernels, num_channels=1):
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    candidate_lengths = candidate_lengths[candidate_lengths < input_length]
    lengths = np.random.choice(candidate_lengths, num_kernels)

    # exponential
    num_channel_indices = (2 ** np.random.uniform(0, np.log2(num_channels + 1), num_kernels)).astype(np.int32)
    channel_indices = np.zeros(num_channel_indices.sum(), dtype=np.int32)

    weights = np.zeros((num_channels, lengths.sum()), dtype=np.float32)
    biases = np.zeros(num_kernels, dtype=np.float32)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    for i in range(num_kernels):

        _weights = np.empty((num_channels, lengths[i]), dtype=np.float32)

        for j in range(num_channels):
            for k in range(lengths[i]):
                _weights[j, k] = np.random.normal()

        a = lengths[:i].sum()
        b = a + lengths[i]
        for j in range(num_channels):
            _weights[j] = _weights[j] - _weights[j].mean()
        weights[:, a:b] = _weights

        a1 = num_channel_indices[:i].sum()
        b1 = a1 + num_channel_indices[i]
        channel_indices[a1:b1] = np.random.choice(np.arange(0, num_channels), num_channel_indices[i], replace=False)

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) // (lengths[i] - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((lengths[i] - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

    return weights, lengths, biases, dilations, paddings, num_channel_indices, channel_indices



@njit(fastmath=True)
def apply_kernel(X, weights, length, bias, dilation, padding, num_channel_indices, channel_indices, stride):
    # zero padding
    if padding > 0:
        _input_length, _num_channels = X.shape
        _X = np.zeros(
            (_input_length + (2 * padding), _num_channels),
            dtype=X.dtype,
        )
        _X[padding:(padding + _input_length), :] = X
        X = _X

    input_length, num_channels = X.shape

    output_length = input_length - ((length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

    for i in range(0, output_length, stride):
        _sum = bias

        for j in range(length):
            for k in range(num_channel_indices):
                _sum += weights[channel_indices[k], j] * X[i + (j * dilation), channel_indices[k]]

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return _ppv / output_length, _max


@njit(parallel=True, fastmath=True)
def apply_kernels(X, kernels, stride=1):
    weights, lengths, biases, dilations, paddings, num_channel_indices, channel_indices = kernels

    num_examples = len(X)
    num_kernels = len(lengths)

    _X = np.zeros((num_examples, num_kernels * 2), dtype=np.float32)  # 2 features per kernel

    for i in prange(num_examples):
        a = 0
        a1 = 0
        for j in range(num_kernels):
            b = a + lengths[j]
            b1 = a1 + num_channel_indices[j]

            _X[i, (j * 2):((j * 2) + 2)] = \
                apply_kernel(X[i], weights[:, a:b], lengths[j], biases[j], dilations[j], paddings[j],
                             num_channel_indices[j], channel_indices[a1:b1], stride)

            a = b
            a1 = b1

    return _X

class TimeSeriesRegressor:
    """
    This is a super class for time series regressors
    """

    def __init__(self,
                 output_directory: str):
        """
        Initialise the regression model
        """
        self.output_directory = output_directory
        self.train_duration = None
        self.name = "TimeSeriesRegressor"
        pass

    def fit(self,
            x_train: np.array,
            y_train: np.array,
            x_val: np.array = None,
            y_val: np.array = None):
        """
        Fit the regression model
        """
        pass

    def predict(self, x: np.array):
        """
        Do prediction using the regression model on x
        """
        pass

class RocketRegressor(TimeSeriesRegressor):
    """
    This is a class implementing Rocket for time series regression.
    The code is adapted by the authors from the original Rocket implementation at https://github.com/angus924/rocket
    """

    def __init__(self,
                 output_directory: str,
                 n_kernels: int = 10000):
        """
        Initialise the Rocket model

        Inputs:
            output_directory: path to store results/models
            n_kernels: number of random kernels
        """
        super().__init__(output_directory)
        print('[{}] Creating Regressor'.format(self.name))
        self.name = name
        self.n_kernels = n_kernels
        self.kernels = None
        self.regressor = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-3,3,10))
        )

    def fit(self,
            x_train: np.array,
            y_train: np.array,
            x_val: np.array = None,
            y_val: np.array = None):
        """
        Fit Rocket

        Inputs:
            x_train: training data (num_examples, num_timestep, num_channels)
            y_train: training target
            x_val: validation data (num_examples, num_timestep, num_channels)
            y_val: validation target
        """
        start_time = time.perf_counter()
        print('[{}] Generating kernels'.format(self.name))
        self.kernels = generate_kernels(x_train.shape[1], self.n_kernels, x_train.shape[2])
        print('[{}] Applying kernels'.format(self.name))
        x_training_transform = apply_kernels(x_train, self.kernels)

        print('[{}] Training'.format(self.name))
        self.regressor.fit(x_training_transform, y_train)
        self.train_duration = time.perf_counter() - start_time


        print('[{}] Training done!, took {}s'.format(self.name, self.train_duration))

    def predict(self, x: np.array):
        """
        Do prediction with Rocket

        Inputs:
            x: data for prediction (num_examples, num_timestep, num_channels)
        Outputs:
            y_pred: prediction
        """
        print('[{}] Predicting'.format(self.name))
        start_time = time.perf_counter()
        print('[{}] Applying kernels'.format(self.name))
        x_test_transform = apply_kernels(x, self.kernels)
        y_pred = self.regressor.predict(x_test_transform)

        test_duration = time.perf_counter() - start_time

        print("[{}] Predicting completed, took {}s".format(self.name, test_duration))

        return y_pred

def rmse(y_true,y_pred):

    return np.sqrt(
        np.mean(
            (y_true-y_pred)**2
        )
    )


def mae(y_true,y_pred):

    return np.mean(
        np.abs(
            y_true-y_pred
        )
    )



def get_runs():

    conn = sqlite3.connect(DB_PATH)

    cur = conn.cursor()

    cur.execute(
        """
        SELECT DISTINCT dataset, train_data, test_data
        FROM runs
        """
    )

    rows = cur.fetchall()

    conn.close()

    return rows


if __name__ == "__main__":

    N_RUNS = 20
    results = []

    for dataset, train_path, test_path in get_runs():

        if dataset == "PPGDalia":
            print(f"Skipping {dataset}")
            continue


        train_path = Path(train_path)
        test_path = Path(test_path)

        versions = [
            (
                "univariate",
                train_path.parent / "train_rocket.arrow",
                test_path.parent / "test_rocket.arrow",
            ),
            (
                "multivariate",
                train_path.parent / "train_full_rocket.arrow",
                test_path.parent / "test_full_rocket.arrow",
            ),
        ]

        for version, train_file, test_file in versions:

            if not train_file.exists():
                continue

            if not test_file.exists():
                continue

            print(f"\nProcessing {dataset} ({version})")

            X_train, y_train = load_arrow(train_file)
            X_test, y_test = load_arrow(test_file)

            print("Train shape:", X_train.shape)

            runs = N_RUNS

            best_rmse = np.inf
            best_mae = np.inf
            best_seed = None

            for seed in range(runs):

                print(f"\nRun {seed + 1}/{runs}")

                np.random.seed(seed)

                model = RocketRegressor(
                    output_directory="/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/Results/Finetuning/TSER",
                    n_kernels=10000,
                )

                model.fit(
                    X_train,
                    y_train,
                )

                pred = model.predict(X_test)

                current_rmse = rmse(y_test, pred)
                current_mae = mae(y_test, pred)

                print(
                    f"RMSE = {current_rmse:.6f}, "
                    f"MAE = {current_mae:.6f}"
                )

                if current_rmse < best_rmse:
                    best_rmse = current_rmse
                    best_mae = current_mae
                    best_seed = seed

            result = {
                "dataset": dataset,
                "version": version,
                "best_seed": best_seed,
                "rmse": best_rmse,
                "mae": best_mae,
            }

            results.append(result)
            print("\nBest:", result)

    pd.DataFrame(results).to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nFinished")