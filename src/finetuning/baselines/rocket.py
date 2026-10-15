import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
import argparse

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import time
from numba import njit, prange

import numpy as np
from gluonts.dataset.arrow import ArrowFile

print("Finished Imports")

name = "Rocket"


DB_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/tser/final/tser_bestConfigs_time.db"
)

OUTPUT_DIR = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/Results/Finetuning/TSER/ROCKET"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/Results/Finetuning/TSER/Predictions/"
)
PREDICTIONS_PATH.mkdir(parents=True, exist_ok=True)


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

    def __init__(
        self,
        output_directory: str,
        n_kernels: int = 10000,
        alphas=None,
    ):
        super().__init__(output_directory)

        self.name = "Rocket"
        self.n_kernels = n_kernels
        self.alphas = (
            [0.01, 0.1, 1.0, 10.0, 100.0]
            if alphas is None else alphas
        )

        self.kernels = None
        self.best_alpha = None
        self.scaler = StandardScaler()
        self.regressor = None

        self.train_duration = None
        self.inference_duration = None

    def fit(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
    ):
        start_time = time.perf_counter()

        print(f"[{self.name}] Generating kernels")

        self.kernels = generate_kernels(
            x_train.shape[1],
            self.n_kernels,
            x_train.shape[2],
        )

        print(f"[{self.name}] Transforming training data")

        X_train_features = apply_kernels(
            x_train, self.kernels
        )

        # Standard ROCKET: fixed alpha = 1.0.
        if x_val is None or y_val is None:
            self.best_alpha = 1.0

        else:
            print(f"[{self.name}] Transforming validation data")

            X_val_features = apply_kernels(
                x_val, self.kernels
            )

            # Fit scaling using training data only.
            X_train_scaled = self.scaler.fit_transform(
                X_train_features
            )
            X_val_scaled = self.scaler.transform(
                X_val_features
            )

            best_val_rmse = np.inf

            print(f"[{self.name}] Tuning Ridge alpha")

            for alpha in self.alphas:
                ridge = Ridge(
                    alpha=alpha,
                    solver="lsqr",
                )

                ridge.fit(X_train_scaled, y_train)

                val_pred = ridge.predict(X_val_scaled)
                val_rmse = rmse(y_val, val_pred)

                print(
                    f"alpha={alpha}: "
                    f"validation RMSE={val_rmse:.6f}"
                )

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    self.best_alpha = alpha

            print(
                f"[{self.name}] Best alpha="
                f"{self.best_alpha}, "
                f"validation RMSE={best_val_rmse:.6f}"
            )

        # Fit the final Ridge model on the complete supplied
        # training set using the selected alpha.
        X_train_scaled = self.scaler.fit_transform(
            X_train_features
        )

        self.regressor = Ridge(
            alpha=self.best_alpha,
            solver="lsqr",
        )

        self.regressor.fit(X_train_scaled, y_train)

        self.train_duration = time.perf_counter() - start_time

        print(
            f"[{self.name}] Training completed in "
            f"{self.train_duration:.2f}s"
        )

    def predict(self, x):
        start_time = time.perf_counter()

        X_features = apply_kernels(x, self.kernels)
        X_scaled = self.scaler.transform(X_features)

        predictions = self.regressor.predict(X_scaled)

        self.inference_duration = (
            time.perf_counter() - start_time
        )

        return predictions

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT dataset, train_data, test_data
        FROM runs
        WHERE id = ?
        """,
        (args.index,),
    )

    runs = cur.fetchall()
    conn.close()

    if not runs:
        raise ValueError(f"No run found for index {args.index}")

    for dataset, train_path, test_path in runs:

        train_path = Path(train_path)
        test_path = Path(test_path)

        versions = [
            (
                "univariate",
                train_path.parent / "train_uni.arrow",
                train_path.parent / "train_small_uni.arrow",
                train_path.parent / "eval_uni.arrow",
                test_path.parent / "test_uni.arrow",
            ),
            (
                "multivariate",
                train_path.parent / "train_multi.arrow",
                train_path.parent / "train_small_multi.arrow",
                train_path.parent / "eval_multi.arrow",
                test_path.parent / "test_multi.arrow",
            ),
        ]

        for version, train_file, train_small_file, val_file, test_file in versions:

            # -------------------------------------------------
            # 1. Load small training split, validation, and test
            # -------------------------------------------------
            X_train_small, y_train_small = load_arrow(train_small_file)
            X_val, y_val = load_arrow(val_file)
            X_test, y_test = load_arrow(test_file)

            if (
                X_train_small.shape[1:] != X_val.shape[1:]
                or X_train_small.shape[1:] != X_test.shape[1:]
            ):
                raise ValueError(
                    f"Incompatible shapes for {dataset} ({version}): "
                    f"train_small={X_train_small.shape}, "
                    f"val={X_val.shape}, test={X_test.shape}"
                )

            # -------------------------------------------------
            # 2. Select alpha using validation data
            # -------------------------------------------------
            np.random.seed(42)

            selector = RocketRegressor(
                output_directory=str(PREDICTIONS_PATH),
                n_kernels=10000,
                alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
            )

            selector.fit(
                X_train_small,
                y_train_small,
                x_val=X_val,
                y_val=y_val,
            )

            best_alpha = selector.best_alpha

            print(f"Selected alpha = {best_alpha}")

            # -------------------------------------------------
            # 3. Refit using the complete training set
            # -------------------------------------------------
            X_train_full, y_train_full = load_arrow(train_file)

            if X_train_full.shape[1:] != X_test.shape[1:]:
                raise ValueError(
                    f"Full training/test shape mismatch for "
                    f"{dataset} ({version}): "
                    f"train={X_train_full.shape}, test={X_test.shape}"
                )

            # Use the same random seed for reproducibility.
            np.random.seed(42)

            final_model = RocketRegressor(
                output_directory=str(PREDICTIONS_PATH),
                n_kernels=10000,
                alphas=[best_alpha],
            )

            final_model.fit(
                X_train_full,
                y_train_full,
            )

            # -------------------------------------------------
            # 4. Evaluate on the held-out test set
            # -------------------------------------------------
            pred = final_model.predict(X_test)

            rocket_rmse = rmse(y_test, pred)
            rocket_mae = mae(y_test, pred)

            print(
                f"Best alpha = {best_alpha}, "
                f"test RMSE = {rocket_rmse:.6f}, "
                f"test MAE = {rocket_mae:.6f}"
            )

            result = {
                "dataset": dataset,
                "version": version,
                "method": "ROCKET Optimized",
                "best_alpha": best_alpha,
                "rmse": rocket_rmse,
                "mae": rocket_mae,
                "selection_train_size": len(y_train_small),
                "final_train_size": len(y_train_full),
                "validation_size": len(y_val),
                "test_size": len(y_test),
                "train_duration": final_model.train_duration,
                "inference_time": final_model.inference_duration,
            }

            output_file = OUTPUT_DIR / f"{dataset}.csv"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Preserve results from the other version of this dataset.
            if output_file.exists():
                existing = pd.read_csv(output_file)
                existing = existing[
                    ~(
                        (existing["version"] == version)
                        & (existing["method"] == "ROCKET Optimized")
                    )
                ]
                results_df = pd.concat(
                    [existing, pd.DataFrame([result])],
                    ignore_index=True,
                )
            else:
                results_df = pd.DataFrame([result])

            results_df.to_csv(output_file, index=False)

            print(f"Saved results to {output_file}")



