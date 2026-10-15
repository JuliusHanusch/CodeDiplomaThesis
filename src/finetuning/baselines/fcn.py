import os

# Set before importing TensorFlow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import random
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from gluonts.dataset.arrow import ArrowFile


print("Finished Imports")



DB_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/src/finetuning/tser/final/"
    "tser_bestConfigs_time.db"
)

OUTPUT_DIR = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/Results/Finetuning/TSER/FCN"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/Results/Finetuning/TSER/Predictions/"
)

PREDICTIONS_PATH.mkdir(parents=True, exist_ok=True)

SEED = 42

# FCN hyperparameters
EPOCHS = 150
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 15

VERBOSE = 1



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


set_seed(SEED)



def load_arrow(path):
    dataset = ArrowFile(Path(path))

    X = []
    y = []

    for entry in dataset:
        x = np.asarray(
            entry["target"],
            dtype=np.float32,
        )

        x = np.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if x.ndim == 1:
            # Univariate: (time,) -> (time, 1)
            x = x[:, None]

        elif x.ndim == 2:
            # Multivariate targets are stored as (channels, time).
            # TensorFlow expects (time, channels).
            x = x.T

        else:
            raise ValueError(
                f"Unsupported target shape {x.shape} in {path}"
            )

        label = float(
            entry.get(
                "label",
                entry.get("y", 0.0),
            )
        )

        X.append(x)
        y.append(label)

    if not X:
        raise ValueError(f"Empty dataset: {path}")

    # Requires equal sequence lengths and channel counts.
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )


def rmse(y_true, y_pred):
    return float(
        np.sqrt(np.mean((y_true - y_pred) ** 2))
    )


def mae(y_true, y_pred):
    return float(
        np.mean(np.abs(y_true - y_pred))
    )


class FCNRegressor:
    """
    Fully Convolutional Network for time-series regression.

    Architecture adapted from the FCN implementation in:
    https://github.com/hfawaz/dl-4-tsc

    Input:
        (n_examples, sequence_length, n_channels)

    Output:
        One continuous prediction per example.
    """

    def __init__(
        self,
        input_shape,
        epochs=2000,
        batch_size=16,
        patience=100,
        verbose=1,
    ):
        self.name = "FCN"

        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose

        self.model = None
        self.history = None

        self.train_duration = None
        self.inference_duration = None
        self.best_epoch = None

    def build_model(self):
        input_layer = tf.keras.layers.Input(
            shape=self.input_shape
        )

        x = tf.keras.layers.Conv1D(
            filters=128,
            kernel_size=8,
            padding="same",
        )(input_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.Conv1D(
            filters=256,
            kernel_size=5,
            padding="same",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.Conv1D(
            filters=128,
            kernel_size=3,
            padding="same",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        output_layer = tf.keras.layers.Dense(
            1,
            activation="linear",
        )(x)

        model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=output_layer,
        )

        model.compile(
            loss="mean_squared_error",
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["mae"],
        )

        return model

    def fit(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
    ):
        start_time = time.perf_counter()

        self.model = self.build_model()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
                mode="min",
                verbose=1,
            ),
        ]

        self.history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=self.verbose,
        )

        self.best_epoch = (
            int(np.argmin(self.history.history["val_loss"])) + 1
        )

        self.train_duration = (
            time.perf_counter() - start_time
        )

        print(
            f"[{self.name}] Training completed in "
            f"{self.train_duration:.2f}s"
        )
        print(
            f"[{self.name}] Best validation epoch: "
            f"{self.best_epoch}"
        )

    def predict(self, x):
        start_time = time.perf_counter()

        predictions = self.model.predict(
            x,
            batch_size=self.batch_size,
            verbose=0,
        ).reshape(-1)

        self.inference_duration = (
            time.perf_counter() - start_time
        )

        return predictions



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT DISTINCT dataset, train_data, test_data
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
                "multivariate",
                train_path.parent / "train_multi.arrow",
                train_path.parent / "train_small_multi.arrow",
                train_path.parent / "eval_multi.arrow",
                test_path.parent / "test_multi.arrow",
            ),
        ]

        for version, train_file, train_small_file, val_file, test_file in versions:

            print(f"\nProcessing {dataset} ({version})")

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

            selection_train_size = len(y_train_small)

            print(f"Small train: {X_train_small.shape}")
            print(f"Validation: {X_val.shape}")
            print(f"Test: {X_test.shape}")

            tf.keras.backend.clear_session()
            set_seed(SEED)

            selection_model = FCNRegressor(
                input_shape=X_train_small.shape[1:],
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                patience=EARLY_STOPPING_PATIENCE,
                verbose=VERBOSE,
            )

            selection_model.fit(
                X_train_small,
                y_train_small,
                x_val=X_val,
                y_val=y_val,
            )

            best_epoch = selection_model.best_epoch
            selection_duration = selection_model.train_duration

            print(f"Selected epoch count: {best_epoch}")

            del selection_model
            tf.keras.backend.clear_session()

            X_train_full, y_train_full = load_arrow(train_file)

            if X_train_full.shape[1:] != X_test.shape[1:]:
                raise ValueError(
                    f"Full train/test shape mismatch for "
                    f"{dataset} ({version}): "
                    f"train={X_train_full.shape}, "
                    f"test={X_test.shape}"
                )

            final_train_size = len(y_train_full)

            set_seed(SEED)

            final_model = FCNRegressor(
                input_shape=X_train_full.shape[1:],
                epochs=best_epoch,
                batch_size=BATCH_SIZE,
                patience=EARLY_STOPPING_PATIENCE,
                verbose=VERBOSE,
            )

            final_model.model = final_model.build_model()

            start_time = time.perf_counter()

            final_model.model.fit(
                X_train_full,
                y_train_full,
                epochs=best_epoch,
                batch_size=BATCH_SIZE,
                shuffle=True,
                verbose=VERBOSE,
            )

            final_model.train_duration = time.perf_counter() - start_time
            final_model.best_epoch = best_epoch

            print(
                f"Final training completed in "
                f"{final_model.train_duration:.2f}s"
            )

            del X_train_small, y_train_small, X_val, y_val
            tf.keras.backend.clear_session()

            predictions = final_model.predict(X_test)

            test_rmse = rmse(y_test, predictions)
            test_mae = mae(y_test, predictions)

            print(
                f"Selected epochs = {best_epoch}, "
                f"test RMSE = {test_rmse:.6f}, "
                f"test MAE = {test_mae:.6f}"
            )

            result = {
                "dataset": dataset,
                "version": version,
                "method": "FCN",
                "rmse": test_rmse,
                "mae": test_mae,
                "train_duration": final_model.train_duration,
                "inference_time": final_model.inference_duration,
            }

            output_file = OUTPUT_DIR / f"{dataset}.csv"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Preserve any existing results for other methods/versions.
            if output_file.exists():
                existing = pd.read_csv(output_file)
                existing = existing[
                    ~(
                        (existing["version"] == version)
                        & (existing["method"] == "FCN")
                    )
                ]
                output_df = pd.concat(
                    [existing, pd.DataFrame([result])],
                    ignore_index=True,
                )
            else:
                output_df = pd.DataFrame([result])

            output_df.to_csv(output_file, index=False)

            print(f"Saved results to {output_file}")

            tf.keras.backend.clear_session()

