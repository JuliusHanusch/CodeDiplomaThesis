import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import argparse
import random
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from gluonts.dataset.arrow import ArrowFile


print("Finished Imports")




DB_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/src/finetuning/tser/final/"
    "tser_bestConfigs_time.db"
)

OUTPUT_DIR = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/Results/Finetuning/TSER/Inception"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/Results/Finetuning/TSER/Predictions/"
)

PREDICTIONS_PATH.mkdir(parents=True, exist_ok=True)

SEED = 42

# InceptionTime hyperparameters
EPOCHS = 40
BATCH_SIZE = 32
NB_FILTERS = 32
DEPTH = 6
KERNEL_SIZE = 41
USE_RESIDUAL = True
USE_BOTTLENECK = True
BOTTLENECK_SIZE = 32
EARLY_STOPPING_PATIENCE = 10

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
            # Multivariate targets stored as (channels, time)
            # -> TensorFlow format (time, channels).
            x = x.T

        else:
            raise ValueError(
                f"Unsupported target shape {x.shape} "
                f"in {path}"
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

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )



def rmse(y_true, y_pred):

    return np.sqrt(
        np.mean((y_true - y_pred) ** 2)
    )


def mae(y_true, y_pred):

    return np.mean(
        np.abs(y_true - y_pred)
    )



class InceptionTimeRegressor:
    """
    Single InceptionTime model for time-series regression.

    Architecture adapted from the InceptionTime implementation
    by Hassan Ismail Fawaz et al.

    Official repository:
    https://github.com/hfawaz/InceptionTime

    Input shape:
        (n_examples, sequence_length, n_channels)

    Output:
        One continuous prediction per example.
    """

    def __init__(
        self,
        input_shape,
        epochs=1500,
        batch_size=64,
        nb_filters=32,
        use_residual=True,
        use_bottleneck=True,
        depth=6,
        kernel_size=41,
        bottleneck_size=32,
        patience=100,
        verbose=1,
    ):

        self.name = "InceptionTime"

        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth

        self.kernel_size = kernel_size - 1
        self.bottleneck_size = bottleneck_size

        self.patience = patience
        self.verbose = verbose

        self.model = None
        self.history = None

        self.train_duration = None
        self.inference_duration = None
        self.best_epoch = None

    def _inception_module(
        self,
        input_tensor,
        activation="linear",
    ):

        input_channels = int(input_tensor.shape[-1])

        if self.use_bottleneck and input_channels > 1:

            input_inception = tf.keras.layers.Conv1D(
                filters=self.bottleneck_size,
                kernel_size=1,
                padding="same",
                activation=activation,
                use_bias=False,
            )(input_tensor)

        else:
            input_inception = input_tensor

        kernel_sizes = [
            self.kernel_size // (2 ** i)
            for i in range(3)
        ]

        conv_list = []

        for kernel_size in kernel_sizes:

            conv = tf.keras.layers.Conv1D(
                filters=self.nb_filters,
                kernel_size=kernel_size,
                strides=1,
                padding="same",
                activation=activation,
                use_bias=False,
            )(input_inception)

            conv_list.append(conv)

        max_pool = tf.keras.layers.MaxPool1D(
            pool_size=3,
            strides=1,
            padding="same",
        )(input_tensor)

        conv_pool = tf.keras.layers.Conv1D(
            filters=self.nb_filters,
            kernel_size=1,
            padding="same",
            activation=activation,
            use_bias=False,
        )(max_pool)

        conv_list.append(conv_pool)

        x = tf.keras.layers.Concatenate(axis=2)(
            conv_list
        )

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)

        return x

    def _shortcut_layer(
        self,
        input_tensor,
        out_tensor,
    ):

        shortcut = tf.keras.layers.Conv1D(
            filters=int(out_tensor.shape[-1]),
            kernel_size=1,
            padding="same",
            use_bias=False,
        )(input_tensor)

        shortcut = tf.keras.layers.BatchNormalization()(
            shortcut
        )

        x = tf.keras.layers.Add()(
            [shortcut, out_tensor]
        )

        x = tf.keras.layers.Activation("relu")(x)

        return x

    def build_model(self):

        input_layer = tf.keras.layers.Input(
            shape=self.input_shape
        )

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:

                x = self._shortcut_layer(
                    input_res,
                    x,
                )

                input_res = x

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

            print(f"Small train: {X_train_small.shape}")
            print(f"Validation: {X_val.shape}")
            print(f"Test: {X_test.shape}")


            tf.keras.backend.clear_session()
            set_seed(SEED)

            selection_model = InceptionTimeRegressor(
                input_shape=X_train_small.shape[1:],
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                nb_filters=NB_FILTERS,
                use_residual=USE_RESIDUAL,
                use_bottleneck=USE_BOTTLENECK,
                depth=DEPTH,
                kernel_size=KERNEL_SIZE,
                bottleneck_size=BOTTLENECK_SIZE,
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

            print(f"Selected epoch count: {best_epoch}")

            selection_duration = selection_model.train_duration

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

            set_seed(SEED)

            final_model = InceptionTimeRegressor(
                input_shape=X_train_full.shape[1:],
                epochs=best_epoch,
                batch_size=BATCH_SIZE,
                nb_filters=NB_FILTERS,
                use_residual=USE_RESIDUAL,
                use_bottleneck=USE_BOTTLENECK,
                depth=DEPTH,
                kernel_size=KERNEL_SIZE,
                bottleneck_size=BOTTLENECK_SIZE,
                patience=EARLY_STOPPING_PATIENCE,
                verbose=VERBOSE,
            )

            start_time = time.perf_counter()

            final_model.model = final_model.build_model()

            final_model.model.fit(
                X_train_full,
                y_train_full,
                epochs=best_epoch,
                batch_size=BATCH_SIZE,
                shuffle=True,
                verbose=VERBOSE,
            )

            final_model.train_duration = (
                time.perf_counter() - start_time
            )

            final_model.best_epoch = best_epoch

            print(
                f"Final training completed in "
                f"{final_model.train_duration:.2f}s "
                f"using {len(y_train_full)} training examples"
            )

            del X_train_small, y_train_small, X_val, y_val
            tf.keras.backend.clear_session()


            pred = final_model.predict(X_test)

            test_rmse = rmse(y_test, pred)
            test_mae = mae(y_test, pred)

            print(
                f"Selected epochs = {best_epoch}, "
                f"test RMSE = {test_rmse:.6f}, "
                f"test MAE = {test_mae:.6f}"
            )

            result = {
                "dataset": dataset,
                "version": version,
                "method": "InceptionTime",
                "rmse": test_rmse,
                "mae": test_mae,
                "best_epoch": best_epoch,
                "train_duration": final_model.train_duration,
                "inference_time": final_model.inference_duration,
            }

            output_file = OUTPUT_DIR / f"{dataset}.csv"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if output_file.exists():
                existing = pd.read_csv(output_file)
                existing = existing[
                    ~(
                        (existing["version"] == version)
                        & (existing["method"] == "InceptionTime")
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

