from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import argparse

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
import random

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

from gluonts.dataset.arrow import ArrowFile
from xgboost import XGBRegressor


DB_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/src/finetuning/tser/final/tser_bestConfigs_time.db"
)

OUTPUT_CSV = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/Results/Finetuning/TSER/tser_baselines_optimized.csv"
)

RANDOM_STATE = 42


knn_param_grid = {
    "scaler": [
        StandardScaler(),
        RobustScaler(),
        "passthrough",
    ],
    "model__n_neighbors": [1, 2, 3, 4, 5],
    "model__metric": [
        "euclidean",
        "manhattan",
    ],
    "model__weights": [
        "uniform",
        "distance",
    ],
}

def rmse(preds, labels):
    preds = np.asarray(preds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return np.sqrt(np.mean((preds - labels) ** 2))


def mae(preds, labels):
    preds = np.asarray(preds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return np.mean(np.abs(preds - labels))



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


def create_features(X):
    """
    Convert univariate or multivariate time series into
    2D feature vectors for scikit-learn and XGBoost.

    Supported shapes per sample:
        Univariate:   (time_steps,)
        Multivariate: (channels, time_steps)
                      or (time_steps, channels)

    The channel/time axis convention must be consistent
    across all samples in a dataset.
    """
    X_feat = []
    valid_indices = []
    expected_shape = None

    for i, series in enumerate(X):
        series = np.asarray(series, dtype=np.float32)

        if series.size == 0:
            continue

        if series.ndim not in (1, 2):
            raise ValueError(
                f"Sample {i} has unsupported shape "
                f"{series.shape}. Expected 1D or 2D data."
            )

        if not np.all(np.isfinite(series)):
            continue

        if expected_shape is None:
            expected_shape = series.shape
        elif series.shape != expected_shape:
            raise ValueError(
                f"Inconsistent sample shapes: sample 0 has "
                f"shape {expected_shape}, but sample {i} "
                f"has shape {series.shape}."
            )

        X_feat.append(series.reshape(-1))
        valid_indices.append(i)

    if not X_feat:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.array([], dtype=np.int64),
        )

    return (
        np.stack(X_feat).astype(np.float32),
        np.asarray(valid_indices, dtype=np.int64),
    )




def load_xy(path):
    X, y = load_arrow(path)
    X_feat, indices = create_features(X)

    y_target = y[indices]
    valid = np.isfinite(y_target)

    return X_feat[valid], y_target[valid]



def tune_knn(
    train_small_path,
    val_path,
):
    X_train, y_train = load_xy(
        train_small_path
    )
    X_val, y_val = load_xy(
        val_path
    )

    if len(X_train) == 0 or len(X_val) == 0:
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {y_val.shape}")
        raise ValueError("KNN training or validation set is empty.")


    param_grid = {
        **knn_param_grid,
        "model__n_neighbors": [
            k for k in [1, 2, 3, 4, 5]
            if k <= len(X_train)
        ],
    }

    best_params = None
    best_rmse = float("inf")

    for params in ParameterGrid(param_grid):
        pipeline = Pipeline([
            ("scaler", params["scaler"]),
            (
                "model",
                KNeighborsRegressor(
                    n_neighbors=params["model__n_neighbors"],
                    metric=params["model__metric"],
                    weights=params["model__weights"],
                    n_jobs=-1,
                ),
            ),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_val)
        val_rmse = rmse(predictions, y_val)

        if val_rmse < best_rmse:
            best_rmse = val_rmse

            scaler = params["scaler"]

            best_params = {
                "scaler": (
                    "StandardScaler"
                    if isinstance(scaler, StandardScaler)
                    else "RobustScaler"
                    if isinstance(scaler, RobustScaler)
                    else "passthrough"
                ),
                "n_neighbors": params["model__n_neighbors"],
                "metric": params["model__metric"],
                "weights": params["model__weights"],
            }

    return best_params, best_rmse


def fit_final_knn(
    train_path,
    params,
):
    X_train, y_train = load_xy(
        train_path
    )

    if params["scaler"] == "StandardScaler":
        scaler = StandardScaler()
    elif params["scaler"] == "RobustScaler":
        scaler = RobustScaler()
    else:
        scaler = "passthrough"

    model = Pipeline([
        ("scaler", scaler),
        (
            "model",
            KNeighborsRegressor(
                n_neighbors=params["n_neighbors"],
                metric=params["metric"],
                weights=params["weights"],
                n_jobs=-1,
            ),
        ),
    ])

    model.fit(X_train, y_train)
    return model


RF_PARAM_GRID = {
        "n_estimators": [10, 20, 50, 100],
        "max_depth": [2, 3, 4, 5],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.5, 1.0],
    }


def tune_random_forest(
    train_small_path,
    val_path,
    random_state=42,
):
    X_train, y_train = load_xy(
        train_small_path
    )
    X_val, y_val = load_xy(
        val_path
    )

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("RF training or validation set is empty.")

    best_params = None
    best_rmse = float("inf")

    all_configs = list(ParameterGrid(RF_PARAM_GRID))

    rng = random.Random(random_state)
    sampled_configs = rng.sample(
        all_configs,
        k=min(100, len(all_configs)),
    )

    for params in sampled_configs:
        model = RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1,
            **params,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        val_rmse = rmse(predictions, y_val)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_params = params.copy()

    return best_params, best_rmse


def fit_final_random_forest(
    train_path,
    params,
    random_state=42
):
    X_train, y_train = load_xy(
        train_path
    )

    model = RandomForestRegressor(
        random_state=random_state,
        n_jobs=-1,
        **params
    )

    model.fit(X_train, y_train)
    return model


XGB_PARAM_GRID = {
    "n_estimators": [10, 20, 50, 100],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda": [1, 10],
}


def tune_xgboost(
    train_small_path,
    val_path,
    random_state=42,
    n_iter=100,
):
    X_train, y_train = load_xy(train_small_path)
    X_val, y_val = load_xy(val_path)

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("XGBoost training or validation set is empty.")

    all_configs = list(ParameterGrid(XGB_PARAM_GRID))

    rng = random.Random(random_state)
    sampled_configs = rng.sample(
        all_configs,
        k=min(n_iter, len(all_configs)),
    )

    best_params = None
    best_rmse = float("inf")

    for i, params in enumerate(sampled_configs, start=1):
        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
            **params,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        val_rmse = rmse(predictions, y_val)

        print(
            f"XGBoost config {i}/{len(sampled_configs)}: "
            f"validation RMSE={val_rmse:.6f}"
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_params = params.copy()

    return best_params, best_rmse


def fit_final_xgboost(
    train_path,
    params,
    random_state=42,
):
    X_train, y_train = load_xy(train_path)

    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        **params,
    )

    model.fit(X_train, y_train)
    return model

def fit_fixed_1nn(
    train_path,
):
    X_train, y_train = load_xy(
        train_path
    )

    model = KNeighborsRegressor(
        n_neighbors=1,
        metric="euclidean",
        weights="uniform",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, test_path):
    X_test, y_test = load_xy(
        test_path
    )

    if len(X_test) == 0:
        raise ValueError("Test set is empty.")

    predictions = model.predict(X_test)

    return {
        "rmse": rmse(predictions, y_test),
        "mae": mae(predictions, y_test),
    }


def evaluate_naive(test_path, context_length=512):

    X, y = load_arrow(test_path)

    predictions_mean = []
    predictions_last = []
    valid_y = []

    for i in range(len(X)):
        series = X[i][-context_length:]

        if len(series) == 0:
            continue

        if not np.all(np.isfinite(series)):
            continue

        if not np.isfinite(y[i]):
            continue

        predictions_mean.append(np.mean(series))
        predictions_last.append(series[-1])
        valid_y.append(y[i])

    valid_y = np.asarray(valid_y, dtype=np.float64)

    if len(valid_y) == 0:
        raise ValueError("No valid test samples for naive baselines.")

    return {
        "mean_rmse": rmse(predictions_mean, valid_y),
        "mean_mae": mae(predictions_mean, valid_y),
        "last_rmse": rmse(predictions_last, valid_y),
        "last_mae": mae(predictions_last, valid_y),
    }



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--index",type=int,required=True)
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()



    cur.execute(
        """
        SELECT DISTINCT dataset, train_data
        FROM runs
        WHERE id = ?
        """,
        (args.index,),
    )

    rows = cur.fetchall()
    conn.close()

    results = []

    for dataset, train_path in rows:

        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset}")
        print(f"{'=' * 60}")

        train_path = Path(train_path)
        dataset_dir = train_path.parent

        train_small_path = dataset_dir / "train_small_uni.arrow"
        val_path = dataset_dir / "eval_uni.arrow"
        final_train_path = dataset_dir / "train_uni.arrow"
        final_test_path = dataset_dir / "test_uni.arrow"

        final_train_path_multi = dataset_dir / "train_multi.arrow"
        train_small_path_multi = dataset_dir / "train_small_multi.arrow"
        val_path_multi = dataset_dir / "eval_multi.arrow"
        final_test_path_multi = dataset_dir / "test_multi.arrow"

        for path in [
            train_small_path,
            val_path,
            final_train_path,
            final_test_path,
            final_train_path_multi,
            train_small_path_multi,
            val_path_multi,
            final_test_path_multi,

        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required data file not found: {path}"
                )



        print("\nTuning KNN...")

        knn_params, knn_val_rmse = tune_knn(
            train_small_path_multi,
            val_path_multi,
        )

        print(f"Best KNN parameters: {knn_params}")
        print(f"Best KNN validation RMSE: {knn_val_rmse:.6f}")

        knn = fit_final_knn(
            final_train_path_multi,
            knn_params,
        )

        knn_metrics = evaluate_model(
            knn,
            final_test_path_multi,
        )

        print("\nFitting fixed 1-NN...")

        nn1 = fit_fixed_1nn(
            final_train_path_multi,
        )

        nn1_metrics = evaluate_model(
            nn1,
            final_test_path_multi,
        )


        print("\nTuning Random Forest...")

        rf_params, rf_val_rmse = tune_random_forest(
            train_small_path_multi,
            val_path_multi,
            RANDOM_STATE,
        )

        print(f"Best RF parameters: {rf_params}")
        print(f"Best RF validation RMSE: {rf_val_rmse:.6f}")

        rf = fit_final_random_forest(
            final_train_path_multi,
            rf_params,
            RANDOM_STATE,
        )

        rf_metrics = evaluate_model(
            rf,
            final_test_path_multi,
        )


        naive_metrics = evaluate_naive(
            final_test_path,
        )

        print("\nTuning XGBoost...")

        xgb_params, xgb_val_rmse = tune_xgboost(
            train_small_path_multi,
            val_path_multi,
            RANDOM_STATE,
        )

        print(f"Best XGBoost parameters: {xgb_params}")
        print(f"Best XGBoost validation RMSE: {xgb_val_rmse:.6f}")

        xgb = fit_final_xgboost(
            final_train_path_multi,
            xgb_params,
            RANDOM_STATE,
        )

        xgb_metrics = evaluate_model(
            xgb,
            final_test_path_multi,
        )


        row = {
            "dataset": dataset,

            "1nn_rmse": nn1_metrics["rmse"],
            "1nn_mae": nn1_metrics["mae"],

            "knn_rmse": knn_metrics["rmse"],
            "knn_mae": knn_metrics["mae"],

            "rf_rmse": rf_metrics["rmse"],
            "rf_mae": rf_metrics["mae"],

            "xgb_rmse": xgb_metrics["rmse"],
            "xgb_mae": xgb_metrics["mae"],

            **naive_metrics,
        }

        results.append(row)

        # Save after every completed dataset.
        output_path = Path(OUTPUT_CSV).parent / "dataset_results" / (
            f"{dataset}_baselines_optimized.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(results).to_csv(
            output_path,
            index=False,
        )

        print(f"Saved progress: {len(results)} datasets -> {output_path}")
