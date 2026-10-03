from pathlib import Path
import numpy as np
import pandas as pd
import sqlite3

from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

from gluonts.dataset.arrow import ArrowFile



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

    series = []
    labels = []

    for entry in dataset:

        target = np.asarray(
            entry["target"],
            dtype=np.float32
        )

        if "label" in entry:
            label = entry["label"]

        elif "y" in entry:
            label = entry["y"]

        else:
            raise KeyError(
                "No label found in dataset entry"
            )

        series.append(target)
        labels.append(float(label))

    return (
        np.stack(series),
        np.asarray(labels, dtype=np.float32)
    )



def extract_features(series):

    series = np.asarray(
        series,
        dtype=np.float64
    )

    # Replace invalid values
    series = np.nan_to_num(
        series,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    if len(series) == 0:
        return np.zeros(6, dtype=np.float32)

    mean = np.mean(series)
    std = np.std(series)
    min_v = np.min(series)
    max_v = np.max(series)
    last = series[-1]

    x = np.arange(len(series), dtype=np.float64)

    if (
        len(series) < 2
        or np.all(series == series[0])
    ):
        slope = 0.0

    else:
        try:
            slope = np.polyfit(
                x,
                series,
                1
            )[0]

        except Exception:
            slope = 0.0

    features = np.array(
        [
            mean,
            std,
            min_v,
            max_v,
            last,
            slope
        ],
        dtype=np.float32
    )

    return np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


def create_features(X, context_length=512):

    X_feat = []
    valid_indices = []

    for i in range(len(X)):

        series = X[i][-context_length:]

        # Skip series with any invalid values
        if not np.all(np.isfinite(series)):
            continue

        X_feat.append(series)
        valid_indices.append(i)

    if not X_feat:
        return (
            np.empty((0, context_length), dtype=np.float32),
            np.array([], dtype=np.int64)
        )

    return (
        np.asarray(X_feat, dtype=np.float32),
        np.asarray(valid_indices, dtype=np.int64)
    )


def train_knn(
    train_path,
    context_length=512,
    n_neighbors=5
):

    X_train, y_train = load_arrow(train_path)

    X_feat, indices = create_features(
        X_train,
        context_length
    )

    y_target = y_train[indices]

    # Remove invalid targets
    valid = np.isfinite(y_target)

    X_feat = X_feat[valid]
    y_target = y_target[valid]

    knn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean"
    )

    knn.fit(X_feat)

    return knn, X_feat, y_target


def evaluate_knn(
    knn,
    y_train,
    test_path,
    context_length=512
):

    X_test, y_test = load_arrow(test_path)

    X_feat, indices = create_features(
        X_test,
        context_length
    )

    y_target = y_test[indices]

    # Remove invalid targets
    valid = np.isfinite(y_target)

    X_feat = X_feat[valid]
    y_target = y_target[valid]

    # Get five nearest neighbors
    distances, indices = knn.kneighbors(
        X_feat,
        n_neighbors=5
    )

    neighbors_y = y_train[indices]


    predictions_1nn = neighbors_y[:, 0]

    predictions_5nn = np.mean(
        neighbors_y,
        axis=1
    )

    return {
        "1nn_rmse": rmse(
            predictions_1nn,
            y_target
        ),

        "1nn_mae": mae(
            predictions_1nn,
            y_target
        ),

        "5nn_rmse": rmse(
            predictions_5nn,
            y_target
        ),

        "5nn_mae": mae(
            predictions_5nn,
            y_target
        )
    }

def train_euclidean_knn(
    train_path,
    context_length=512
):

    X_train, y_train = load_arrow(train_path)

    X_feat, indices = create_features(
        X_train,
        context_length
    )

    y_target = y_train[indices]

    valid = np.isfinite(y_target)

    X_feat = X_feat[valid]
    y_target = y_target[valid]

    knn = NearestNeighbors(
        n_neighbors=5,
        metric="euclidean"
    )

    knn.fit(X_feat)

    return knn, X_feat, y_target

def evaluate_euclidean_knn(
    knn,
    y_train,
    test_path,
    context_length=512
):

    X_test, y_test = load_arrow(test_path)

    X_feat, indices = create_features(
        X_test,
        context_length
    )

    y_target = y_test[indices]

    valid = np.isfinite(y_target)

    X_feat = X_feat[valid]
    y_target = y_target[valid]

    distances, indices = knn.kneighbors(
        X_feat,
        n_neighbors=5
    )

    neighbors_y = y_train[indices]

    predictions_1nn = neighbors_y[:, 0]
    predictions_5nn = np.mean(neighbors_y, axis=1)

    return {
        "1nn_rmse": rmse(
            predictions_1nn,
            y_target
        ),
        "1nn_mae": mae(
            predictions_1nn,
            y_target
        ),
        "5nn_rmse": rmse(
            predictions_5nn,
            y_target
        ),
        "5nn_mae": mae(
            predictions_5nn,
            y_target
        )
    }


def evaluate_naive(
    test_path,
    context_length=512
):

    X, y = load_arrow(test_path)

    predictions_mean = []
    predictions_last = []
    valid_y = []

    for i in range(len(X)):

        series = X[i][-context_length:]

        valid_series = series[
            np.isfinite(series)
        ]

        if len(valid_series) == 0:
            continue

        if not np.isfinite(y[i]):
            continue

        predictions_mean.append(
            np.mean(valid_series)
        )

        predictions_last.append(
            valid_series[-1]
        )

        valid_y.append(
            y[i]
        )

    valid_y = np.asarray(
        valid_y,
        dtype=np.float32
    )

    return {
        "mean_rmse": rmse(
            predictions_mean,
            valid_y
        ),

        "mean_mae": mae(
            predictions_mean,
            valid_y
        ),

        "last_rmse": rmse(
            predictions_last,
            valid_y
        ),

        "last_mae": mae(
            predictions_last,
            valid_y
        )
    }

if __name__ == "__main__":

    DB_PATH = (
        "/data/horse/ws/"
        "juha972b-AION-BERT-Chronos/"
        "BERTi/src/finetuning/tser/Final/"
        "tser_bestConfigs_time.db"
    )

    OUTPUT_CSV = (
        "/data/horse/ws/"
        "juha972b-AION-BERT-Chronos/"
        "BERTi/Results/Finetuning/TSER/"
        "tser_baselines.csv"
    )

    CONTEXT_LENGTH = 512

    conn = sqlite3.connect(DB_PATH)

    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT
            dataset,
            train_data,
            test_data
        FROM runs
    """)

    rows = cur.fetchall()

    conn.close()

    results = []


    for dataset, train_path, test_path in rows:

        print(
            f"\nProcessing {dataset}"
        )

        print("Training kNN...")

        knn, X_train_feat, y_train = train_euclidean_knn(
            train_path,
            context_length=CONTEXT_LENGTH
        )

        knn_metrics = evaluate_euclidean_knn(
            knn,
            y_train,
            test_path,
            context_length=CONTEXT_LENGTH
        )


        print("Evaluating naive baselines...")

        naive_metrics = evaluate_naive(
            test_path,
            context_length=CONTEXT_LENGTH
        )


        row = {
            "dataset": dataset,

            **knn_metrics,

            **naive_metrics
        }

        results.append(row)


        print(
            f"1-NN RMSE: "
            f"{knn_metrics['1nn_rmse']:.6f}"
        )

        print(
            f"5-NN RMSE: "
            f"{knn_metrics['5nn_rmse']:.6f}"
        )

        print(
            f"Mean RMSE: "
            f"{naive_metrics['mean_rmse']:.6f}"
        )

        print(
            f"Last RMSE: "
            f"{naive_metrics['last_rmse']:.6f}"
        )

    df = pd.DataFrame(results)

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        f"\nSaved results to {OUTPUT_CSV}"
    )