from pathlib import Path
import numpy as np
import pandas as pd
import sqlite3
import sys
import argparse

from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

root_dir = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi")
sys.path.append(str(root_dir.resolve()))
sys.path.append(str((root_dir / "src").resolve()))
sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))

from gluonts.dataset.arrow import ArrowFile


# -------------------------
# METRICS
# -------------------------
def rmse(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return np.sqrt(np.mean((preds - labels) ** 2))


def mae(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return np.mean(np.abs(preds - labels))

def train_knn_ed(train_path, context_length=512):

    X, y = load_arrow(train_path)

    X_feat = []

    for i in range(len(X)):
        series = X[i][-context_length:]
        X_feat.append(extract_features(series))

    X_feat = np.vstack(X_feat)
    y = np.asarray(y, dtype=np.float32)

    knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    knn.fit(X_feat)

    return knn, X_feat, y

def evaluate_knn(knn, X_train_feat, y_train, test_path, context_length=512):

    X, y = load_arrow(test_path)

    preds_1nn = []
    preds_5nn = []

    for i in range(len(X)):

        series = X[i][-context_length:]
        feat = extract_features(series).reshape(1, -1)

        distances, indices = knn.kneighbors(feat, n_neighbors=5)

        neighbors_y = y_train[indices[0]]

        # 1-NN
        preds_1nn.append(neighbors_y[0])

        # 5-NN (mean regression)
        preds_5nn.append(np.mean(neighbors_y))

    return {
        "1nn_rmse": rmse(preds_1nn, y),
        "1nn_mae": mae(preds_1nn, y),

        "5nn_rmse": rmse(preds_5nn, y),
        "5nn_mae": mae(preds_5nn, y),
    }
# -------------------------
# FEATURE ENGINEERING
# -------------------------
def extract_features(series: np.ndarray):
    series = np.asarray(series)
    series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)

    mean = np.mean(series)
    std = np.std(series)
    min_v = np.min(series)
    max_v = np.max(series)
    last = series[-1]

    x = np.arange(len(series))

    if len(series) < 2 or np.all(series == series[0]):
        slope = 0.0
    else:
        try:
            slope = np.polyfit(x, series, 1)[0]
        except Exception:
            slope = 0.0

    feat = np.array([mean, std, min_v, max_v, last, slope], dtype=np.float32)
    return np.nan_to_num(feat)


# -------------------------
# DATA LOADER
# -------------------------
def load_arrow(path: str):
    dataset = ArrowFile(Path(path))

    series, labels = [], []

    for entry in dataset:
        target = np.asarray(entry["target"], dtype=np.float32)

        if "label" in entry:
            label = entry["label"]
        elif "y" in entry:
            label = entry["y"]
        else:
            raise KeyError("No label found in dataset entry")

        series.append(target)
        labels.append(float(label))

    return np.stack(series), np.array(labels, dtype=np.float32)


# -------------------------
# TRAIN RIDGE
# -------------------------
def train_ridge(train_path, context_length=512):

    X, y = load_arrow(train_path)

    X_feat = []
    y_target = []

    for i in range(len(X)):
        series = X[i][-context_length:]
        X_feat.append(extract_features(series))
        y_target.append(y[i])

    X_feat = np.vstack(X_feat)
    y_target = np.asarray(y_target, dtype=np.float32)

    model = Ridge(alpha=1.0)
    model.fit(X_feat, y_target)

    return model


# -------------------------
# EVAL
# -------------------------
def evaluate_all(ridge_model, test_path, context_length=512):

    X, y = load_arrow(test_path)

    preds_mean, preds_last, preds_ridge = [], [], []

    for i in range(len(X)):
        series = X[i][-context_length:]

        preds_mean.append(np.mean(series))
        preds_last.append(series[-1])

        feat = extract_features(series).reshape(1, -1)
        preds_ridge.append(ridge_model.predict(feat)[0])

    return {
        "mean_rmse": rmse(preds_mean, y),
        "mean_mae": mae(preds_mean, y),

        "last_rmse": rmse(preds_last, y),
        "last_mae": mae(preds_last, y),

        "ridge_rmse": rmse(preds_ridge, y),
        "ridge_mae": mae(preds_ridge, y),
    }


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/tser/tser.db"
    OUTPUT_CSV = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/Results/Finetuning/TSER/tser_baselines.csv"

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT dataset, train_data, test_data
        FROM runs
    """)

    rows = cur.fetchall()
    conn.close()

    results = []

    for dataset, train_path, test_path in rows:

        print(f"\nProcessing {dataset}")

        # Ridge baseline
        ridge_model = train_ridge(train_path)
        ridge_metrics = evaluate_all(ridge_model, test_path)

        # kNN-ED baseline
        knn, X_train_feat, y_train = train_knn_ed(train_path)
        knn_metrics = evaluate_knn(knn, X_train_feat, y_train, test_path)

        row = {
            "dataset": dataset,

            **ridge_metrics,
            **knn_metrics
        }

        results.append(row)

        print(f"Ridge RMSE: {ridge_metrics['ridge_rmse']:.6f}")
        print(f"1NN RMSE: {knn_metrics['1nn_rmse']:.6f}")
        print(f"5NN RMSE: {knn_metrics['5nn_rmse']:.6f}")
        
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved results to {OUTPUT_CSV}")