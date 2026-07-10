import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from collections import Counter
from gluonts.dataset.arrow import ArrowFile

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/classification.db"
OUTPUT_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/Results/Finetuning/Classification/all_baselines.csv"



def load_arrow(path: str):
    dataset = ArrowFile(Path(path))

    series = []
    labels = []

    for entry in dataset:
        target = np.asarray(entry["target"], dtype=np.float32)

        if "label" in entry:
            label = entry["label"]
        elif "y" in entry:
            label = entry["y"]
        else:
            raise KeyError("No label found in Arrow dataset")

        series.append(target)
        labels.append(float(label))

    X = np.stack(series)
    y = np.array(labels, dtype=np.float32)

    return X, y

def load_runs_from_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT dataset, train_data, test_data
        FROM runs
    """)

    rows = cur.fetchall()
    conn.close()
    return rows


def load_ucr_tsv(tsv_path, context_length=512):
    df = pd.read_csv(tsv_path, sep="\t", header=None).values

    y = df[:, 0]
    X = df[:, 1:].astype(np.float32)


    y = y.astype(int)
    unique = np.unique(y)
    label_map = {v: i for i, v in enumerate(unique)}
    y = np.vectorize(label_map.get)(y)

    # pad / truncate
    if X.shape[1] < context_length:
        pad = context_length - X.shape[1]
        X = np.pad(X, ((0, 0), (0, pad)), mode="constant")
    else:
        X = X[:, -context_length:]

    return X, y


def load_uci_har(path, context_length=512):
    path = Path(path)

    X = np.loadtxt(path / "X_test.txt")
    y = np.loadtxt(path / "y_test.txt").astype(int) - 1

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.shape[1] < context_length:
        pad = context_length - X.shape[1]
        X = np.pad(X, ((0, 0), (0, pad)))
    else:
        X = X[:, -context_length:]

    return X, y


def load_dataset(dataset, train_path, test_path, context_length=512):

    if dataset == "UCI-HAR":
        X_train, y_train = load_arrow(train_path)
        X_test, y_test = load_uci_har(Path(test_path), context_length)
        return X_train, y_train, X_test, y_test

    # UCR / TSV
    X_train, y_train,= load_arrow(train_path)
    X_test, y_test, = load_ucr_tsv(test_path, context_length=context_length)

    return X_train, y_train, X_test, y_test


def run_baselines(X_train, y_train, X_test, y_test):

    results = {}

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    majority = Counter(y_train).most_common(1)[0][0]
    preds = np.full_like(y_test, fill_value=majority)

    results["naive_acc"] = accuracy_score(y_test, preds)
    results["naive_f1"] = f1_score(y_test, preds, average="weighted")

    scaler = StandardScaler()

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0)

    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    knn.fit(X_train_s, y_train)
    preds = knn.predict(X_test_s)

    results["knn_acc"] = accuracy_score(y_test, preds)
    results["knn_f1"] = f1_score(y_test, preds, average="weighted")

    rf = RandomForestClassifier(
        n_estimators=20,
        max_depth=3,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    results["rf_acc"] = accuracy_score(y_test, preds)
    results["rf_f1"] = f1_score(y_test, preds, average="weighted")

    return results

if __name__ == "__main__":

    context_length = 512

    rows = load_runs_from_db()

    all_results = []

    for dataset, train_path, test_path in rows:

        print(f"\nProcessing {dataset}")

        X_train, y_train, X_test, y_test = load_dataset(
            dataset,
            train_path,
            test_path,
            context_length=context_length
        )

        metrics = run_baselines(X_train, y_train, X_test, y_test)

        result_row = {
            "dataset": dataset,
            **metrics
        }

        all_results.append(result_row)

        print(f"Naive ACC: {metrics['naive_acc']:.4f}")
        print(f"1NN ACC:   {metrics['knn_acc']:.4f}")
        print(f"RF ACC:    {metrics['rf_acc']:.4f}")

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved all results to: {OUTPUT_PATH}")