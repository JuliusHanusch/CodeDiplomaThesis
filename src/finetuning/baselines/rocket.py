import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from collections import Counter

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from gluonts.dataset.arrow import ArrowFile


# =========================================================
# CONFIG
# =========================================================
DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/tser/tser.db"
OUTPUT_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/Results/Finetuning/TSER/rocket_results.csv"


# =========================================================
# SAFE LOADER (ARROW + TSV)
# =========================================================
def load_arrow(path):
    dataset = ArrowFile(Path(path))

    X, y = [], []

    for entry in dataset:
        x = np.asarray(entry["target"], dtype=np.float32)

        # FIX NaNs/infs at source
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        y_val = float(entry.get("label", entry.get("y", 0.0)))

        X.append(x)
        y.append(y_val)

    return np.asarray(X), np.asarray(y, dtype=np.float32)


def load_tsv(path, context_length=512):
    df = pd.read_csv(path, sep="\t", header=None).values

    y = df[:, 0].astype(float)
    X = df[:, 1:].astype(np.float32)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if X.shape[1] < context_length:
        pad = context_length - X.shape[1]
        X = np.pad(X, ((0, 0), (0, pad)), mode="constant")
    else:
        X = X[:, -context_length:]

    return X, y


# =========================================================
# ROCKET (ORIGINAL STYLE FEATURE MAP)
# =========================================================
def generate_kernels(n_kernels, input_length):
    lengths = np.random.choice([7, 9, 11], size=n_kernels)

    weights = []
    biases = []

    for L in lengths:
        w = np.random.normal(0, 1, L)
        w = w - w.mean()
        weights.append(w)
        biases.append(np.random.uniform(-1, 1))

    return weights, np.array(biases), lengths


def apply_kernels(X, kernels):
    """
    X: (n_samples, length)
    returns: (n_samples, 2 * n_kernels)
    """
    weights, biases, lengths = kernels

    n_samples = X.shape[0]
    n_kernels = len(weights)

    feats = np.zeros((n_samples, n_kernels * 2), dtype=np.float32)

    for i in range(n_samples):
        x = X[i]
        out = []

        for k in range(n_kernels):
            w = weights[k]
            b = biases[k]

            L = len(w)

            # convolution (valid)
            if len(x) < L:
                conv = np.array([0.0])
            else:
                conv = np.convolve(x, w, mode="valid") + b

            # PPV + MAX (original ROCKET idea)
            ppv = np.mean(conv > 0)
            mx = np.max(conv)

            out.extend([ppv, mx])

        feats[i] = np.array(out)

    return feats


# =========================================================
# ROCKET MODEL
# =========================================================
class RocketRegressor:

    def __init__(self, n_kernels=1000):
        self.n_kernels = n_kernels
        self.kernels = None

        # regression head (IMPORTANT)
        self.model = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 3, 10))
        )

    def fit(self, X, y):
        X = np.nan_to_num(X, nan=0.0)

        self.kernels = generate_kernels(self.n_kernels, X.shape[1])
        X_feat = apply_kernels(X, self.kernels)

        self.model.fit(X_feat, y)

    def predict(self, X):
        X = np.nan_to_num(X, nan=0.0)
        X_feat = apply_kernels(X, self.kernels)
        return self.model.predict(X_feat)


# =========================================================
# METRICS
# =========================================================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


# =========================================================
# DATASETS FROM DB
# =========================================================
def get_runs():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT dataset, train_data, test_data
        FROM runs
    """)

    rows = cur.fetchall()
    conn.close()
    return rows


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    results = []

    for dataset, train_path, test_path in get_runs():

        print(f"\nProcessing {dataset}")

       
        X_train, y_train = load_arrow(train_path)
        
        X_test, y_test = load_arrow(test_path)
        # -------------------------
        # TRAIN ROCKET
        # -------------------------
        model = RocketRegressor(n_kernels=10000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        # -------------------------
        # METRICS
        # -------------------------
        row = {
            "dataset": dataset,
            "rmse": rmse(y_test, preds),
            "mae": mae(y_test, preds),
        }

        results.append(row)

        print(f"RMSE: {row['rmse']:.6f}")
        print(f"MAE : {row['mae']:.6f}")

    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved → {OUTPUT_PATH}")