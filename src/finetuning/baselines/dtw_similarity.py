import sqlite3
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from gluonts.dataset.arrow import ArrowFile

from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier


from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier



# ============================================================
# PATHS
# ============================================================

root_dir = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi")
sys.path.append(str(root_dir.resolve()))
sys.path.append(str((root_dir / "src").resolve()))


DB_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/src/finetuning/similarity/similarity.db"
)

RESULT_DIR = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/Results/Finetuning/Similarity"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

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
    y = np.array(labels, dtype=np.int64)

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return X, y


def load_ucr_tsv(tsv_path, context_length=512):

    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None
    ).values

    y = df[:, 0]
    X = df[:, 1:].astype(np.float32)

    y = y.astype(int)

    unique = np.unique(y)
    label_map = {
        v: i for i, v in enumerate(unique)
    }
    y = np.vectorize(label_map.get)(y)
    if X.shape[1] < context_length:
        pad = context_length - X.shape[1]
        X = np.pad(
            X,
            ((0,0),(0,pad)),
            mode="constant"
        )
    else:
        X = X[:, -context_length:]


    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
)

    return X, y



def load_uci_har(test_dir, context_length=512):

    test_dir = Path(test_dir)

    X = np.loadtxt(
        test_dir / "X_test.txt"
    )
    y = np.loadtxt(
        test_dir / "y_test.txt"
    ).astype(int)

    y = y - 1
    if X.ndim == 1:
        X = X.reshape(1,-1)
    if X.shape[1] < context_length:
        pad = context_length - X.shape[1]

        X = np.pad(
            X,
            ((0,0),(0,pad))
        )

    else:
        X = X[:, -context_length:]


    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return X, y


def evaluate_classifier_predictions(
    y_true,
    y_pred,
    scores=None
):

    result = {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),
        "f1":
            f1_score(
                y_true,
                y_pred,
                average="macro"
            )
    }
    if scores is not None:
        try:

            result["auroc"] = roc_auc_score(
                y_true,
                scores,
                multi_class="ovr"
            )
        except:

            result["auroc"] = np.nan
    else:
        result["auroc"] = np.nan
    return result





def dtw_1nn(X_train, y_train, X_test):

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    X_train = X_train[:, np.newaxis, :]
    X_test = X_test[:, np.newaxis, :]

    clf = KNeighborsTimeSeriesClassifier(
        distance="dtw",
        n_neighbors=1,
        distance_params={"window": 0.1},
    )

    clf.fit(
        X_train,
        y_train
    )

    return clf.predict(X_test)

def load_dataset(dataset, train_path, test_path, context_length=512):

    if dataset == "UCI-HAR":
        X_train, y_train = load_arrow(train_path)
        X_test, y_test = load_uci_har(Path(test_path), context_length)
        return X_train, y_train, X_test, y_test

    # UCR / TSV
    X_train, y_train,= load_arrow(train_path)
    X_test, y_test, = load_ucr_tsv(test_path, context_length=context_length)

    return X_train, y_train, X_test, y_test



if __name__ == "__main__":

    
    CONTEXT_LENGTH = 512
    SEED = 42

    np.random.seed(SEED)

    conn = sqlite3.connect(
        DB_PATH
    )

    cur = conn.cursor()

    cur.execute(
        """
        SELECT DISTINCT
            dataset,
            train_data,
            test_data
        FROM runs
        """
    )

    datasets = cur.fetchall()

    results = []

    datasets

    for dataset, train_data, test_data in datasets:

        if dataset != "UCI-HAR":
            continue

        print(
            f"\nEvaluating: {dataset}"
        )


        X_train, y_train, X_test, y_test = load_dataset(
            dataset,
            train_data,
            test_data,
            context_length=CONTEXT_LENGTH
        )
        print(
            f"Train: {X_train.shape}, Test: {X_test.shape}"
        )

        dtw_preds = dtw_1nn(
            X_train,
            y_train,
            X_test
        )


        res = evaluate_classifier_predictions(
            y_test,
            dtw_preds
        )

        print(
            f"{dataset} DTW-1NN results: "
            f"Accuracy={res['accuracy']:.4f}, "
            f"F1={res['f1']:.4f}, "
            f"AUROC={res['auroc']:.4f}"
        )

        results.append(
            {
                "dataset": dataset,
                "method": "DTW-1NN",
                **res
            }
        )

    conn.close()


    results = pd.DataFrame(
        results
    )


    output = RESULT_DIR / "dtw_similarity.csv"


    results.to_csv(
        output,
        index=False
    )


    print("\nFinished")
    print(
        results
    )

    print(
        f"\nSaved to {output}"
    )