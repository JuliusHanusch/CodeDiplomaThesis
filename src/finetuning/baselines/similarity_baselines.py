import sqlite3
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from gluonts.dataset.arrow import ArrowFile

from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances

from scipy.spatial.distance import cdist


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
    y = np.array(labels, dtype=np.float32)

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



def random_baseline(
    y
):

    classes = np.unique(y)

    preds = np.random.choice(
        classes,
        size=len(y)
    )

    return evaluate_classifier_predictions(
        y,
        preds
    )



def knn_baseline(
    X_train,
    y_train,
    X_test,
    y_test,
    k
):

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(
        X_train,
        y_train
    )

    preds = model.predict(
        X_test
    )


    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(
            X_test
        )

        scores = probs

    else:

        scores = None


    return evaluate_classifier_predictions(
        y_test,
        preds,
        scores
    )



# ============================================================
# DTW
# ============================================================


def dtw_distance(
    x,
    y
):

    n = len(x)
    m = len(y)

    dtw = np.full(
        (n+1,m+1),
        np.inf
    )

    dtw[0,0] = 0


    for i in range(1,n+1):

        for j in range(1,m+1):

            cost = abs(
                x[i-1]-y[j-1]
            )

            dtw[i,j] = cost + min(
                dtw[i-1,j],
                dtw[i,j-1],
                dtw[i-1,j-1]
            )


    return dtw[n,m]



def dtw_1nn(
    X_train,
    y_train,
    X_test
):

    preds = []


    for x in tqdm(
        X_test,
        desc="DTW 1-NN"
    ):

        distances = []

        for xt in X_train:

            distances.append(
                dtw_distance(
                    x,
                    xt
                )
            )


        idx = np.argmin(
            distances
        )

        preds.append(
            y_train[idx]
        )


    return np.array(preds)

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

    for dataset, train_data, test_data in datasets:

        print(
            f"\nEvaluating: {dataset}"
        )


        X_train, y_train, X_test, y_test = load_dataset(
            dataset,
            train_data,
            test_data,
            context_length=CONTEXT_LENGTH
        )



        res = random_baseline(
            y_test
        )

        results.append(
            {
                "dataset": dataset,
                "method": "Random",
                **res
            }
        )


        res = knn_baseline(
            X_train,
            y_train,
            X_test,
            y_test,
            1
        )
        results.append(
            {
                "dataset": dataset,
                "method": "1-NN",
                **res
            }
        )


        res = knn_baseline(
            X_train,
            y_train,
            X_test,
            y_test,
            5
        )

        results.append(
            {
                "dataset": dataset,
                "method": "5-NN",
                **res
            }
        )


    conn.close()


    results = pd.DataFrame(
        results
    )


    output = RESULT_DIR / "similarity_baselines.csv"


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