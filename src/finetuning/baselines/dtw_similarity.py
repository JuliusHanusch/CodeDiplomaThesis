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

datasets = [
    {
        "name": "UCI-HAR",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCI_HAR/UCI HAR Dataset/train/",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCI_HAR/UCI HAR Dataset/test/",
    },
    {
        "name": "ArrowHead",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/ArrowHead/ArrowHead_TRAIN.tsv",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/ArrowHead/ArrowHead_TEST.tsv",
    },
    {
        "name": "DistalPhalanxTW",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/DistalPhalanxTW/DistalPhalanxTW_TRAIN.tsv",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/DistalPhalanxTW/DistalPhalanxTW_TEST.tsv",
    },
    {
        "name": "GestureMidAirD2",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/GestureMidAirD2/GestureMidAirD2_TRAIN.tsv",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/GestureMidAirD2/GestureMidAirD2_TEST.tsv",
    },
    {
        "name": "Wafer",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/Wafer/Wafer_TRAIN.tsv",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/Wafer/Wafer_TEST.tsv",
    },
    {
        "name": "ArabicSpokenDigits",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Similarity/ArabicSpokenDigits/arabic_digits_univariate_train.npz",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Similarity/ArabicSpokenDigits/arabic_digits_univariate_test.npz",

    },
]

def load_ucr_tsv(tsv_path):

    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None
    ).values

    y = df[:, 0].astype(int)
    X = df[:, 1:].astype(np.float32)

    # map labels to 0..N
    unique = np.unique(y)
    label_map = {v: i for i, v in enumerate(unique)}
    y = np.vectorize(label_map.get)(y)

    return X, y



def load_uci_har(train_dir, test_dir):

    train_dir = Path(train_dir)
    test_dir = Path(test_dir)

    X_train = np.loadtxt(
        train_dir / "X_train.txt"
    ).astype(np.float32)

    y_train = np.loadtxt(
        train_dir / "y_train.txt"
    ).astype(int) - 1


    X_test = np.loadtxt(
        test_dir / "X_test.txt"
    ).astype(np.float32)

    y_test = np.loadtxt(
        test_dir / "y_test.txt"
    ).astype(int) - 1

    return X_train, y_train, X_test, y_test

def load_arabic_digits(
    train_path,
    test_path,
    task="digit"
):

    train = np.load(
        train_path,
        allow_pickle=True
    )

    test = np.load(
        test_path,
        allow_pickle=True
    )


    X_train = list(train["X"])
    X_test = list(test["X"])


    if task == "digit":

        y_train = train["digit_labels"]
        y_test = test["digit_labels"]

    elif task == "voice":

        y_train = train["speaker_labels"]
        y_test = test["speaker_labels"]

    else:
        raise ValueError(
            "task must be digit or voice"
        )


    return (
        X_train,
        y_train,
        X_test,
        y_test
    )


def load_dataset(info):

    if info["name"] == "UCI-HAR":

        return load_uci_har(
            info["train"],
            info["test"]
        )

    elif info["name"] == "ArabicSpokenDigits":

        X_train, y_train = load_arabic_digits(
            info["train"]
        )

        X_test, y_test = load_arabic_digits(
            info["test"]
        )

        return (
            X_train,
            y_train,
            X_test,
            y_test
        )

    else:

        X_train, y_train = load_ucr_tsv(
            info["train"]
        )

        X_test, y_test = load_ucr_tsv(
            info["test"]
        )

        return (
            X_train,
            y_train,
            X_test,
            y_test
        )


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

def dtw_distance(x,y,window=None):

    n=len(x)
    m=len(y)

    if window is None:
        window=max(n,m)

    dtw=np.full(
        (n+1,m+1),
        np.inf
    )

    dtw[0,0]=0

    for i in range(1,n+1):

        for j in range(
            max(1,i-window),
            min(m,i+window)+1
        ):

            cost=abs(
                x[i-1]-y[j-1]
            )

            dtw[i,j]=cost+min(
                dtw[i-1,j],
                dtw[i,j-1],
                dtw[i-1,j-1]
            )

    return dtw[n,m]

def create_similarity_pairs(X,y,n_pairs):

    X1=[]
    X2=[]
    labels=[]

    classes=np.unique(y)

    while len(labels)<n_pairs:

        if np.random.rand()<0.5:

            c=np.random.choice(classes)

            idx=np.where(
                y==c
            )[0]

            if len(idx)<2:
                continue

            i,j=np.random.choice(
                idx,
                2,
                replace=False
            )

            label=1

        else:

            i,j=np.random.choice(
                len(X),
                2,
                replace=False
            )

            if y[i]==y[j]:
                continue

            label=0


        X1.append(X[i])
        X2.append(X[j])
        labels.append(label)


    return (
        X1,
        X2,
        np.array(labels)
    )


def dtw_similarity_baseline(
    X1,
    X2
):

    scores=[]

    for x1,x2 in tqdm(
        zip(X1,X2),
        total=len(X1),
        desc="DTW similarity"
    ):

        d=dtw_distance(
            x1,
            x2
        )

        scores.append(
            np.exp(-d)
        )

    scores=np.array(
        scores
    )

    preds=(
        scores>0.5
    ).astype(int)

    return preds,scores




if __name__=="__main__":

    RESULT="/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/Results/Finetuning/Similarity/dtw_similarity.csv"

    np.random.seed(42)

    results=[]

    for info in datasets:

        print(
            f"\nEvaluating: {info['name']}"
        )

        tasks=["class"]

        if info["name"]=="ArabicSpokenDigits":
            tasks=[
                "digit",
                "voice"
            ]

        for task in tasks:

            if info["name"]=="ArabicSpokenDigits":

                X_train,y_train,X_test,y_test=load_arabic_digits(
                    info["train"],
                    info["test"],
                    task
                )

            elif info["name"]=="UCI-HAR":

                X_train,y_train,X_test,y_test=load_uci_har(
                    info["train"],
                    info["test"]
                )

            else:

                X_train,y_train=load_ucr_tsv(
                    info["train"]
                )

                X_test,y_test=load_ucr_tsv(
                    info["test"]
                )


            X1_test,X2_test,y_pairs=create_similarity_pairs(
                X_test,
                y_test,
                len(X_test)*5
            )


            dtw_preds,dtw_scores=dtw_similarity_baseline(
                X1_test,
                X2_test
            )


            res=evaluate_classifier_predictions(
                y_pairs,
                dtw_preds,
                dtw_scores
            )


            print(
                f"{info['name']} ({task}) "
                f"Accuracy={res['accuracy']:.4f}, "
                f"AUROC={res['auroc']:.4f}"
            )


            results.append(
                {
                    "dataset":info["name"],
                    "task":task,
                    "method":"DTW",
                    "auroc":res["auroc"],
                    "accuracy":res["accuracy"]
                }
            )


    df=pd.DataFrame(
        results
    )


    df=df[
        [
            "dataset",
            "task",
            "method",
            "auroc",
            "accuracy"
        ]
    ]


    print("\nFinished")
    print(df)


    df.to_csv(
        RESULT,
        index=False
    )


    print(
        f"Saved to {RESULT}"
    )