import sqlite3
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from gluonts.dataset.arrow import ArrowFile
from scipy.signal import resample

from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances

from scipy.spatial.distance import cdist


# ============================================================
# PATHS
# ============================================================
RESULT="/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/Results/Finetuning/Similarity/similarity_baseline.csv"

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

def evaluate_binary(y_true,y_pred,scores=None):

    result={}

    result["accuracy"]=accuracy_score(
        y_true,
        y_pred
    )

    result["f1"]=f1_score(
        y_true,
        y_pred
    )

    if scores is not None:
        try:
            result["auroc"]=roc_auc_score(
                y_true,
                scores
            )
        except:
            result["auroc"]=np.nan
    else:
        result["auroc"]=np.nan

    return result

def random_similarity_baseline(y_pairs):
    preds=np.random.randint(
        0,
        2,
        size=len(y_pairs)
    )

    return evaluate_binary(
        y_pairs,
        preds
    )


def knn_classifier_baseline(
    X_train,
    y_train,
    X1_test,
    X2_test,
    y_pairs,
    k,
    dataset
):

    X_train_emb = np.array(
        [
            sequence_embedding(x, dataset)
            for x in X_train
        ]
    )

    y_pred=[]

    for a,b in zip(X1_test,X2_test):

        ea = sequence_embedding(a, dataset)
        eb = sequence_embedding(b, dataset)

        distance_a = np.linalg.norm(
            X_train_emb - ea,
            axis=1
        )

        distance_b = np.linalg.norm(
            X_train_emb - eb,
            axis=1
        )

        idx_a = np.argsort(distance_a)[:k]
        idx_b = np.argsort(distance_b)[:k]

        class_a = np.bincount(
            y_train[idx_a]
        ).argmax()

        class_b = np.bincount(
            y_train[idx_b]
        ).argmax()

        y_pred.append(
            int(class_a == class_b)
        )

    return evaluate_binary(
        y_pairs,
        np.array(y_pred)
    )

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
        np.array(X1,dtype=object),
        np.array(X2,dtype=object),
        np.array(labels)
    )


def sequence_embedding(x, dataset=None):

    x = np.asarray(x, dtype=np.float32)

    if dataset == "ArabicSpokenDigits":
        return resample(
            x,
            512
        )

    else:
        return x



def evaluate_dataset(info):

    print(
        "\nEvaluating",
        info["name"]
    )

    results=[]

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

        else:

            X_train,y_train,X_test,y_test=load_dataset(
                info
            )


        X1,X2,y_pairs=create_similarity_pairs(
            X_test,
            y_test,
            len(X_test)*5
        )


        # Naive

        metrics=random_similarity_baseline(
            y_pairs
        )

        metrics["dataset"]=info["name"]
        metrics["task"]=task
        metrics["method"]="Naive"

        results.append(metrics)


        # 1-NN

        metrics = knn_classifier_baseline(
            X_train,
            y_train,
            X1,
            X2,
            y_pairs,
            1,
            info["name"]
        )

        metrics["dataset"]=info["name"]
        metrics["task"]=task
        metrics["method"]="1-NN"

        results.append(metrics)


        # 5-NN

        metrics = knn_classifier_baseline(
            X_train,
            y_train,
            X1,
            X2,
            y_pairs,
            5,
            info["name"]
        )

        metrics["dataset"]=info["name"]
        metrics["task"]=task
        metrics["method"]="5-NN"

        results.append(metrics)


    return results


if __name__=="__main__":

    np.random.seed(42)

    all_results=[]

    for info in datasets:

        result=evaluate_dataset(
            info
        )

        all_results.extend(
            result
        )

    df=pd.DataFrame(
        all_results
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

    print(df)

    df.to_csv(
        RESULT,
        index=False
    )

    print(
        "Saved:",
        RESULT
    )