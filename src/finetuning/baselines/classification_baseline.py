import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


OUTPUT_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/Results/Finetuning/Classification/Baselines.csv"


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



def load_dataset(info):

    if info["name"] == "UCI-HAR":
        return load_uci_har(
            info["train"],
            info["test"]
        )

    else:
        X_train, y_train = load_ucr_tsv(
            info["train"]
        )

        X_test, y_test = load_ucr_tsv(
            info["test"]
        )

        return X_train, y_train, X_test, y_test



from collections import Counter
from sklearn.ensemble import RandomForestClassifier


def run_baselines(
    X_train,
    y_train,
    X_test,
    y_test
):

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    X_train = np.nan_to_num(
        X_train,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    X_test = np.nan_to_num(
        X_test,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    results = {}


    # -------------------------
    # Naive majority baseline
    # -------------------------

    majority = Counter(y_train).most_common(1)[0][0]

    preds = np.full(
        len(y_test),
        majority
    )

    results["Naive_accuracy"] = accuracy_score(
        y_test,
        preds
    )

    results["Naive_f1"] = f1_score(
        y_test,
        preds,
        average="macro"
    )


    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    knn1 = KNeighborsClassifier(
        n_neighbors=1,
        metric="euclidean",
        n_jobs=-1
    )

    knn1.fit(
        X_train_scaled,
        y_train
    )

    preds = knn1.predict(
        X_test_scaled
    )

    results["1NN_accuracy"] = accuracy_score(
        y_test,
        preds
    )

    results["1NN_f1"] = f1_score(
        y_test,
        preds,
        average="macro"
    )


    knn5 = KNeighborsClassifier(
        n_neighbors=5,
        metric="euclidean",
        n_jobs=-1
    )

    knn5.fit(
        X_train_scaled,
        y_train
    )

    preds = knn5.predict(
        X_test_scaled
    )

    results["5NN_accuracy"] = accuracy_score(
        y_test,
        preds
    )

    results["5NN_f1"] = f1_score(
        y_test,
        preds,
        average="macro"
    )


    rf_small = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )

    rf_small.fit(
        X_train,
        y_train
    )

    preds = rf_small.predict(
        X_test
    )

    results["RF_small_accuracy"] = accuracy_score(
        y_test,
        preds
    )

    results["RF_small_f1"] = f1_score(
        y_test,
        preds,
        average="macro"
    )



    rf_large = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    rf_large.fit(
        X_train,
        y_train
    )

    preds = rf_large.predict(
        X_test
    )

    results["RF_large_accuracy"] = accuracy_score(
        y_test,
        preds
    )

    results["RF_large_f1"] = f1_score(
        y_test,
        preds,
        average="macro"
    )


    return results



if __name__ == "__main__":

    results = []

    for dataset in datasets:

        print("\n======================")
        print(dataset["name"])
        print("======================")


        X_train, y_train, X_test, y_test = load_dataset(dataset)


        metrics = run_baselines(
            X_train,
            y_train,
            X_test,
            y_test
        )


        row = {
            "dataset": dataset["name"],
            **metrics
        }

        results.append(row)


    df = pd.DataFrame(results)


    # Add average
    avg = {
        "dataset": "Average"
    }

    for col in df.columns:
        if col != "dataset":
            avg[col] = df[col].mean()


    df = pd.concat(
        [
            df,
            pd.DataFrame([avg])
        ],
        ignore_index=True
    )


    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True
    )


    df.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print("\nSaved:")
    print(OUTPUT_PATH)

    print(df)