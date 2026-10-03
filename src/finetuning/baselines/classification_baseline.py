import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from sklearn.ensemble import RandomForestClassifier


OUTPUT_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/Results/Finetuning/Classification/Baselines.csv"

DATA_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018"
)

EXCLUDED_DATASETS = {
    "Fungi",
}

def get_datasets():

    datasets = []

    for dataset_dir in sorted(DATA_ROOT.iterdir()):

        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name

        if dataset in EXCLUDED_DATASETS:
            print(f"Skipping {dataset}")
            continue

        train_path = dataset_dir / f"{dataset}_TRAIN.tsv"
        test_path = dataset_dir / f"{dataset}_TEST.tsv"

        if not train_path.exists():
            print(
                f"[WARNING] Missing train file for {dataset}: "
                f"{train_path}"
            )
            continue

        if not test_path.exists():
            print(
                f"[WARNING] Missing test file for {dataset}: "
                f"{test_path}"
            )
            continue

        datasets.append({
            "name": dataset,
            "train": str(train_path),
            "test": str(test_path),
        })

    return datasets


def load_ucr_tsv(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", header=None).values

    y = df[:, 0]
    X = df[:, 1:].astype(np.float32)

    y = y.astype(int)
    unique = np.unique(y)
    label_map = {v: i for i, v in enumerate(unique)}
    y = np.vectorize(label_map.get)(y)

    return X, y



def load_dataset(info):


    X_train, y_train = load_ucr_tsv(
        info["train"]
    )

    X_test, y_test = load_ucr_tsv(
        info["test"]
    )

    return X_train, y_train, X_test, y_test



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
    datasets = get_datasets()


    for dataset_info in datasets:

        dataset = dataset_info["name"]
        train_data = dataset_info["train"]
        test_data = dataset_info["test"]

        X_train, y_train = load_ucr_tsv(train_data)
        X_test, y_test = load_ucr_tsv(test_data)

        train_labels = np.unique(y_train)
        test_labels = np.unique(y_test)


        metrics = run_baselines(
            X_train,
            y_train,
            X_test,
            y_test
        )


        row = {
            "dataset": dataset_info["name"],
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