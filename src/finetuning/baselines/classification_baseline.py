import json
import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.metrics import accuracy_score


OUTPUT_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/Results/Finetuning/Classification/"
    "Baselines_Optimized.csv"
)

DATA_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/data/finetuning/UCR_extracted/UCRArchive_2018"
)

EXCLUDED_DATASETS = {
        "Fungi",
        "PigAirwayPressure",
        "PigArtPressure",
        "PigCVP",
    }

RANDOM_STATE = 42
N_JOBS = -1


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

        if not train_path.exists() or not test_path.exists():
            print(f"[WARNING] Missing data for {dataset}")
            continue

        datasets.append({
            "name": dataset,
            "train": str(train_path),
            "test": str(test_path),
        })

    return datasets


def load_ucr_tsv(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", header=None).values

    # Preserve the original class labels across train and test.
    y = df[:, 0].astype(int)
    X = df[:, 1:].astype(np.float32)

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return X, y

def create_cv_splits(y, n_splits=5, seed=42):
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    fold_val_indices = [[] for _ in range(n_splits)]

    for cls in np.unique(y):
        indices = np.where(y == cls)[0]
        rng.shuffle(indices)

        if len(indices) == 1:
            continue

        for i, idx in enumerate(indices):
            fold_val_indices[i % n_splits].append(idx)

    all_indices = np.arange(len(y))

    for fold in range(n_splits):
        val_idx = np.array(fold_val_indices[fold], dtype=int)
        train_idx = np.setdiff1d(all_indices, val_idx)

        yield train_idx, val_idx

def run_baselines(X_train, y_train, X_test, y_test):
    results = {}

    majority_class = Counter(y_train).most_common(1)[0][0]
    y_pred_naive = np.full(len(y_test), majority_class)

    results["Naive_accuracy"] = accuracy_score(y_test, y_pred_naive)

    knn_1 = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            KNeighborsClassifier(
                n_neighbors=1,
                metric="euclidean",
                weights="uniform",
                n_jobs=N_JOBS,
            ),
        ),
    ])

    knn_1.fit(X_train, y_train)
    y_pred_1nn = knn_1.predict(X_test)

    results["1NN_accuracy"] = accuracy_score(y_test, y_pred_1nn)

    cv = list(
        create_cv_splits(
            y_train,
            n_splits=5,
            seed=RANDOM_STATE,
        )
    )

    for fold, (train_idx, val_idx) in enumerate(cv, start=1):
        if len(val_idx) == 0:
            raise ValueError(f"Fold {fold} has an empty validation set")

        if len(train_idx) < 5:
            raise ValueError(
                f"Fold {fold} has fewer than five training samples; "
                "5-NN cannot be evaluated reliably."
            )

    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=1,
            ),
        ),
    ])

    knn_param_grid = {
        "scaler": [
            StandardScaler(),
            RobustScaler(),
            "passthrough",
        ],
        "model__n_neighbors": [1, 2, 3, 4, 5],
        "model__metric": [
            "euclidean",
            "manhattan",
        ],
        "model__weights": [
            "uniform",
            "distance",
        ],
    }

    knn_search = GridSearchCV(
        estimator=knn_pipeline,
        param_grid=knn_param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=N_JOBS,
        refit=True,
        error_score="raise",
    )

    knn_search.fit(X_train, y_train)

    y_pred_knn = knn_search.best_estimator_.predict(X_test)

    results["5NN_accuracy"] = accuracy_score(y_test, y_pred_knn)
    results["5NN_best_params"] = json.dumps(
        {
            key: (
                value
                if isinstance(value, (str, int, float, bool, type(None)))
                else value.__class__.__name__
            )
            for key, value in knn_search.best_params_.items()
        }
    )


    rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    rf_param_distributions = {
        "n_estimators": [10, 50, 100],
        "max_depth": [2, 3, 4, 5],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.5, 1.0],
    }

    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_param_distributions,
        n_iter=100,
        scoring="accuracy",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        refit=True,
        error_score="raise",
    )

    rf_search.fit(X_train, y_train)

    y_pred_rf = rf_search.best_estimator_.predict(X_test)

    results["RF_accuracy"] = accuracy_score(y_test, y_pred_rf)
    results["RF_best_params"] = json.dumps(
        rf_search.best_params_
    )

    return results


if __name__ == "__main__":

    results = []

    output_file = Path(OUTPUT_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    for info in get_datasets():

        dataset = info["name"]

        try:
            X_train, y_train = load_ucr_tsv(info["train"])
            X_test, y_test = load_ucr_tsv(info["test"])

            metrics = run_baselines(
                X_train,
                y_train,
                X_test,
                y_test,
            )

            row = {
                "dataset": dataset,
                **metrics,
            }

            results.append(row)

            print(
                f"{dataset}: "
                f"5-NN={metrics['5NN_accuracy']:.4f}, "
                f"RF-small={metrics['RF_accuracy']:.4f}"
            )

            # Save after every successfully evaluated dataset.
            df = pd.DataFrame(results)

            # Add average row to the saved CSV.
            avg = {"dataset": "Average"}

            for col in [
                "Naive_accuracy",
                "1NN_accuracy",
                "5NN_accuracy",
                "RF_accuracy",
            ]:
                avg[col] = df[col].mean()

            df = pd.concat(
                [df, pd.DataFrame([avg])],
                ignore_index=True,
            )

            df.to_csv(output_file, index=False)

            print(f"Saved progress: {output_file}")

        except Exception as exc:
            print(f"[ERROR] {dataset}: {exc}")

    if not results:
        raise RuntimeError("No datasets were evaluated.")

    print(f"\nFinished. Results saved to: {output_file}")
    print(df.to_string(index=False))
