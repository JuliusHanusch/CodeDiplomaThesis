from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from tslearn.metrics import dtw


RESULT = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/"
    "Results/Finetuning/Similarity/similarity_baseline_optimized.csv"
)

root_dir = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi"
)

sys.path.append(str(root_dir.resolve()))
sys.path.append(str((root_dir / "src").resolve()))


UCR_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/"
    "finetuning/UCR_extracted/UCRArchive_2018"
)

ARABIC_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/"
    "finetuning/Similarity/ArabicSpokenDigits"
)

VARIABLE_LENGTH_DATASETS = {
    "ArabicSpokenDigits1",
    "ArabicSpokenDigits2",
}

N_SPLITS = 5
SEED = 42

DTW_RADIUS_OPTIONS = [
    None,
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.50,
]


def create_cv_splits(y, n_splits=5, seed=42):

    y = np.asarray(y)

    rng = np.random.RandomState(seed)

    fold_val_indices = [[] for _ in range(n_splits)]

    for cls in np.unique(y):

        indices = np.where(y == cls)[0]

        rng.shuffle(indices)

        for i, idx in enumerate(indices):
            fold_val_indices[i % n_splits].append(idx)

    all_indices = np.arange(len(y))

    for fold in range(n_splits):

        val_idx = np.array(
            fold_val_indices[fold],
            dtype=int,
        )

        train_idx = np.setdiff1d(
            all_indices,
            val_idx,
        )

        yield train_idx, val_idx


def optimized_dtw_similarity_baseline(
    X1_train,
    X2_train,
    y_train,
    X1_test,
    X2_test,
    y_test,
    radius_options=DTW_RADIUS_OPTIONS,
    n_splits=N_SPLITS,
    seed=SEED,
):
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    splits = list(
        create_cv_splits(
            y_train,
            n_splits=n_splits,
            seed=seed,
        )
    )

    best_radius = None
    best_cv_auc = -np.inf

    for radius in radius_options:
        fold_aucs = []

        for fit_idx, val_idx in splits:
            fit_distances = calculate_pair_dtw_distances_optimized(
                X1_train[fit_idx],
                X2_train[fit_idx],
                radius,
            )

            val_distances = calculate_pair_dtw_distances_optimized(
                X1_train[val_idx],
                X2_train[val_idx],
                radius,
            )

            threshold, _ = find_best_threshold(
                fit_distances,
                y_train[fit_idx],
            )

            valid = (
                np.isfinite(val_distances)
                & np.isfinite(y_train[val_idx])
            )

            val_distances = val_distances[valid]
            y_val = y_train[val_idx][valid]

            if len(np.unique(y_val)) < 2:
                continue

            fold_aucs.append(
                roc_auc_score(y_val, -val_distances)
            )

        mean_auc = (
            float(np.mean(fold_aucs))
            if fold_aucs else np.nan
        )

        print(
            f"Radius={radius}, "
            f"CV AUROC={mean_auc:.4f}"
            if np.isfinite(mean_auc)
            else f"Radius={radius}, no valid CV AUROC"
        )

        if np.isfinite(mean_auc) and mean_auc > best_cv_auc:
            best_cv_auc = mean_auc
            best_radius = radius

    if not np.isfinite(best_cv_auc):
        raise ValueError("No valid DTW configuration found.")

    print(
        f"Selected radius={best_radius}, "
        f"CV AUROC={best_cv_auc:.4f}"
    )

    train_distances = calculate_pair_dtw_distances_optimized(
        X1_train,
        X2_train,
        best_radius,
    )

    threshold, train_accuracy = find_best_threshold(
        train_distances,
        y_train,
    )

    test_distances = calculate_pair_dtw_distances_optimized(
        X1_test,
        X2_test,
        best_radius,
    )

    valid_test = (
        np.isfinite(test_distances)
        & np.isfinite(y_test)
    )

    test_distances = test_distances[valid_test]
    y_test_valid = y_test[valid_test]

    y_pred = (test_distances < threshold).astype(np.int64)

    metrics = evaluate_binary(
        y_test_valid,
        y_pred,
        scores=-test_distances,
    )

    metrics["threshold"] = threshold
    metrics["train_accuracy"] = train_accuracy
    metrics["dtw_radius_fraction"] = (
        "unconstrained" if best_radius is None else best_radius
    )
    metrics["cv_auroc"] = best_cv_auc

    return metrics



def getDatasets():

    datasets = []



    for dataset_dir in sorted(UCR_ROOT.iterdir()):

        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name

        train_pairs = (
            dataset_dir
            / "similarity"
            / "train_pairs.npz"
        )

        test_pairs = (
            dataset_dir
            / "similarity"
            / "test_pairs.npz"
        )

        if not train_pairs.exists():
            print(
                f"WARNING: {dataset_name}: "
                f"train_pairs.npz not found -> skipped"
            )
            continue

        if not test_pairs.exists():
            print(
                f"WARNING: {dataset_name}: "
                f"test_pairs.npz not found -> skipped"
            )
            continue

        datasets.append({
            "name": dataset_name,
            "train": str(train_pairs),
            "test": str(test_pairs),
            "task": "similarity",
        })


    datasets.extend([
        {
            "name": "ArabicSpokenDigits1",
            "train": str(
                ARABIC_ROOT
                / "similarity"
                / "univariate"
                / "digit"
                / "train_pairs.npz"
            ),
            "test": str(
                ARABIC_ROOT
                / "similarity"
                / "univariate"
                / "digit"
                / "test_pairs.npz"
            ),
            "task": "digit",
        },
        {
            "name": "ArabicSpokenDigits2",
            "train": str(
                ARABIC_ROOT
                / "similarity"
                / "univariate"
                / "voice"
                / "train_pairs.npz"
            ),
            "test": str(
                ARABIC_ROOT
                / "similarity"
                / "univariate"
                / "voice"
                / "test_pairs.npz"
            ),
            "task": "voice",
        },
    ])

    return datasets


def load_pairs(path):

    data = np.load(
        path,
        allow_pickle=True
    )

    pairs_1 = data["pairs_1"]
    pairs_2 = data["pairs_2"]
    labels = data["labels"]

    return (
        pairs_1,
        pairs_2,
        labels,
    )

def calculate_pair_dtw_distances_optimized(
    X1,
    X2,
    radius_fraction,
):
    distances = []

    for x1, x2 in zip(X1, X2):
        x1 = np.asarray(x1, dtype=np.float64).flatten()
        x2 = np.asarray(x2, dtype=np.float64).flatten()

        x1 = x1[np.isfinite(x1)]
        x2 = x2[np.isfinite(x2)]

        if len(x1) == 0 or len(x2) == 0:
            distances.append(np.nan)
            continue

        if radius_fraction is None:
            distance = dtw(x1, x2)
        else:
            radius = max(
                abs(len(x1) - len(x2)),
                int(np.ceil(
                    radius_fraction * max(len(x1), len(x2))
                )),
            )

            distance = dtw(
                x1,
                x2,
                global_constraint="sakoe_chiba",
                sakoe_chiba_radius=radius,
            )

        distances.append(distance)

    return np.asarray(distances, dtype=np.float64)

def calculate_pair_dtw_distances(
    X1,
    X2,
):

    distances = []

    for x1, x2 in zip(X1, X2):

        x1 = np.asarray(
            x1,
            dtype=np.float64
        ).flatten()

        x2 = np.asarray(
            x2,
            dtype=np.float64
        ).flatten()

        x1 = x1[np.isfinite(x1)]
        x2 = x2[np.isfinite(x2)]

        if len(x1) == 0 or len(x2) == 0:
            distances.append(np.nan)
            continue

        distance = dtw(
            x1,
            x2
        )

        distances.append(
            distance
        )

    return np.asarray(
        distances,
        dtype=np.float64
    )


def dtw_similarity_baseline(
    X1_train,
    X2_train,
    y_train,
    X1_test,
    X2_test,
    y_test,
):

    train_distances = calculate_pair_dtw_distances(
        X1_train,
        X2_train,
    )

    threshold, train_accuracy = find_best_threshold(
        train_distances,
        y_train,
    )

    test_distances = calculate_pair_dtw_distances(
        X1_test,
        X2_test,
    )

    y_pred = (
        test_distances < threshold
    ).astype(
        np.int64
    )

    scores = -test_distances

    metrics = evaluate_binary(
        y_test,
        y_pred,
        scores,
    )

    metrics["threshold"] = threshold
    metrics["train_accuracy"] = train_accuracy

    return metrics

def evaluate_binary(
    y_true,
    y_pred,
    scores=None,
):

    result = {}

    result["accuracy"] = accuracy_score(
        y_true,
        y_pred,
    )

    result["f1"] = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    if scores is not None:

        try:

            result["auroc"] = roc_auc_score(
                y_true,
                scores,
            )

        except Exception:

            result["auroc"] = np.nan

    else:

        result["auroc"] = np.nan

    return result



def random_similarity_baseline(
    y_pairs,
):

    preds = np.random.randint(
        0,
        2,
        size=len(y_pairs),
    )

    return evaluate_binary(
        y_pairs,
        preds,
    )


def sequence_embedding(x):

    return np.asarray(
        x,
        dtype=np.float32,
    ).flatten()



VARIABLE_LENGTH_DATASETS = {
    "ArabicSpokenDigits1",
    "ArabicSpokenDigits2",
}


def resample_series(x, target_length=128):
    """
    Resample one time series to a fixed length.

    Only used for variable-length Arabic Spoken Digits datasets.
    """

    x = np.asarray(
        x,
        dtype=np.float64
    ).flatten()

    valid = np.isfinite(x)

    if not np.any(valid):
        return None

    x = x[valid]

    if len(x) == 0:
        return None

    if len(x) == target_length:
        return x

    old_positions = np.linspace(
        0.0,
        1.0,
        len(x)
    )

    new_positions = np.linspace(
        0.0,
        1.0,
        target_length
    )

    return np.interp(
        new_positions,
        old_positions,
        x
    )


def calculate_pair_distances(
    X1,
    X2,
    dataset_name,
    target_length=128,
):
    """
    Calculate the Euclidean distance between the two
    time series in every pair.

    For Arabic Spoken Digits:
        Both series are resampled to target_length because
        the original series can have different lengths.

    For all other datasets:
        The original series are used unchanged.

    NaN/inf values are handled pairwise.
    """

    distances = []

    for x1, x2 in zip(X1, X2):


        if dataset_name in VARIABLE_LENGTH_DATASETS:

            x1 = resample_series(
                x1,
                target_length
            )

            x2 = resample_series(
                x2,
                target_length
            )

            if x1 is None or x2 is None:
                distances.append(np.nan)
                continue

        else:

            x1 = np.asarray(
                x1,
                dtype=np.float64
            ).flatten()

            x2 = np.asarray(
                x2,
                dtype=np.float64
            ).flatten()


            if len(x1) != len(x2):

                raise ValueError(
                    f"{dataset_name}: "
                    "Time series in a pair have different lengths: "
                    f"{len(x1)} vs {len(x2)}"
                )


        valid = (
            np.isfinite(x1)
            &
            np.isfinite(x2)
        )

        if not np.any(valid):
            distances.append(np.nan)
            continue

        distance = np.linalg.norm(
            x1[valid] - x2[valid]
        )

        distances.append(
            distance
        )

    return np.asarray(
        distances,
        dtype=np.float64
    )

def find_best_threshold(
    distances,
    labels,
):

    """
    Find the distance threshold that maximizes
    classification accuracy on the training pairs.

    Smaller distance -> same label (1)
    Larger distance -> different label (0)

    Prediction:

        distance < threshold -> 1
        distance >= threshold -> 0
    """

    distances = np.asarray(
        distances,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    valid = (
        np.isfinite(distances)
        & np.isfinite(labels)
    )

    distances = distances[valid]
    labels = labels[valid]

    if len(distances) == 0:
        raise ValueError(
            "No valid training distances."
        )

    order = np.argsort(
        distances
    )

    sorted_distances = distances[order]
    sorted_labels = labels[order]


    n_positive = np.sum(
        sorted_labels == 1
    )

    n_negative = np.sum(
        sorted_labels == 0
    )

    best_correct = n_negative

    best_threshold = sorted_distances[0]

    correct = best_correct

    i = 0
    n = len(sorted_distances)

    while i < n:

        current_distance = sorted_distances[i]

        j = i

        while (
            j < n
            and sorted_distances[j] == current_distance
        ):
            j += 1

        group_labels = sorted_labels[i:j]

        group_positive = np.sum(
            group_labels == 1
        )

        group_negative = np.sum(
            group_labels == 0
        )


        correct += (
            group_positive
            - group_negative
        )

        if correct > best_correct:

            best_correct = correct


            if j < n:

                next_distance = sorted_distances[j]

                best_threshold = (
                    current_distance
                    + next_distance
                ) / 2.0

            else:

                best_threshold = (
                    current_distance
                    + 1e-8
                )

        i = j

    return (
        float(best_threshold),
        float(best_correct / len(distances)),
    )



def euclidean_similarity_baseline(
    dataset_name,
    X1_train,
    X2_train,
    y_train,
    X1_test,
    X2_test,
    y_test,
):


    train_distances = calculate_pair_distances(
        X1_train,
        X2_train,
        dataset_name,
        target_length = 128
    )


    threshold, train_accuracy = find_best_threshold(
        train_distances,
        y_train,
    )


    test_distances = calculate_pair_distances(
        X1_test,
        X2_test,
        dataset_name,
        target_length = 128
    )


    y_pred = (
        test_distances < threshold
    ).astype(
        np.int64
    )

    scores = -test_distances

    metrics = evaluate_binary(
        y_test,
        y_pred,
        scores,
    )

    metrics["threshold"] = threshold
    metrics["train_accuracy"] = train_accuracy

    return metrics



if __name__ == "__main__":

    results = []

    np.random.seed(42)

    datasets = getDatasets()

    print(
        f"\nFound {len(datasets)} datasets."
    )

    for info in datasets:

        dataset_name = info["name"]
        train_data = info["train"]
        test_data = info["test"]
        task = info["task"]

        print(dataset_name)


        X1_train, X2_train, labels_train = load_pairs(
            train_data
        )

        X1_test, X2_test, labels_test = load_pairs(
            test_data
        )

        metrics = random_similarity_baseline(
            labels_test
        )

        metrics["dataset"] = dataset_name
        metrics["task"] = task
        metrics["method"] = "Naive"

        results.append(
            metrics
        )

        metrics = euclidean_similarity_baseline(
            dataset_name,
            X1_train,
            X2_train,
            labels_train,
            X1_test,
            X2_test,
            labels_test,
        )

        metrics["dataset"] = dataset_name
        metrics["task"] = task
        metrics["method"] = "Euclidean"

        results.append(
            metrics
        )

        metrics = dtw_similarity_baseline(
            X1_train,
            X2_train,
            labels_train,
            X1_test,
            X2_test,
            labels_test,
        )


        metrics["dataset"] = dataset_name
        metrics["task"] = task
        metrics["method"] = "DTW"

        results.append(
            metrics
        )

        metrics = optimized_dtw_similarity_baseline(
            X1_train,
            X2_train,
            labels_train,
            X1_test,
            X2_test,
            labels_test,
        )

        metrics["dataset"] = dataset_name
        metrics["task"] = task
        metrics["method"] = "DTW Optimized"

        results.append(metrics)

        df = pd.DataFrame(results)
        df = df[
            [
                "dataset",
                "task",
                "method",
                "auroc",
                "accuracy",
                "f1",
                "dtw_radius_fraction",
                "cv_auroc",
            ]
        ]

        result_path = Path(RESULT)
        result_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(result_path, index=False)

        print(f"Saved {len(results)} results to {RESULT}")