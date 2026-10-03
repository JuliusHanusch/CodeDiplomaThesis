import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

UCR_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/data/finetuning/UCR_extracted/UCRArchive_2018"
)

SEEDS = {
    "TRAIN_small": 1001,
    "TRAIN": 1002,
    "EVAL": 1003,
    "TEST": 1004,
}

SPLITS = [
    "TRAIN_small",
    "TRAIN",
    "EVAL",
    "TEST",
]


# ============================================================
# LOAD UCR DATA
# ============================================================

def load_ucr_tsv(path):
    """
    Load a UCR TSV file.

    Supports variable-length time series.
    Each row is:
        label <tab> value <tab> value <tab> ...

    Returns:
        X: list of np.ndarray
        y: np.ndarray
    """

    X = []
    y = []

    with open(path, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            values = line.split("\t")

            y.append(int(float(values[0])))

            series = np.asarray(
                [float(v) for v in values[1:] if v != ""],
                dtype=np.float32
            )

            X.append(series)

    # Normalize labels to 0 ... num_classes-1
    y = np.asarray(y)

    unique_labels = np.unique(y)

    label_map = {
        label: i
        for i, label in enumerate(unique_labels)
    }

    y = np.asarray(
        [label_map[label] for label in y],
        dtype=np.int64
    )

    return X, y


# ============================================================
# CREATE PAIRS
# ============================================================

def create_pairs(
    X,
    y,
    n_pairs=None,
    seed=None
):

    if n_pairs is None:
        n_pairs = len(X)

    rng = np.random.default_rng(seed)

    pairs_1 = []
    pairs_2 = []
    labels = []

    classes = np.unique(y)

    while len(labels) < n_pairs:

        # ----------------------------------------------------
        # Positive pair
        # ----------------------------------------------------

        if rng.random() < 0.5:

            c = rng.choice(classes)

            idx = np.where(y == c)[0]

            # Cannot create a positive pair
            if len(idx) < 2:
                continue

            i, j = rng.choice(
                idx,
                size=2,
                replace=False
            )

            label = 1

        # ----------------------------------------------------
        # Negative pair
        # ----------------------------------------------------

        else:

            i = rng.integers(len(X))

            different_class_indices = np.where(
                y != y[i]
            )[0]

            # No other class available
            if len(different_class_indices) == 0:
                continue

            j = rng.choice(
                different_class_indices
            )

            label = 0

        pairs_1.append(X[i])
        pairs_2.append(X[j])
        labels.append(label)

    return (
        pairs_1,
        pairs_2,
        np.asarray(labels, dtype=np.int64)
    )

def save_pairs(output_path, pairs_1, pairs_2, labels):
    pairs_1_obj = np.empty(len(pairs_1), dtype=object)
    pairs_2_obj = np.empty(len(pairs_2), dtype=object)

    for i, x in enumerate(pairs_1):
        pairs_1_obj[i] = np.asarray(x, dtype=np.float32)

    for i, x in enumerate(pairs_2):
        pairs_2_obj[i] = np.asarray(x, dtype=np.float32)

    np.savez_compressed(
        output_path,
        pairs_1=pairs_1_obj,
        pairs_2=pairs_2_obj,
        labels=np.asarray(labels, dtype=np.int64),
    )


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    dataset_dir,
    split_name
):

    dataset_name = dataset_dir.name

    tsv_path = (
        dataset_dir /
        f"{dataset_name}_{split_name}.tsv"
    )

    if not tsv_path.exists():
        print(
            f"  Skipping {split_name}: "
            f"{tsv_path.name} not found"
        )
        return

    similarity_dir = (
        dataset_dir /
        "similarity"
    )

    similarity_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_name = {
        "TRAIN_small": "train_small_pairs.npz",
        "TRAIN": "train_pairs.npz",
        "EVAL": "eval_pairs.npz",
        "TEST": "test_pairs.npz",
    }[split_name]

    output_path = similarity_dir / output_name

    print()
    print(
        f"{dataset_name} / {split_name}"
    )
    print(
        f"  Input: {tsv_path}"
    )

    X, y = load_ucr_tsv(tsv_path)

    print(
        f"  Samples: {len(X):,}"
    )

    print(
        f"  Classes: {len(np.unique(y))}"
    )

    # Exactly len(X) pairs
    pairs_1, pairs_2, labels = create_pairs(
        X=X,
        y=y,
        n_pairs=None,
        seed=SEEDS[split_name]
    )

    assert len(pairs_1) == len(X)
    assert len(pairs_2) == len(X)
    assert len(labels) == len(X)

    save_pairs(
        output_path,
        pairs_1,
        pairs_2,
        labels
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    dataset_dirs = sorted(
        p for p in UCR_ROOT.iterdir()
        if p.is_dir()
    )

    print(
        f"Found {len(dataset_dirs)} datasets"
    )

    for dataset_dir in dataset_dirs:

        for split_name in SPLITS:

            process_split(
                dataset_dir,
                split_name
            )

    print()
    print("Done.")
