import numpy as np
from pathlib import Path


DATASET_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/data/finetuning/Similarity/ArabicSpokenDigits"
)

PAIR_SEEDS = {
    "digit": 3000,
    "voice": 4000,
}

N_PAIRS = None


def load_npz(path):
    data = np.load(path, allow_pickle=True)

    return (
        list(data["X"]),
        np.asarray(data["digit_labels"]),
        np.asarray(data["speaker_labels"]),
    )


def create_pairs(X, y, n_pairs=None, seed=None):
    if n_pairs is None:
        n_pairs = len(X)

    rng = np.random.default_rng(seed)

    pairs_1 = []
    pairs_2 = []
    labels = []

    classes = np.unique(y)

    while len(labels) < n_pairs:
        if rng.random() < 0.5:
            c = rng.choice(classes)
            idx = np.where(y == c)[0]

            if len(idx) < 2:
                continue

            i, j = rng.choice(idx, size=2, replace=False)
            label = 1

        else:
            i = rng.integers(len(X))

            different = np.where(y != y[i])[0]

            if len(different) == 0:
                continue

            j = rng.choice(different)
            label = 0

        pairs_1.append(X[i])
        pairs_2.append(X[j])
        labels.append(label)

    return (
        np.asarray(pairs_1, dtype=object),
        np.asarray(pairs_2, dtype=object),
        np.asarray(labels, dtype=np.int64),
    )


def generate_pairs(input_path, output_path, task):
    X, digit_labels, speaker_labels = load_npz(input_path)

    if task == "digit":
        y = digit_labels
    elif task == "voice":
        y = speaker_labels
    else:
        raise ValueError(f"Unknown task: {task}")

    pairs_1, pairs_2, labels = create_pairs(
        X,
        y,
        n_pairs=N_PAIRS,
        seed=PAIR_SEEDS[task],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        pairs_1=pairs_1,
        pairs_2=pairs_2,
        labels=labels,
    )

    n_positive = np.sum(labels == 1)
    n_negative = np.sum(labels == 0)

    print(
        f"{task:5s} | "
        f"{input_path.name:55s} | "
        f"{len(labels):6d} pairs | "
        f"positive={n_positive:6d} | "
        f"negative={n_negative:6d}"
    )


def get_splits():
    return [
        (
            "univariate",
            "train",
            DATASET_ROOT / "arabic_digits_univariate_train.npz",
        ),
        (
            "univariate",
            "test",
            DATASET_ROOT / "arabic_digits_univariate_test.npz",
        ),
        (
            "multivariate",
            "train",
            DATASET_ROOT / "arabic_digits_multivariate_train.npz",
        ),
        (
            "multivariate",
            "test",
            DATASET_ROOT / "arabic_digits_multivariate_test.npz",
        ),
    ]


def generate_all_pairs():
    similarity_dir = DATASET_ROOT / "similarity"

    for representation, split, input_path in get_splits():
        if not input_path.exists():
            raise FileNotFoundError(f"Missing dataset:\n{input_path}")

        for task in ["digit", "voice"]:
            output_path = (
                similarity_dir
                / representation
                / task
                / f"{split}_pairs.npz"
            )

            generate_pairs(
                input_path=input_path,
                output_path=output_path,
                task=task,
            )


if __name__ == "__main__":
    generate_all_pairs()
    print("\nDone.")