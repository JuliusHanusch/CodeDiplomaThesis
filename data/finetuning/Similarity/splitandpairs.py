import numpy as np
from pathlib import Path


DATASET_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/data/finetuning/Similarity/ArabicSpokenDigits"
)

TRAIN_SMALL_RATIO = 0.8

SPLIT_SEED = 2000

PAIR_SEEDS = {
    "digit": 3000,
    "voice": 4000,
}

# None -> exactly len(X) pairs
N_PAIRS = None


def load_npz(path):
    data = np.load(path, allow_pickle=True)

    return (
        list(data["X"]),
        np.asarray(data["digit_labels"]),
        np.asarray(data["speaker_labels"]),
    )


def save_npz(path, X, digit_labels, speaker_labels):

    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=object),
        digit_labels=np.asarray(digit_labels),
        speaker_labels=np.asarray(speaker_labels),
    )


def create_train_small_and_eval():

    uni_train_path = (
        DATASET_ROOT /
        "arabic_digits_univariate_train.npz"
    )

    multi_train_path = (
        DATASET_ROOT /
        "arabic_digits_multivariate_train.npz"
    )

    X_uni, digit_uni, speaker_uni = load_npz(
        uni_train_path
    )

    X_multi, digit_multi, speaker_multi = load_npz(
        multi_train_path
    )


    assert len(X_uni) == len(X_multi), (
        f"Uni/multi sample count mismatch: "
        f"{len(X_uni)} vs {len(X_multi)}"
    )

    assert np.array_equal(
        digit_uni,
        digit_multi
    ), "Digit labels differ between uni and multi."

    assert np.array_equal(
        speaker_uni,
        speaker_multi
    ), "Speaker labels differ between uni and multi."


    rng = np.random.default_rng(SPLIT_SEED)

    indices = np.arange(len(X_uni))
    rng.shuffle(indices)

    n_train_small = int(
        len(indices) * TRAIN_SMALL_RATIO
    )

    train_small_idx = indices[:n_train_small]
    eval_idx = indices[n_train_small:]


    save_npz(
        DATASET_ROOT /
        "train_small" /
        "arabic_digits_univariate_train_small.npz",

        [X_uni[i] for i in train_small_idx],
        digit_uni[train_small_idx],
        speaker_uni[train_small_idx],
    )

    save_npz(
        DATASET_ROOT /
        "eval" /
        "arabic_digits_univariate_eval.npz",

        [X_uni[i] for i in eval_idx],
        digit_uni[eval_idx],
        speaker_uni[eval_idx],
    )

    save_npz(
        DATASET_ROOT /
        "train_small" /
        "arabic_digits_multivariate_train_small.npz",

        [X_multi[i] for i in train_small_idx],
        digit_multi[train_small_idx],
        speaker_multi[train_small_idx],
    )

    save_npz(
        DATASET_ROOT /
        "eval" /
        "arabic_digits_multivariate_eval.npz",

        [X_multi[i] for i in eval_idx],
        digit_multi[eval_idx],
        speaker_multi[eval_idx],
    )

    print("\nCreated train_small and eval splits.")

def create_pairs(
    X,
    y,
    n_pairs=None,
    seed=None,
):

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

            # Cannot create positive pair
            # if class contains only one sample
            if len(idx) < 2:
                continue

            i, j = rng.choice(
                idx,
                size=2,
                replace=False,
            )

            label = 1


        else:

            i = rng.integers(len(X))

            different = np.where(
                y != y[i]
            )[0]

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


def generate_pairs(
    input_path,
    output_path,
    task,
):

    X, digit_labels, speaker_labels = load_npz(
        input_path
    )

    if task == "digit":

        y = digit_labels

    elif task == "voice":

        y = speaker_labels

    else:

        raise ValueError(
            f"Unknown task: {task}"
        )


    pairs_1, pairs_2, labels = create_pairs(
        X,
        y,
        n_pairs=N_PAIRS,
        seed=PAIR_SEEDS[task],
    )



    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            DATASET_ROOT /
            "arabic_digits_univariate_train.npz",
        ),

        (
            "univariate",
            "train_small",
            DATASET_ROOT /
            "train_small" /
            "arabic_digits_univariate_train_small.npz",
        ),

        (
            "univariate",
            "eval",
            DATASET_ROOT /
            "eval" /
            "arabic_digits_univariate_eval.npz",
        ),

        (
            "univariate",
            "test",
            DATASET_ROOT /
            "arabic_digits_univariate_test.npz",
        ),


        (
            "multivariate",
            "train",
            DATASET_ROOT /
            "arabic_digits_multivariate_train.npz",
        ),

        (
            "multivariate",
            "train_small",
            DATASET_ROOT /
            "train_small" /
            "arabic_digits_multivariate_train_small.npz",
        ),

        (
            "multivariate",
            "eval",
            DATASET_ROOT /
            "eval" /
            "arabic_digits_multivariate_eval.npz",
        ),

        (
            "multivariate",
            "test",
            DATASET_ROOT /
            "arabic_digits_multivariate_test.npz",
        ),
    ]


def generate_all_pairs():

    similarity_dir = DATASET_ROOT / "similarity"

    for representation, split, input_path in get_splits():

        if not input_path.exists():

            raise FileNotFoundError(
                f"Missing dataset:\n{input_path}"
            )

        for task in ["digit", "voice"]:

            output_path = (
                similarity_dir /
                representation /
                task /
                f"{split}_pairs.npz"
            )

            generate_pairs(
                input_path=input_path,
                output_path=output_path,
                task=task,
            )



if __name__ == "__main__":


    create_train_small_and_eval()

    generate_all_pairs()

    print("\nDone.")