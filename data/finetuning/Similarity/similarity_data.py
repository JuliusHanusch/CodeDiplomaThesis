from pathlib import Path
import requests
import numpy as np
import zipfile


URL = (
    "https://archive.ics.uci.edu/static/public/195/"
    "spoken+arabic+digit.zip"
)

ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/data/finetuning/Similarity/ArabicSpokenDigits"
)

ROOT.mkdir(exist_ok=True)


def download():

    zip_path = ROOT / "spoken_arabic_digit.zip"

    if zip_path.exists():
        return zip_path

    print("Downloading dataset...")

    r = requests.get(URL)
    r.raise_for_status()

    with open(zip_path, "wb") as f:
        f.write(r.content)

    return zip_path


def unzip(zip_path):

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(ROOT)


def load_file(path, train=True):

    with open(path) as f:
        lines = f.readlines()


    sequences = []
    current = []

    for line in lines:

        line = line.strip()

        if line == "":

            if current:
                sequences.append(
                    np.asarray(
                        current,
                        dtype=np.float32
                    )
                )
                current=[]

        else:
            current.append(
                [
                    float(x)
                    for x in line.split()
                ]
            )


    if current:
        sequences.append(
            np.asarray(
                current,
                dtype=np.float32
            )
        )


    digit_labels=[]
    speaker_labels=[]


    utterances_per_digit = 660 if train else 220


    for idx in range(len(sequences)):

        digit = idx // utterances_per_digit

        within_digit = idx % utterances_per_digit

        speaker = within_digit // 10


        if train:
            speaker_id=speaker
        else:
            speaker_id=speaker+66


        digit_labels.append(digit)
        speaker_labels.append(speaker_id)


    return (
        sequences,
        np.array(digit_labels),
        np.array(speaker_labels)
    )


def create_univariate(sequences):

    """
    Convert:
        (time,13)

    into:
        (time,)

    using MFCC coefficient 0
    """

    return [
        seq[:,0].astype(np.float32)
        for seq in sequences
    ]


def create_multivariate(sequences):

    """
    Keep original MFCC representation

    (time,13)
    """

    return [
        seq.astype(np.float32)
        for seq in sequences
    ]
def split_similarity_dataset(
    sequences,
    digit_labels,
    speaker_labels,
    train_ratio=0.8,
    seed=42
):

    rng = np.random.default_rng(seed)

    indices = np.arange(
        len(sequences)
    )

    rng.shuffle(indices)

    split = int(
        len(indices) * train_ratio
    )

    train_idx = indices[:split]
    test_idx = indices[split:]

    return (
        [sequences[i] for i in train_idx],
        digit_labels[train_idx],
        speaker_labels[train_idx],

        [sequences[i] for i in test_idx],
        digit_labels[test_idx],
        speaker_labels[test_idx]
    )


def save_dataset(
        filename,
        sequences,
        digit_labels,
        speaker_labels
):

    np.savez(
        ROOT / filename,

        X=np.array(
            sequences,
            dtype=object
        ),

        digit_labels=digit_labels,

        speaker_labels=speaker_labels
    )



def main():

    zip_path=download()

    unzip(zip_path)


    train_sequences_uci, train_digits_uci, train_speakers_uci = load_file(
        ROOT / "Train_Arabic_Digit.txt",
        train=True
    )

    test_sequences_uci, test_digits_uci, test_speakers_uci = load_file(
        ROOT / "Test_Arabic_Digit.txt",
        train=False
    )

    sequences = (
        train_sequences_uci
        +
        test_sequences_uci
    )

    digit_labels = np.concatenate(
        [
            train_digits_uci,
            test_digits_uci
        ]
    )

    speaker_labels = np.concatenate(
        [
            train_speakers_uci,
            test_speakers_uci
        ]
    )

    train_sequences, train_digits, train_speakers, test_sequences, test_digits, test_speakers = split_similarity_dataset(
        sequences,
        digit_labels,
        speaker_labels
    )


    save_dataset(
        "arabic_digits_multivariate_train.npz",
        create_multivariate(train_sequences),
        train_digits,
        train_speakers
    )

    save_dataset(
        "arabic_digits_multivariate_test.npz",
        create_multivariate(test_sequences),
        test_digits,
        test_speakers
    )


    save_dataset(
        "arabic_digits_univariate_train.npz",
        create_univariate(train_sequences),
        train_digits,
        train_speakers
    )

    save_dataset(
        "arabic_digits_univariate_test.npz",
        create_univariate(test_sequences),
        test_digits,
        test_speakers
    )


    print("Finished")

    print(
        "Example multivariate:",
        train_sequences[0].shape
    )

    print(
        "Example univariate:",
        create_univariate(train_sequences)[0].shape
    )


if __name__=="__main__":
    main()