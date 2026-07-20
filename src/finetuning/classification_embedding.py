#from momentfm import MOMENTPipeline
from gluonts.dataset.arrow import ArrowFile
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from gluonts.dataset.arrow import ArrowFile
import pandas as pd
import torch.nn as nn
from sklearn.svm import SVC
import sys
from pathlib import Path

root_dir = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi")
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"src").resolve()))  
sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos import ChronosPipeline

import warnings


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn"
)

class UCIHARDataset(Dataset):
    def __init__(self, train_path, test_path, split="train", seq_len=512):

        self.seq_len = seq_len

        X_train = np.loadtxt(train_path)
        y_train = np.loadtxt(
            train_path.replace("X_train.txt", "y_train.txt")
        ).astype(int)

        X_test = np.loadtxt(test_path)
        y_test = np.loadtxt(
            test_path.replace("X_test.txt", "y_test.txt")
        ).astype(int)

        # UCI labels 1-6 -> 0-5
        y_train = y_train - 1
        y_test = y_test - 1

        # Same as UCR
        scaler = StandardScaler()

        scaler.fit(
            X_train.reshape(-1, 1)
        )

        X_train = scaler.transform(
            X_train.reshape(-1, 1)
        ).reshape(X_train.shape)

        X_test = scaler.transform(
            X_test.reshape(-1, 1)
        ).reshape(X_test.shape)


        if split == "train":
            self.series = X_train.astype(np.float32)
            self.labels = y_train

        else:
            self.series = X_test.astype(np.float32)
            self.labels = y_test


    def __len__(self):
        return len(self.series)


    def __getitem__(self, idx):

        ts = self.series[idx]

        input_mask = np.ones(
            self.seq_len,
            dtype=np.float32
        )

        # same padding as UCR
        if len(ts) < self.seq_len:

            pad = self.seq_len - len(ts)

            ts = np.pad(
                ts,
                (pad, 0)
            )

            input_mask[:pad] = 0


        elif len(ts) > self.seq_len:

            ts = ts[-self.seq_len:]


        return (
            torch.tensor(ts)
                 .float()
                 .unsqueeze(0),

            torch.tensor(input_mask)
                 .float(),

            torch.tensor(
                self.labels[idx]
            ).long()
        )
class UCRMomentDataset(Dataset):
    def __init__(self, train_path, test_path, split="train", seq_len=512):
        self.seq_len = seq_len

        train = pd.read_csv(train_path, sep="\t", header=None).values
        test = pd.read_csv(test_path, sep="\t", header=None).values

        train_labels = train[:, 0].astype(int)
        test_labels = test[:, 0].astype(int)

        # EXACTLY like MOMENT
        unique = np.unique(train_labels)
        mapping = {l: i for i, l in enumerate(unique)}

        train_labels = np.vectorize(mapping.get)(train_labels)
        test_labels = np.vectorize(mapping.get)(test_labels)

        train_series = train[:, 1:].astype(np.float32)
        test_series = test[:, 1:].astype(np.float32)

        self.scaler = StandardScaler()

        self.scaler.fit(train_series.reshape(-1, 1))

        train_series = self.scaler.transform(
            train_series.reshape(-1, 1)
        ).reshape(train_series.shape)

        test_series = self.scaler.transform(
            test_series.reshape(-1, 1)
        ).reshape(test_series.shape)

        if split == "train":
            self.series = train_series
            self.labels = train_labels
        else:
            self.series = test_series
            self.labels = test_labels

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):

        ts = self.series[idx]

        input_mask = np.ones(self.seq_len, dtype=np.float32)

        if len(ts) < self.seq_len:
            pad = self.seq_len - len(ts)
            ts = np.pad(ts, (pad, 0))
            input_mask[:pad] = 0

        elif len(ts) > self.seq_len:
            ts = ts[-self.seq_len:]

        return (
            torch.tensor(ts).float().unsqueeze(0),
            torch.tensor(input_mask).float(),
            torch.tensor(self.labels[idx]).long(),
        )
    
def get_embedding(model, tokenizer, dataloader):

    embeddings = []
    labels_all = []

    with torch.no_grad():

        for batch_x, batch_masks, batch_labels in tqdm(
            dataloader,
            total=len(dataloader)
        ):

            batch_x = batch_x.float()

            # Chronos tokenizer expects CPU tensors
            context = batch_x.squeeze(1)

            input_ids, attention_mask, _ = (
                tokenizer.context_input_transform(context)
            )

            # Only model inputs go to GPU
            input_ids = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")


            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states[-1]


            embedding = (
                hidden_states * attention_mask.unsqueeze(-1)
            ).sum(dim=1) / attention_mask.sum(
                dim=1,
                keepdim=True
            )


            embeddings.append(
                embedding.cpu().numpy()
            )

            labels_all.append(
                batch_labels.numpy()
            )


    embeddings = np.concatenate(embeddings)
    labels_all = np.concatenate(labels_all)

    return embeddings, labels_all

results = []

datasets = [
    # EXAMPLE:
        {
        "name": "UCI-HAR",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCI_HAR/UCI_HAR.arrow",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCI_HAR/UCI HAR Dataset/test",
        "labels": 6
    },
    {
        "name": "ArrowHead",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_arrow/ArrowHead_train.arrow",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/ArrowHead/ArrowHead_TEST.tsv",
        "labels": 3
    },
    {
        "name": "DistalPhalanxTW",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_arrow/DistalPhalanxTW_train.arrow",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/DistalPhalanxTW/DistalPhalanxTW_TEST.tsv",
        "labels": 6
    },
    {
        "name": "GestureMidAirD2",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_arrow/GestureMidAirD2_train.arrow",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/GestureMidAirD2/GestureMidAirD2_TEST.tsv",
        "labels": 26
    },
    {
        "name": "Wafer",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_arrow/Wafer_train.arrow",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/Wafer/Wafer_TEST.tsv",
        "labels": 2
    },
]

for dataset_info in datasets:

    print("\n==============================")
    print(dataset_info["name"])
    print("==============================")

    name = dataset_info["name"]


    # -------------------------------
    # Dataset loading
    # -------------------------------

    if name == "UCI-HAR":

        train_dataset = UCIHARDataset(
            train_path="/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCI_HAR/UCI HAR Dataset/train/X_train.txt",
            test_path="/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCI_HAR/UCI HAR Dataset/test/X_test.txt",
            split="train",
        )


        test_dataset = UCIHARDataset(
            train_path="/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCI_HAR/UCI HAR Dataset/train/X_train.txt",
            test_path="/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCI_HAR/UCI HAR Dataset/test/X_test.txt",
            split="test",
        )

    else:

        train_path = dataset_info["train"]

        # Convert arrow path to TRAIN.tsv path
        test_path = dataset_info["test"]

        train_tsv = test_path.replace(
            "_TEST.tsv",
            "_TRAIN.tsv"
        )

        train_dataset = UCRMomentDataset(
            train_path=train_tsv,
            test_path=test_path,
            split="train",
        )

        test_dataset = UCRMomentDataset(
            train_path=train_tsv,
            test_path=test_path,
            split="test",
        )


    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
    )


    pipeline = ChronosPipeline.from_pretrained(
        "juliushanusch/ChronosBERT-Optimized",
    )

    model = pipeline.model
    tokenizer = pipeline.tokenizer

    model.to("cuda")
    model.eval()

    train_embeddings, train_labels = get_embedding(
        model,
        tokenizer,
        train_loader
    )


    test_embeddings, test_labels = get_embedding(
        model,
        tokenizer,
        test_loader
    )

    print(train_embeddings.shape)
    print(test_embeddings.shape)

    clf = SVC(
    kernel="rbf"
    )

    clf.fit(
        train_embeddings,
        train_labels
    )


    train_accuracy = clf.score(
        train_embeddings,
        train_labels
    )

    test_accuracy = clf.score(
        test_embeddings,
        test_labels
    )

    print("train_accuracy", train_accuracy)
    print("test_accuracy", test_accuracy)

    results.append(
        {
            "dataset": name,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_samples": len(train_labels),
            "test_samples": len(test_labels),
            "embedding_dim": train_embeddings.shape[1],
        }
    )

results_df = pd.DataFrame(results)

print("\nFinal Results")
print(results_df)

results_df.to_csv(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/chronos_embedding.csv",
    index=False
)

print(
    "\nSaved results to chronos_embedding.csv"
)