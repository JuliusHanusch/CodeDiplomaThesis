from momentfm import MOMENTPipeline
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch.nn as nn
from sklearn.svm import SVC
from pathlib import Path
import warnings
import time

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn"
)


    
class UCRMomentDataset(Dataset):
    def __init__(self, train_path, test_path, split="train", seq_len=512):
        self.seq_len = seq_len

        train = pd.read_csv(train_path, sep="\t", header=None).values
        test = pd.read_csv(test_path, sep="\t", header=None).values

        train_labels = train[:, 0].astype(int)
        test_labels = test[:, 0].astype(int)

        unique = np.unique(train_labels)
        mapping = {l: i for i, l in enumerate(unique)}

        train_labels = np.vectorize(mapping.get)(train_labels)
        test_labels = np.vectorize(mapping.get)(test_labels)

        train_series = train[:, 1:].astype(np.float32)
        test_series = test[:, 1:].astype(np.float32)

        self.scaler = StandardScaler()

        self.scaler.fit(train_series)

        train_series = self.scaler.transform(train_series)
        test_series = self.scaler.transform(test_series)

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
    
def get_embedding(model, dataloader):
    embeddings, labels = [], []

    with torch.no_grad():

        for batch_x, batch_masks, batch_labels in tqdm(
            dataloader,
            total=len(dataloader)
        ):

            batch_x = batch_x.to("cuda").float()
            batch_masks = batch_masks.to("cuda")

            output = model(
                x_enc=batch_x,
                input_mask=batch_masks
            )

            embedding = output.embeddings

            embeddings.append(
                embedding.detach().cpu().numpy()
            )

            labels.append(
                batch_labels.cpu().numpy()
            )

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    return embeddings, labels

datasets = []

DATA_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/data/finetuning/UCR_extracted/UCRArchive_2018"
)

for dataset_dir in sorted(DATA_ROOT.iterdir()):

    if not dataset_dir.is_dir():
        continue

    dataset_name = dataset_dir.name

    train_path = dataset_dir / f"{dataset_name}_TRAIN.tsv"
    test_path = dataset_dir / f"{dataset_name}_TEST.tsv"

    if train_path.exists() and test_path.exists():

        datasets.append(
            {
                "name": dataset_name,
                "train": str(train_path),
                "test": str(test_path),
            }
        )

print(f"Found {len(datasets)} datasets")

results = []


for dataset_info in datasets:

    print("\n==============================")
    print(dataset_info["name"])
    print("==============================")

    name = dataset_info["name"]

    train_path = dataset_info["train"]

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

    start_time = time.perf_counter()


    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-base",
        model_kwargs={
            "task_name": "embedding",
        },
    )

    model.init()

    model.to("cuda").float()

    model.eval()

    train_embeddings, train_labels = get_embedding(
        model,
        train_loader
    )

    test_embeddings, test_labels = get_embedding(
        model,
        test_loader
    )

    print(train_embeddings.shape)
    print(test_embeddings.shape)

    from momentfm.models.statistical_classifiers import fit_svm

    clf = fit_svm(features=train_embeddings, y=train_labels)

    y_pred_train = clf.predict(train_embeddings)
    y_pred_test = clf.predict(test_embeddings)
    train_accuracy = clf.score(train_embeddings, train_labels)
    test_accuracy = clf.score(test_embeddings, test_labels)

    total_inference_time = time.perf_counter() - start_time


    print("train_accuracy", train_accuracy)
    print("test_accuracy", test_accuracy)

    results.append(
        {
            "dataset": name,
            "test_accuracy": test_accuracy,
            "inference_time": total_inference_time,

        }
    )

results_df = pd.DataFrame(results)

print("\nFinal Results")
print(results_df)

avg_accuracy = results_df["test_accuracy"].mean()
avg_time = results_df["inference_time"].mean()

print("\nAverage Test Accuracy:", avg_accuracy)

results_df = pd.concat(
    [
        results_df,
        pd.DataFrame(
            [{
                "dataset": "Average",
                "test_accuracy": avg_accuracy,
                "inference_time": avg_time,

            }]
        )
    ],
    ignore_index=True
)

results_df.to_csv(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/Results/Finetuning/Classification/moment_small_allData.csv",
    index=False
)

print("\nSaved results")