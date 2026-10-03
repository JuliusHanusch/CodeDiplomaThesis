import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
from torch.nn.utils.rnn import pad_sequence
import copy

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DB_PATH="/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/similarity/similarity.db"

UCR_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018"
)

ARABIC_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Similarity/ArabicSpokenDigits"
)

class SiameseRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=128,
        embedding_dim=128,
        aggregation="last"
    ):
        super().__init__()

        if aggregation not in ["last", "average"]:
            raise ValueError(
                "aggregation must be 'last' or 'average'"
            )

        self.aggregation = aggregation

        self.rnn = nn.GRU(
            input_size,
            hidden_size,
            batch_first=True
        )

        self.projection = nn.Linear(
            hidden_size,
            embedding_dim
        )

    def encode(self, x, lengths):

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        if self.aggregation == "last":

            # SRN-L:
            # representation from the last valid timestep
            _, h = self.rnn(packed)
            representation = h[-1]

        else:

            # SRN-A:
            # representation from the average of all valid
            # hidden states
            output, _ = self.rnn(packed)

            output, _ = nn.utils.rnn.pad_packed_sequence(
                output,
                batch_first=True
            )

            mask = (
                torch.arange(
                    output.size(1),
                    device=output.device
                )[None, :]
                < lengths[:, None]
            )

            representation = (
                output * mask.unsqueeze(-1)
            ).sum(dim=1)

            representation = (
                representation /
                lengths.unsqueeze(-1)
            )

        return F.normalize(
            self.projection(representation),
            dim=-1
        )

    def forward(self, x1, x2, len1, len2):

        h1 = self.encode(x1, len1)
        h2 = self.encode(x2, len2)

        return h1 * h2


class SimilarityHead(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.fc=nn.Linear(dim,1)

    def forward(self,x):
        return self.fc(x).squeeze(-1)


class PairDataset(Dataset):

    def __init__(self,X1,X2,y):

        self.X1=X1
        self.X2=X2
        self.y=y


    def __len__(self):
        return len(self.y)


    def __getitem__(self, i):

        x1 = torch.tensor(
            self.X1[i],
            dtype=torch.float32
        )

        x2 = torch.tensor(
            self.X2[i],
            dtype=torch.float32
        )

        if x1.ndim == 1:
            x1 = x1.unsqueeze(-1)
            x2 = x2.unsqueeze(-1)

        return (
            x1,
            x2,
            torch.tensor(len(x1), dtype=torch.long),
            torch.tensor(len(x2), dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.float32)
        )


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

def load_pairs(path):

    data = np.load(
        path,
        allow_pickle=True
    )

    return (
        data["pairs_1"],
        data["pairs_2"],
        data["labels"]
    )

def normalize(X):
    X = np.asarray(X)

    if X.dtype == object:
        return normalize_sequences(X)

    X = np.nan_to_num(
        X,
        nan=np.nanmean(X),
        posinf=0,
        neginf=0
    )

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8

    return (X - mean) / std


def normalize_sequences(X):
    normalized = []

    for x in X:
        x = np.asarray(x, dtype=np.float32)

        x = np.nan_to_num(
            x,
            nan=np.nanmean(x),
            posinf=0,
            neginf=0
        )

        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True) + 1e-8

        normalized.append((x - mean) / std)

    return normalized

def siamese_collate(batch):

    x1 = [b[0] for b in batch]
    x2 = [b[1] for b in batch]

    len1 = torch.stack([b[2] for b in batch])
    len2 = torch.stack([b[3] for b in batch])

    labels = torch.stack([b[4] for b in batch])

    x1 = pad_sequence(
        x1,
        batch_first=True,
        padding_value=0.0
    )

    x2 = pad_sequence(
        x2,
        batch_first=True,
        padding_value=0.0
    )

    return x1, x2, len1, len2, labels



def getDatasets():
    datasets = []

    for dataset_dir in sorted(UCR_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        train_pairs = dataset_dir / "similarity/train_pairs.npz"
        eval_pairs = dataset_dir / "similarity/test_pairs.npz"


        datasets.append({
            "name": dataset_name,
            "train": str(train_pairs),
            "test": str(eval_pairs),
            "task": "similarity"
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
            "task": "digit"
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
            "task": "voice"
        },
        {
            "name": "ArabicSpokenDigits1_Multivariate",
            "train": str(
                ARABIC_ROOT
                / "similarity"
                / "multivariate"
                / "digit"
                / "train_pairs.npz"
            ),
            "test": str(
                ARABIC_ROOT
                / "similarity"
                / "multivariate"
                / "digit"
                / "test_pairs.npz"
            ),
            "task": "digit"
        },
        {
            "name": "ArabicSpokenDigits2_Multivariate",
            "train": str(
                ARABIC_ROOT
                / "similarity"
                / "multivariate"
                / "voice"
                / "train_pairs.npz"
            ),
            "test": str(
                ARABIC_ROOT
                / "similarity"
                / "multivariate"
                / "voice"
                / "test_pairs.npz"
            ),
            "task": "voice"
        },
    ])
    return datasets

@torch.no_grad()
def evaluate(
    encoder,
    head,
    X1,
    X2,
    y,
    batch_size=32,
    threshold=0.5
):
    encoder.eval()
    head.eval()

    dataset = PairDataset(X1, X2, y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=siamese_collate
    )

    probs = []

    for a, b, len_a, len_b, _ in loader:

        a, b = a.to(DEVICE), b.to(DEVICE)
        len_a, len_b = len_a.to(DEVICE), len_b.to(DEVICE)

        logits = head(
            encoder(a, b, len_a, len_b)
        )

        p = torch.sigmoid(logits).cpu().numpy()

        p = np.nan_to_num(
            p,
            nan=0.5,
            posinf=1.0,
            neginf=0.0
        )

        probs.extend(p)

    probs = np.asarray(probs)
    y = np.asarray(y)

    pred = (probs > threshold).astype(int)

    return {
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "auroc": (
            roc_auc_score(y, probs)
            if len(np.unique(y)) > 1
            else np.nan
        ),
        "probs": probs,
    }

def find_best_threshold(y_true, probs):

    thresholds = np.unique(
        np.concatenate([
            [0.0, 0.5, 1.0],
            probs
        ])
    )

    best_threshold = 0.5
    best_accuracy = -1.0

    for threshold in thresholds:

        pred = (probs > threshold).astype(int)

        accuracy = accuracy_score(
            y_true,
            pred
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy

def select_pairs(X1, X2, y, indices):
    X1_selected = [X1[i] for i in indices]
    X2_selected = [X2[i] for i in indices]
    y_selected = np.asarray(y)[indices]

    return X1_selected, X2_selected, y_selected

def train(
    X1, X2, y,
    X1_val=None, X2_val=None, y_val=None,
    batch_size=32,
    aggregation="last",
    epochs=40,
    eval_checkpoints=False
):

    sample = X1[0]

    if sample.ndim == 1:
        input_size = 1
    else:
        input_size = sample.shape[1]

    encoder = SiameseRNN(
        input_size=input_size,
        aggregation=aggregation
    ).to(DEVICE)

    head = SimilarityHead(128).to(DEVICE)

    generator = torch.Generator()
    generator.manual_seed(42)

    train_dataset = PairDataset(
        X1,
        X2,
        y
    )

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=siamese_collate,
        generator=generator
    )

    params = (
        list(encoder.parameters())
        + list(head.parameters())
    )

    opt = torch.optim.Adam(
        params,
        lr=1e-4
    )

    checkpoint_results = []

    checkpoints = [5, 10, 20, 40]

    for epoch in range(1, epochs + 1):

        encoder.train()
        head.train()

        loss_total = 0.0

        for a, b, len_a, len_b, l in loader:

            a, b = a.to(DEVICE), b.to(DEVICE)
            len_a, len_b = len_a.to(DEVICE), len_b.to(DEVICE)
            l = l.to(DEVICE)

            distance = encoder(
                a, b, len_a, len_b
            )

            logits = head(distance)

            loss = F.binary_cross_entropy_with_logits(
                logits,
                l
            )

            if torch.isnan(loss):
                continue

            opt.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                params,
                5.0
            )

            opt.step()

            loss_total += loss.item()

        print(
            f"Epoch {epoch}: "
            f"Loss={loss_total / len(loader):.4f}"
        )


        if (
            eval_checkpoints
            and epoch in checkpoints
            and X1_val is not None
        ):

            metrics = evaluate(
                encoder,
                head,
                X1_val,
                X2_val,
                y_val,
                batch_size=batch_size
            )

            probs = metrics.pop("probs")
            preds = (probs > 0.5).astype(int)

            metrics["accuracy"] = accuracy_score(y_val, preds)
            metrics["f1"] = f1_score(y_val, preds)

            metrics["epoch"] = epoch

            checkpoint_results.append(metrics)

            print(
                f"  Val Accuracy: {metrics['accuracy']:.4f}, "
                f"F1: {metrics['f1']:.4f}, "
                f"AUROC: {metrics['auroc']:.4f}, "
            )

    return encoder, head, checkpoint_results

if __name__ == "__main__":

    results = []

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    datasets = getDatasets()

    N_SPLITS = 5
    SEED = 42

    output_dir = Path(
        "/data/horse/ws/juha972b-AION-BERT-Chronos/"
        "BERTi/Results/Finetuning/Similarity"
    )

    all_fold_results = []
    all_test_results = []

    for info in datasets:

        dataset_name = info["name"]
        train_data = info["train"]
        test_data = info["test"]

        if "ArabicSpokenDigits" not in dataset_name:
            continue


        X1, X2, labels = load_pairs(train_data)

        print("\n===== Training Label Distribution =====")
        print(pd.Series(labels).value_counts().sort_index())

        print("\nRelative frequencies:")
        print(pd.Series(labels).value_counts(normalize=True).sort_index())



        if "ArabicSpokenDigits" in dataset_name:

            X1 = normalize_sequences(np.asarray(X1))
            X2 = normalize_sequences(np.asarray(X2))

        else:

            X1 = normalize(np.asarray(X1))
            X2 = normalize(np.asarray(X2))


        splits = list(
            create_cv_splits(
                labels,
                n_splits=N_SPLITS,
                seed=SEED
            )
        )

        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):


            X1_train, X2_train, y_train = select_pairs(
                X1, X2, labels, train_idx
            )

            X1_val, X2_val, y_val = select_pairs(
                X1, X2, labels, val_idx
            )

            for aggregation in ["last", "average"]:

                model_name = (
                    "SRN-L"
                    if aggregation == "last"
                    else "SRN-A"
                )

                print(f"\nTraining {model_name}")

                encoder, head, checkpoint_results = train(
                    X1_train,
                    X2_train,
                    y_train,
                    X1_val,
                    X2_val,
                    y_val,
                    batch_size=32,
                    aggregation=aggregation,
                    epochs=40,
                    eval_checkpoints=True
                )

                for metrics in checkpoint_results:

                    row = {
                        "dataset": dataset_name,
                        "fold": fold_idx + 1,
                        "model": model_name,
                        "epoch": metrics["epoch"],
                        "threshold": metrics["threshold"],
                        "accuracy": metrics["accuracy"],
                        "f1": metrics["f1"],
                        "auroc": metrics["auroc"],
                    }

                    fold_results.append(row)
                    all_fold_results.append(row)


        df = pd.DataFrame(fold_results)

        cv_summary = (
            df.groupby(["dataset", "model", "epoch"])
            .agg(
                accuracy_mean=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                f1_mean=("f1", "mean"),
                f1_std=("f1", "std"),
                auroc_mean=("auroc", "mean"),
                auroc_std=("auroc", "std"),
                threshold_mean=("threshold", "mean"),
            )
            .reset_index()
        )

        print("\n===== CV Summary =====")
        print(cv_summary)


        best_epochs = (
            cv_summary
            .sort_values("accuracy_mean", ascending=False)
            .drop_duplicates(["dataset", "model"])
            .copy()
        )

        print("\n===== Selected Epochs =====")
        print(
            best_epochs[
                [
                    "dataset",
                    "model",
                    "epoch",
                    "accuracy_mean",
                    "threshold_mean"
                ]
            ]
        )


        X1_test, X2_test, labels_test = load_pairs(test_data)

        if "ArabicSpokenDigits" in dataset_name:

            X1_test = normalize_sequences(np.asarray(X1_test))
            X2_test = normalize_sequences(np.asarray(X2_test))

        else:

            X1_test = normalize(np.asarray(X1_test))
            X2_test = normalize(np.asarray(X2_test))


        for _, row in best_epochs.iterrows():

            model_name = row["model"]
            best_epoch = int(row["epoch"])
            best_threshold = float(row["threshold_mean"])

            aggregation = (
                "last"
                if model_name == "SRN-L"
                else "average"
            )

            print(f"\n{'='*60}")
            print(f"Final Training: {model_name}")
            print(f"Selected Epoch: {best_epoch}")
            print(f"Selected Threshold: {best_threshold:.4f}")
            print(f"{'='*60}")

            # Train on the entire training set
            encoder, head, _ = train(
                X1,
                X2,
                labels,
                batch_size=32,
                aggregation=aggregation,
                epochs=best_epoch
            )

            # Evaluate on the independent test set
            test_metrics = evaluate(
                encoder,
                head,
                X1_test,
                X2_test,
                labels_test,
                batch_size=32,
                threshold=best_threshold
            )

            test_metrics.pop("probs")

            test_metrics["dataset"] = dataset_name
            test_metrics["model"] = model_name
            test_metrics["best_epoch"] = best_epoch
            test_metrics["threshold"] = best_threshold

            all_test_results.append(test_metrics)

            print(
                f"Test Accuracy: {test_metrics['accuracy']:.4f}, "
                f"F1: {test_metrics['f1']:.4f}, "
                f"AUROC: {test_metrics['auroc']:.4f}"
            )

    pd.DataFrame(all_fold_results).to_csv(
        output_dir / "siamese_rnn_cv_arabic_fold_results.csv",
        index=False
    )

    pd.DataFrame(all_test_results).to_csv(
        output_dir / "siamese_rnn_cv_arabic_test_results.csv",
        index=False
    )