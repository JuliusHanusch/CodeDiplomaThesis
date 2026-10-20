import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from sklearn.model_selection import ParameterSampler
from scipy.stats import loguniform

import argparse


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


UCR_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/data/finetuning/UCR_extracted/UCRArchive_2018"
)

ARABIC_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/data/finetuning/Similarity/ArabicSpokenDigits"
)

OUTPUT_DIR = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/Results/Finetuning/Similarity"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


HPO_RESULTS_PATH = (
    OUTPUT_DIR / "siamese_rnn_hpo.csv"
)

TEST_RESULTS_PATH = (
    OUTPUT_DIR / "siamese_rnn_hpo_test_results.csv"
)

BEST_CONFIGS_PATH = (
    OUTPUT_DIR / "siamese_rnn_hpo_best_configs.csv"
)

# Experiment settings
N_SPLITS = 5
N_TRIALS = 100
SEED = 42

RESUME = True


HPO_SPACE = {
    "learning_rate": loguniform(1e-5, 1e-3),
    "hidden_size": [64, 128, 256],
    "embedding_dim": [64, 128, 256],
    "aggregation": ["last", "average"],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "batch_size": [16, 32, 64],
    "epochs": [10, 20, 40],
}




def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SiameseRNN(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=128,
        embedding_dim=128,
        aggregation="last",
        dropout=0.0,
    ):
        super().__init__()

        if aggregation not in ["last", "average"]:
            raise ValueError(
                "aggregation must be 'last' or 'average'"
            )

        self.aggregation = aggregation

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.Dropout(dropout),
        )

    def encode(self, x, lengths):

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        if self.aggregation == "last":

            _, h = self.rnn(packed)
            representation = h[-1]

        else:

            output, _ = self.rnn(packed)

            output, _ = nn.utils.rnn.pad_packed_sequence(
                output,
                batch_first=True,
            )

            mask = (
                torch.arange(
                    output.size(1),
                    device=output.device,
                )[None, :]
                < lengths[:, None]
            )

            representation = (
                output * mask.unsqueeze(-1)
            ).sum(dim=1)

            representation = (
                representation / lengths.unsqueeze(-1)
            )

        embedding = self.projection(representation)

        normalized = F.normalize(embedding, dim=-1)

        if not self.training:

            def print_diagnostics(name, tensor):

                tensor = tensor.detach().float()

                print(f"\n--- {name} ---")
                print("Mean:", tensor.mean().item())
                print("Std:", tensor.std().item())

                if tensor.size(0) > 1:

                    normalized_tensor = F.normalize(
                        tensor, dim=-1
                    )

                    similarities = (
                        normalized_tensor @ normalized_tensor.T
                    )

                    n = similarities.size(0)
                    mask = ~torch.eye(
                        n,
                        dtype=torch.bool,
                        device=similarities.device,
                    )

                    similarities = similarities[mask]

                    print(
                        "Mean pairwise cosine:",
                        similarities.mean().item(),
                    )
                    print(
                        "Min pairwise cosine:",
                        similarities.min().item(),
                    )
                    print(
                        "Max pairwise cosine:",
                        similarities.max().item(),
                    )

            print_diagnostics("GRU representation", representation)
            print_diagnostics("Projection output", embedding)
            print_diagnostics("Normalized embedding", normalized)

        return normalized

    def forward(self, x1, x2, len1, len2):

        h1 = self.encode(x1, len1)
        h2 = self.encode(x2, len2)

        return h1 * h2


class SimilarityHead(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.fc = nn.Linear(dim, 1)

    def forward(self, x):

        return self.fc(x).squeeze(-1)



class PairDataset(Dataset):

    def __init__(self, X1, X2, y):

        self.X1 = X1
        self.X2 = X2
        self.y = y

    def __len__(self):

        return len(self.y)

    def __getitem__(self, i):

        x1 = torch.tensor(
            self.X1[i],
            dtype=torch.float32,
        )

        x2 = torch.tensor(
            self.X2[i],
            dtype=torch.float32,
        )

        if x1.ndim == 1:
            x1 = x1.unsqueeze(-1)
            x2 = x2.unsqueeze(-1)

        return (
            x1,
            x2,
            torch.tensor(len(x1), dtype=torch.long),
            torch.tensor(len(x2), dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.float32),
        )


def siamese_collate(batch):

    x1 = [b[0] for b in batch]
    x2 = [b[1] for b in batch]

    len1 = torch.stack([b[2] for b in batch])
    len2 = torch.stack([b[3] for b in batch])

    labels = torch.stack([b[4] for b in batch])

    x1 = pad_sequence(
        x1,
        batch_first=True,
        padding_value=0.0,
    )

    x2 = pad_sequence(
        x2,
        batch_first=True,
        padding_value=0.0,
    )

    return x1, x2, len1, len2, labels


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



def load_pairs(path):

    data = np.load(
        path,
        allow_pickle=True,
    )

    return (
        data["pairs_1"],
        data["pairs_2"],
        data["labels"],
    )


def normalize(X):

    X = np.asarray(X)

    if X.dtype == object:
        return normalize_sequences(X)

    X = np.nan_to_num(
        X,
        nan=np.nanmean(X),
        posinf=0,
        neginf=0,
    )

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8

    return (X - mean) / std


def normalize_sequences(X):

    normalized = []

    for x in X:

        x = np.asarray(
            x,
            dtype=np.float32,
        )

        x = np.nan_to_num(
            x,
            nan=np.nanmean(x),
            posinf=0,
            neginf=0,
        )

        mean = x.mean(
            axis=0,
            keepdims=True,
        )

        std = x.std(
            axis=0,
            keepdims=True,
        ) + 1e-8

        normalized.append(
            (x - mean) / std
        )

    return normalized


def select_pairs(X1, X2, y, indices):

    X1_selected = [X1[i] for i in indices]
    X2_selected = [X2[i] for i in indices]

    y_selected = np.asarray(y)[indices]

    return X1_selected, X2_selected, y_selected



EXCLUDED_DATASETS = {
    "Fungi"
    "plaid",
    "FordA",
    "HandOutlines",
}


def getDatasets():

    datasets = []

    for dataset_dir in sorted(UCR_ROOT.iterdir()):

        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name

        if dataset_name in EXCLUDED_DATASETS:
            print(f"[SKIP] {dataset_name}: excluded dataset")
            continue

        train_pairs = (
            dataset_dir / "similarity/train_pairs.npz"
        )

        test_pairs = (
            dataset_dir / "similarity/test_pairs.npz"
        )

        datasets.append({
            "name": dataset_name,
            "train": str(train_pairs),
            "test": str(test_pairs),
            "task": "similarity",
        })

    # ArabicSpokenDigits datasets remain unchanged
    datasets.extend([
        {
            "name": "ArabicSpokenDigits1",
            "train": str(ARABIC_ROOT / "similarity/univariate/digit/train_pairs.npz"),
            "test": str(ARABIC_ROOT / "similarity/univariate/digit/test_pairs.npz"),
            "task": "digit",
        },
        {
            "name": "ArabicSpokenDigits2",
            "train": str(ARABIC_ROOT / "similarity/univariate/voice/train_pairs.npz"),
            "test": str(ARABIC_ROOT / "similarity/univariate/voice/test_pairs.npz"),
            "task": "voice",
        },
        {
            "name": "ArabicSpokenDigits1_Multivariate",
            "train": str(ARABIC_ROOT / "similarity/multivariate/digit/train_pairs.npz"),
            "test": str(ARABIC_ROOT / "similarity/multivariate/digit/test_pairs.npz"),
            "task": "digit",
        },
        {
            "name": "ArabicSpokenDigits2_Multivariate",
            "train": str(ARABIC_ROOT / "similarity/multivariate/voice/train_pairs.npz"),
            "test": str(ARABIC_ROOT / "similarity/multivariate/voice/test_pairs.npz"),
            "task": "voice",
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
    threshold=0.5,
):

    encoder.eval()
    head.eval()

    dataset = PairDataset(
        X1,
        X2,
        y,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=siamese_collate,
    )

    probs = []

    all_h1 = []
    all_h2 = []
    all_pair_repr = []
    all_logits = []

    for a, b, len_a, len_b, _ in loader:

        a = a.to(DEVICE)
        b = b.to(DEVICE)

        len_a = len_a.to(DEVICE)
        len_b = len_b.to(DEVICE)

        h1 = encoder.encode(a, len_a)
        h2 = encoder.encode(b, len_b)

        pair_repr = h1 * h2

        logits = head(pair_repr)

        p = torch.sigmoid(logits).cpu().numpy()

        p = np.nan_to_num(
            p,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        )

        probs.extend(p)

        all_h1.append(h1.cpu())
        all_h2.append(h2.cpu())
        all_pair_repr.append(pair_repr.cpu())
        all_logits.append(logits.detach().cpu())

    probs = np.asarray(probs)

    all_h1 = torch.cat(all_h1, dim=0)
    all_h2 = torch.cat(all_h2, dim=0)
    all_pair_repr = torch.cat(all_pair_repr, dim=0)
    all_logits = torch.cat(all_logits, dim=0)


    pred = (probs > threshold).astype(int)



    return {
        "accuracy": accuracy_score(y, pred),

        "f1": f1_score(
            y,
            pred,
            zero_division=0,
        ),

        "auroc": (
            roc_auc_score(y, probs)
            if len(np.unique(y)) > 1
            else np.nan
        ),

        "probs": probs,
    }


def train(
    X1,
    X2,
    y,
    batch_size=32,
    aggregation="last",
    hidden_size=128,
    embedding_dim=128,
    dropout=0.0,
    learning_rate=1e-4,
    epochs=40,
):

    sample = np.asarray(X1[0])

    input_size = (
        1 if sample.ndim == 1
        else sample.shape[1]
    )

    encoder = SiameseRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        embedding_dim=embedding_dim,
        aggregation=aggregation,
        dropout=dropout,
    ).to(DEVICE)

    head = SimilarityHead(
        embedding_dim
    ).to(DEVICE)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_dataset = PairDataset(
        X1,
        X2,
        y,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=siamese_collate,
        generator=generator,
    )

    params = (
        list(encoder.parameters())
        + list(head.parameters())
    )

    opt = torch.optim.Adam(
        params,
        lr=learning_rate,
    )

    for epoch in range(1, epochs + 1):

        encoder.train()
        head.train()

        loss_total = 0.0

        for a, b, len_a, len_b, labels_batch in loader:

            a = a.to(DEVICE)
            b = b.to(DEVICE)

            len_a = len_a.to(DEVICE)
            len_b = len_b.to(DEVICE)

            labels_batch = labels_batch.to(DEVICE)

            distance = encoder(
                a,
                b,
                len_a,
                len_b,
            )

            logits = head(distance)

            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels_batch,
            )

            if torch.isnan(loss):
                continue

            opt.zero_grad()

            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                params,
                5.0,
            )

            opt.step()

            loss_total += loss.item()

        print(
            f"Epoch {epoch}/{epochs}: "
            f"Loss={loss_total / max(len(loader), 1):.8f}"
        )

    return encoder, head


def run_hpo(
    X1,
    X2,
    y,
    splits,
    dataset_name,
    n_trials=100,
    seed=42,
):

    configs = list(
        ParameterSampler(
            HPO_SPACE,
            n_iter=n_trials,
            random_state=seed,
        )
    )

    trial_results = []

    for trial_idx, config in enumerate(
        configs,
        start=1,
    ):

        print("\n" + "=" * 70)

        print(
            f"Dataset: {dataset_name} | "
            f"Trial {trial_idx}/{n_trials}"
        )

        print(config)

        fold_aurocs = []
        fold_accuracies = []
        fold_f1s = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            splits,
            start=1,
        ):

            print(
                f"\nTrial {trial_idx} | "
                f"Fold {fold_idx}/{len(splits)}"
            )

            set_seed(seed)

            X1_train, X2_train, y_train = select_pairs(
                X1,
                X2,
                y,
                train_idx,
            )

            X1_val, X2_val, y_val = select_pairs(
                X1,
                X2,
                y,
                val_idx,
            )

            encoder, head = train(
                X1_train,
                X2_train,
                y_train,
                **config,
            )

            metrics = evaluate(
                encoder,
                head,
                X1_val,
                X2_val,
                y_val,
                batch_size=config["batch_size"],
                threshold=0.5,
            )

            fold_aurocs.append(
                metrics["auroc"]
            )

            fold_accuracies.append(
                metrics["accuracy"]
            )

            fold_f1s.append(
                metrics["f1"]
            )

            print(
                f"Fold {fold_idx}: "
                f"AUROC={metrics['auroc']:.4f}, "
                f"Accuracy={metrics['accuracy']:.4f}, "
                f"F1={metrics['f1']:.4f}"
            )

            del encoder, head

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_auroc = (
            np.nanmean(fold_aurocs)
            if not np.all(np.isnan(fold_aurocs))
            else np.nan
        )

        result = {
            "dataset": dataset_name,
            "trial": trial_idx,
            **config,
            "mean_auroc": mean_auroc,
            "std_auroc": np.nanstd(fold_aurocs),
            "mean_accuracy": np.mean(fold_accuracies),
            "std_accuracy": np.std(fold_accuracies),
            "mean_f1": np.mean(fold_f1s),
            "std_f1": np.std(fold_f1s),
        }

        trial_results.append(result)

        print(
            f"\nTrial {trial_idx} completed: "
            f"Mean AUROC={mean_auroc:.4f}"
        )

    results_df = pd.DataFrame(trial_results)

    results_df = results_df.sort_values(
        "mean_auroc",
        ascending=False,
        na_position="last",
    )

    if results_df.empty or results_df["mean_auroc"].isna().all():
        raise RuntimeError(
            f"No valid AUROC values for {dataset_name}."
        )

    best_config = results_df.iloc[0][
        list(HPO_SPACE.keys())
    ].to_dict()

    print("\n===== Best Configuration =====")
    print(best_config)

    return best_config, results_df


def append_or_create_csv(df, path):

    if df.empty:
        return

    if path.exists():

        df.to_csv(
            path,
            mode="a",
            header=False,
            index=False,
        )

    else:

        df.to_csv(
            path,
            index=False,
        )


def load_completed_datasets():

    if not RESUME or not TEST_RESULTS_PATH.exists():
        return set()

    existing = pd.read_csv(
        TEST_RESULTS_PATH
    )

    if "dataset" not in existing.columns:
        return set()

    return set(
        existing["dataset"].astype(str)
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    set_seed(SEED)

    datasets = getDatasets()

    if args.index < 1 or args.index > len(datasets):
        raise ValueError(
            f"Index must be between 1 and {len(datasets)}, "
            f"got {args.index}"
        )

    info = datasets[args.index - 1]
    dataset_name = info["name"]

    print(f"Selected dataset {args.index}/{len(datasets)}: {dataset_name}")

    completed_datasets = load_completed_datasets()

    if dataset_name in completed_datasets:
        print(f"\nSkipping completed dataset: {dataset_name}")

    else:
        print("\n" + "=" * 80)
        print(f"Processing dataset: {dataset_name}")
        print("=" * 80)

        X1, X2, labels = load_pairs(
            info["train"]
        )

        labels = np.asarray(labels)

        
        X1 = normalize_sequences(
            np.asarray(X1)
        )

        X2 = normalize_sequences(
            np.asarray(X2)
        )


        splits = list(
            create_cv_splits(
                labels,
                n_splits=N_SPLITS,
                seed=SEED,
            )
        )


        best_config, hpo_df = run_hpo(
            X1,
            X2,
            labels,
            splits,
            dataset_name=dataset_name,
            n_trials=N_TRIALS,
            seed=SEED,
        )

        append_or_create_csv(
            hpo_df,
            HPO_RESULTS_PATH,
        )


        best_config_row = {
            "dataset": dataset_name,
            **best_config,
        }

        append_or_create_csv(
            pd.DataFrame([best_config_row]),
            BEST_CONFIGS_PATH,
        )

        print("\nSelected configuration:")
        print(best_config)


        X1_test, X2_test, labels_test = load_pairs(
            info["test"]
        )

        labels_test = np.asarray(
            labels_test
        )

        X1_test = normalize_sequences(
            np.asarray(X1_test)
        )

        X2_test = normalize_sequences(
            np.asarray(X2_test)
        )


        print("\n" + "=" * 70)
        print("Final training on complete training set")
        print("=" * 70)

        set_seed(SEED)

        encoder, head = train(
            X1,
            X2,
            labels,
            **best_config,
        )


        test_metrics = evaluate(
            encoder,
            head,
            X1_test,
            X2_test,
            labels_test,
            batch_size=best_config["batch_size"],
            threshold=0.5,
        )

        test_metrics.pop("probs")

        model_name = (
            "SRN-L"
            if best_config["aggregation"] == "last"
            else "SRN-A"
        )

        test_row = {
            "dataset": dataset_name,
            "model": model_name,
            **best_config,
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_auroc": test_metrics["auroc"],
        }


        append_or_create_csv(
            pd.DataFrame([test_row]),
            TEST_RESULTS_PATH,
        )

        print("\n===== Final Test Results =====")

        print(
            f"Dataset: {dataset_name}"
        )

        print(
            f"Model: {model_name}"
        )

        print(
            f"Test Accuracy: "
            f"{test_metrics['accuracy']:.4f}"
        )

        print(
            f"Test F1: "
            f"{test_metrics['f1']:.4f}"
        )

        print(
            f"Test AUROC: "
            f"{test_metrics['auroc']:.4f}"
        )

        print(
            f"\nTest results saved to: "
            f"{TEST_RESULTS_PATH}"
        )

        del encoder, head

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"\nFinished dataset: {dataset_name}"
        )

    print("\nAll requested datasets processed.")