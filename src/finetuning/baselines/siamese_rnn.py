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

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DB_PATH="/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/similarity/similarity.db"


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

def load_arabic_digits(path, task="digit", multivariate=False):

    data=np.load(
        path,
        allow_pickle=True
    )

    X=list(data["X"])


    if not multivariate:

        X=[
            x[:,0] if x.ndim==2 else x
            for x in X
        ]


    if task=="digit":
        y=data["digit_labels"]

    elif task=="voice":
        y=data["speaker_labels"]

    else:
        raise ValueError


    return X,y

def load_ucr(path):
    df=pd.read_csv(path,sep="\t",header=None).values
    return df[:,1:].astype(np.float32),df[:,0].astype(int)


def load_uci(path, split):

    path = Path(path)

    X = np.loadtxt(
        path / f"X_{split}.txt"
    ).astype(np.float32)

    y = np.loadtxt(
        path / f"y_{split}.txt"
    ).astype(int) - 1

    return X,y

def normalize(X):
    X=np.nan_to_num(
        X,
        nan=np.nanmean(X),
        posinf=0,
        neginf=0
    )

    mean=X.mean(axis=0)
    std=X.std(axis=0)+1e-8

    return (X-mean)/std

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

        normalized.append(
            (x - mean) / std
        )

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

def create_pairs(X,y,n):
    X1,X2,Y=[],[],[]
    classes=np.unique(y)

    while len(Y)<n:
        if np.random.rand()<0.5:
            c=np.random.choice(classes)
            ids=np.where(y==c)[0]
            if len(ids)<2:
                continue
            i,j=np.random.choice(ids,2,replace=False)
            label=1
        else:
            i,j=np.random.choice(len(X),2,replace=False)
            if y[i]==y[j]:
                continue
            label=0

        X1.append(X[i])
        X2.append(X[j])
        Y.append(label)

    return X1, X2, np.array(Y)


def train(
    X1,X2,y,X1_test,X2_test,y_test,dataset,aggregation="last"
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
    head=SimilarityHead(128).to(DEVICE)

    if "ArabicSpokenDigits" in dataset:
        loader = DataLoader(
            PairDataset(X1, X2, y),
            batch_size=64,
            shuffle=True,
            collate_fn=siamese_collate,
        )
    else:
        loader = DataLoader(
            PairDataset(X1, X2, y),
            batch_size=64,
            shuffle=True,
        )

    params=list(encoder.parameters())+list(head.parameters())

    opt=torch.optim.Adam(
        params,
        lr=1e-4
    )

    checkpoints=[5,10,20,40,60,100,150,200]
    results=[]

    for epoch in range(1,201):

        encoder.train()
        head.train()

        loss_total=0

        for a,b,len_a,len_b,l in loader:

            a,b = a.to(DEVICE), b.to(DEVICE)
            len_a,len_b = len_a.to(DEVICE), len_b.to(DEVICE)
            l=l.to(DEVICE)

            distance = encoder(a,b,len_a,len_b)
            logits=head(distance)

            loss=F.binary_cross_entropy_with_logits(
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

            loss_total+=loss.item()


        print(
            f"Epoch {epoch}: {loss_total/len(loader):.4f}"
        )


        if epoch in checkpoints:

            metrics=evaluate(
                encoder,
                head,
                X1_test,
                X2_test,
                y_test,
                dataset
            )

            metrics["epoch"]=epoch
            results.append(metrics)

            print(metrics)

    return encoder,head,results


@torch.no_grad()
def evaluate(encoder,head,X1,X2,y, dataset):
    encoder.eval()
    head.eval()

    if "ArabicSpokenDigits" in dataset:
        loader = DataLoader(
            PairDataset(X1, X2, y),
            batch_size=64,
            shuffle=False,
            collate_fn=siamese_collate,
        )
    else:
        loader = DataLoader(
            PairDataset(X1, X2, y),
            batch_size=64,
            shuffle=False,
        )

    probs=[]

    for a,b,len_a,len_b,_ in loader:

        a,b = a.to(DEVICE), b.to(DEVICE)
        len_a,len_b = len_a.to(DEVICE), len_b.to(DEVICE)

        logits = head(
            encoder(a,b,len_a,len_b)
        )

        p=torch.sigmoid(logits).cpu().numpy()
        p=np.nan_to_num(
            p,
            nan=0.5,
            posinf=1.0,
            neginf=0.0
        )

        probs.extend(p)

    probs=np.array(probs)
    pred=(probs>0.5).astype(int)

    return {
        "accuracy":accuracy_score(y,pred),
        "f1":f1_score(y,pred),
        "auroc":roc_auc_score(y,probs) if len(np.unique(y))>1 else np.nan
    }

datasets = [
    # EXAMPLE:
    {
        "name": "ArabicSpokenDigits1Uni",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Similarity/ArabicSpokenDigits/arabic_digits_univariate_train.npz",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Similarity/ArabicSpokenDigits/arabic_digits_univariate_test.npz",
        "task": "digit"
    },
    {
        "name": "ArabicSpokenDigits2Uni",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Similarity/ArabicSpokenDigits/arabic_digits_univariate_train.npz",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Similarity/ArabicSpokenDigits/arabic_digits_univariate_test.npz",
        "task": "voice"
    },
    {
        "name": "ArabicSpokenDigits1Multi",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Similarity/ArabicSpokenDigits/arabic_digits_multivariate_train.npz",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Similarity/ArabicSpokenDigits/arabic_digits_multivariate_test.npz",
        "task": "digit"
    },
        {
        "name": "ArabicSpokenDigits2Multi",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Similarity/ArabicSpokenDigits/arabic_digits_multivariate_train.npz",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Similarity/ArabicSpokenDigits/arabic_digits_multivariate_test.npz",
        "task": "voice"
    },
    {
        "name": "UCI-HAR",
        "train":"/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCI_HAR/UCI HAR Dataset/train/",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCI_HAR/UCI HAR Dataset/test/",
        "task": "similarity"
    },
    {
        "name": "ArrowHead",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/ArrowHead/ArrowHead_TRAIN.tsv",
        "test":  "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/ArrowHead/ArrowHead_TEST.tsv",
        "task": "similarity"
    },
    {
        "name": "DistalPhalanxTW",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/DistalPhalanxTW/DistalPhalanxTW_TRAIN.tsv",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/DistalPhalanxTW/DistalPhalanxTW_TEST.tsv",
        "task": "similarity"
    },
    {
        "name": "GestureMidAirD2",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/GestureMidAirD2/GestureMidAirD2_TRAIN.tsv",
        "test":  "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/GestureMidAirD2/GestureMidAirD2_TEST.tsv",
        "task": "similarity"
    },
    {
        "name": "Wafer",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/Wafer/Wafer_TRAIN.tsv",
        "test":  "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/Wafer/Wafer_TEST.tsv",
        "task": "similarity"
    },
]

if __name__ == "__main__":

    results = []
    np.random.seed(42)


    for info in datasets:

        dataset = info["name"]
        train_data = info["train"]
        test_data = info["test"]
        task = info["task"]

        print("train_data", train_data)
        print("test_data", test_data)

        if dataset=="UCI-HAR":
            X_train,y_train=load_uci(train_data, "train")
            X_test,y_test=load_uci(test_data, "test")
        elif "ArabicSpokenDigits" in dataset:

            multivariate = "multivariate" in train_data

            X_train,y_train = load_arabic_digits(
                train_data,
                task=task,
                multivariate=multivariate
            )

            X_test,y_test = load_arabic_digits(
                test_data,
                task=task,
                multivariate=multivariate
            )
        else:
            X_train,y_train=load_ucr(train_data)
            X_test,y_test=load_ucr(test_data)

        if "ArabicSpokenDigits" in dataset:

            X_train = normalize_sequences(X_train)
            X_test = normalize_sequences(X_test)

        else:

            X_train = normalize(X_train)
            X_test = normalize(X_test)

        train_pairs=create_pairs(
            X_train,
            y_train,
            len(X_train)*5
        )

        test_pairs=create_pairs(
            X_test,
            y_test,
            len(X_test)*5
        )

        for aggregation in ["last", "average"]:

            print(f"\n{'='*60}")
            print(f"Training SRN-{aggregation}")
            print(f"{'='*60}")

            encoder, head, metrics = train(
                train_pairs[0],
                train_pairs[1],
                train_pairs[2],
                test_pairs[0],
                test_pairs[1],
                test_pairs[2],
                dataset,
                aggregation=aggregation
            )

            for m in metrics:
                m["dataset"] = dataset
                m["model"] = (
                    "SRN-L"
                    if aggregation == "last"
                    else "SRN-A"
                )

                results.append(m)

    pd.DataFrame(results).to_csv(
        "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/Results/Finetuning/Similarity/siamese_rnn_final_results.csv",
        index=False
    )
