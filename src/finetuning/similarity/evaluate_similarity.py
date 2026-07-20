import argparse
import json
import sqlite3
from pathlib import Path
import sys
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

root_dir = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi")
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"src").resolve()))  
sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos import ChronosPipeline



def load_ucr_tsv(tsv_path, context_length=512):

    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None
    ).values

    y = df[:, 0]
    X = df[:, 1:].astype(np.float32)

    y = y.astype(int)

    unique = np.unique(y)
    label_map = {
        v: i for i, v in enumerate(unique)
    }

    y = np.vectorize(label_map.get)(y)


    if X.shape[1] < context_length:
        pad = context_length - X.shape[1]
        X = np.pad(
            X,
            ((0,0),(0,pad)),
            mode="constant"
        )
    else:
        X = X[:, -context_length:]


    return X, y



def load_uci_har(test_dir, context_length=512):

    test_dir = Path(test_dir)

    X = np.loadtxt(
        test_dir / "X_test.txt"
    )

    y = np.loadtxt(
        test_dir / "y_test.txt"
    ).astype(int)

    y = y - 1


    if X.ndim == 1:
        X = X.reshape(1,-1)


    if X.shape[1] < context_length:
        pad = context_length - X.shape[1]

        X = np.pad(
            X,
            ((0,0),(0,pad))
        )

    else:
        X = X[:, -context_length:]


    return X, y


def create_pairs(
    X,
    y,
    n_pairs=None
):

    if n_pairs is None:
        n_pairs = len(X)


    pairs_1 = []
    pairs_2 = []
    labels = []


    classes = np.unique(y)


    for _ in range(n_pairs):

        # positive
        if np.random.rand() < 0.5:

            c = np.random.choice(classes)

            idx = np.where(y == c)[0]

            if len(idx) < 2:
                continue

            i,j = np.random.choice(
                idx,
                2,
                replace=False
            )

            label = 1


        # negative
        else:

            i,j = np.random.choice(
                len(X),
                2,
                replace=False
            )

            while y[i] == y[j]:
                j = np.random.randint(len(X))

            label = 0


        pairs_1.append(X[i])
        pairs_2.append(X[j])
        labels.append(label)


    return (
        np.asarray(pairs_1),
        np.asarray(pairs_2),
        np.asarray(labels)
    )



@torch.no_grad()
def evaluate_similarity(
    model,
    tokenizer,
    X1,
    X2,
    y,
    batch_size=32
):

    device = next(model.parameters()).device

    similarities = []

    model.eval()


    for start in tqdm(
        range(0, len(y), batch_size),
        desc="Evaluating similarity",
        total=(len(y) + batch_size - 1) // batch_size
    ):

        end = start + batch_size

        c1 = torch.tensor(
            X1[start:end],
            dtype=torch.float32
        ).to(device)

        c2 = torch.tensor(
            X2[start:end],
            dtype=torch.float32
        ).to(device)


        ids1, mask1, _ = tokenizer.context_input_transform(c1)

        ids2, mask2, _ = tokenizer.context_input_transform(c2)


        outputs = model(
            input_ids_1=ids1,
            attention_mask_1=mask1,
            input_ids_2=ids2,
            attention_mask_2=mask2,
        )


        z1 = outputs["embeddings_1"]
        z2 = outputs["embeddings_2"]


        # cosine similarity [-1,1]
        sim = F.cosine_similarity(
            z1,
            z2,
            dim=-1
        )


        similarities.append(
            sim.cpu()
        )


    similarities = torch.cat(
        similarities
    ).numpy()


    probs = (similarities + 1) / 2


    preds = (
        probs > 0.5
    ).astype(int)


    return {
        "accuracy": accuracy_score(
            y,
            preds
        ),
        "f1": f1_score(
            y,
            preds
        ),
        "auroc": roc_auc_score(
            y,
            probs
        )
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=int,
        required=True
    )
    args = parser.parse_args()
    idx = args.index

    np.random.seed(42)
    context_length = 512
    batch_size = 32


    DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/similarity/similarity.db"


    conn = sqlite3.connect(DB_PATH)

    cur = conn.cursor()


    cur.execute(
        """
        SELECT config,
               model_path,
               test_data,
               dataset
        FROM runs
        WHERE id = ?
        """,
        (idx,)
    )

    row = cur.fetchone()


    if row is None:
        raise ValueError(
            f"No run found id={idx}"
        )


    config_json, model_path, test_data, dataset = row

    config = json.loads(
        config_json
    )

    pipeline = ChronosPipeline.from_pretrained(
        model_path,
        task="similarity"
    )

    model = pipeline.model

    tokenizer = pipeline.tokenizer
    head_path = Path(model_path) / "projection.pt"

    model.projection.load_state_dict(
        torch.load(
            head_path,
            map_location="cpu"
        )
    )

    if dataset == "UCI-HAR":
        X_test, y_test = load_uci_har(
            test_data,
            context_length
        )
    else:
        X_test, y_test = load_ucr_tsv(
            test_data,
            context_length
        )


    X1, X2, pair_labels = create_pairs(
        X_test,
        y_test,
        n_pairs=len(X_test)
    )

    results = evaluate_similarity(
        model,
        tokenizer,
        X1,
        X2,
        pair_labels,
        batch_size
    )

    print(results)

    cur.execute(
        """
        UPDATE runs
        SET accuracy=?,
            f1=?,
            auroc=?,
            status='DONE'
        WHERE id=?
        """,
        (
            results["accuracy"],
            results["f1"],
            results["auroc"],
            idx
        )
    )

    conn.commit()
    conn.close()