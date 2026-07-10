import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
import sys
import os
from collections import Counter
import argparse
import sqlite3
import json

import numpy as np
from pathlib import Path

root_dir = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi")
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"src").resolve()))  
sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos import ChronosPipeline


def load_ucr_tsv(tsv_path, context_length=512):
    df = pd.read_csv(tsv_path, sep="\t", header=None).values

    y = df[:, 0]
    X = df[:, 1:].astype(np.float32)


    y = y.astype(int)
    unique = np.unique(y)
    label_map = {v: i for i, v in enumerate(unique)}
    y = np.vectorize(label_map.get)(y)

    # pad / truncate
    if X.shape[1] < context_length:
        pad = context_length - X.shape[1]
        X = np.pad(X, ((0, 0), (0, pad)), mode="constant")
    else:
        X = X[:, -context_length:]

    return X, y


def load_uci_har(test_dir: str, context_length: int = 512):

    test_dir = Path(test_dir)

    x_path = test_dir / "X_test.txt"
    y_path = test_dir / "y_test.txt"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing: {y_path}")

    X = np.loadtxt(x_path)
    y = np.loadtxt(y_path).astype(int)

    y = y - 1

    if X.ndim == 1:
        X = X.reshape(1, -1)


    if X.shape[1] < context_length:
        pad = context_length - X.shape[1]
        X = np.pad(X, ((0, 0), (0, pad)))
    else:
        X = X[:, -context_length:]

    return X, y


def evaluate_model(model, tokenizer, X, y, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    preds_all, labels_all = [], []

    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size), desc="Evaluating"):
            batch_X = X[i:i + batch_size]
            batch_y = y[i:i + batch_size]

            context = torch.tensor(batch_X, dtype=torch.float32)

            input_ids, attention_mask, _ = tokenizer.context_input_transform(context)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs["logits"]

            # safety fix (some models return (B,T,C))
            if logits.ndim == 3:
                logits = logits[:, -1, :]

            preds = torch.argmax(logits, dim=-1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(batch_y)

    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="weighted")


    return {
        "accuracy": acc,
        "f1": f1,
        "predictions": preds_all,
        "labels": labels_all,
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    idx = args.index

    batch_size = 32
    context_length = 512

    DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/classification.db"

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT config, model_path, test_data, dataset
        FROM runs
        WHERE id = ?
    """, (idx,))

    row = cur.fetchone()

    if row is None:
        raise ValueError(f"No run found for id={idx}")

    config_json, model_path, test_data, dataset = row
    config = json.loads(config_json)

    num_labels = config["num_labels"]

    pipeline = ChronosPipeline.from_pretrained(
        model_path,
        task="classification",
        num_labels=num_labels
    )

    model = pipeline.model
    tokenizer = pipeline.tokenizer

    classifier_path = Path(model_path) / "classifier.pt"
    if classifier_path.exists():
        model.classifier.load_state_dict(
            torch.load(classifier_path, map_location="cpu")
        )

    if dataset == "UCI-HAR":

        X_test, y_test = load_uci_har(
            test_dir=test_data,
            context_length=context_length
        )

    else:

        X_test, y_test = load_ucr_tsv(
            tsv_path=test_data,
            context_length=context_length
        )




    results = evaluate_model(model, tokenizer, X_test, y_test, batch_size)



    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        UPDATE runs
        SET accuracy = ?,
            f1 = ?,
            status = 'DONE'
        WHERE id = ?
    """, (
        results["accuracy"],
        results["f1"],
        idx,
    ))

    conn.commit()
    conn.close()
