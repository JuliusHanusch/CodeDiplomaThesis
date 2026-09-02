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
import time

import numpy as np
from pathlib import Path

root_dir = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi")
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"src").resolve()))  
sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos import ChronosPipeline

DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/classification_allData.db"

def get_db_connection():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=60,
    )
    conn.execute("PRAGMA busy_timeout=60000")
    return conn

def execute_db_update(sql, params, description="database update"):
    for attempt in range(5):

        conn = None

        try:
            conn = get_db_connection()

            conn.execute(sql, params)
            conn.commit()
            conn.close()

            return

        except sqlite3.OperationalError as e:

            if conn is not None:
                conn.close()

            if "database is locked" not in str(e):
                raise

            print(
                f"[SQLite] Database locked during {description} "
                f"- retry {attempt + 1}/5",
                flush=True
            )

            if attempt < 4:
                time.sleep(5)

    raise RuntimeError(
        f"[SQLite] Database remained locked during "
        f"{description} after 5 attempts."
    )


def load_ucr_tsv(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", header=None).values

    y = df[:, 0]
    X = df[:, 1:].astype(np.float32)


    y = y.astype(int)
    unique = np.unique(y)
    label_map = {v: i for i, v in enumerate(unique)}
    y = np.vectorize(label_map.get)(y)

    return X, y


def load_uci_har(test_dir: str):

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

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(f"""
        SELECT
            config,
            model_path,
            eval_data,
            dataset
        FROM runs
        WHERE id = ?
    """, (idx,))

    row = cur.fetchone()

    if row is None:
        raise ValueError(f"No run found for id={idx}")

    config_json, model_path, eval_data, dataset = row

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
            test_dir=eval_data,
        )
    else:
        X_test, y_test = load_ucr_tsv(
            tsv_path=eval_data,
        )


    results = evaluate_model(model, tokenizer, X_test, y_test, batch_size)

    execute_db_update(
        """
            UPDATE runs
            SET
                accuracy = ?,
                f1 = ?
            WHERE id = ?
        """, (
            results["accuracy"],
            results["f1"],
            idx,
        ),
        description="Results"
    )

    conn.commit()
    conn.close()
