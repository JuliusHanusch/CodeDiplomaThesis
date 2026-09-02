from pathlib import Path
import numpy as np
import torch
import pandas as pd
import sys
import sqlite3
import argparse
import time


root_dir = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi")
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"src").resolve()))  
sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos import ChronosPipeline

from gluonts.dataset.arrow import ArrowFile

DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/tser/tser_allData.db"

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

def rmse(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return np.sqrt(np.mean((preds - labels) ** 2))


def mae(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return np.mean(np.abs(preds - labels))

def extract_features(series: np.ndarray):
    series = np.asarray(series)

    series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)

    mean = np.mean(series)
    std = np.std(series)
    min_v = np.min(series)
    max_v = np.max(series)
    last = series[-1]

    x = np.arange(len(series))

    if len(series) < 2 or np.all(series == series[0]):
        slope = 0.0
    else:
        try:
            slope = np.polyfit(x, series, 1)[0]
        except Exception:
            slope = 0.0

    feat = np.array([mean, std, min_v, max_v, last, slope], dtype=np.float32)

    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    return feat


def load_arrow(path: Path):
    """
    Reads TS-ARROW dataset using GluonTS ArrowFile reader.
    """

    dataset = ArrowFile(path)

    series = []
    labels = []

    for entry in dataset:
        target = np.asarray(entry["target"], dtype=np.float32)

        # handle possible label keys safely
        if "label" in entry:
            label = entry["label"]
        elif "y" in entry:
            label = entry["y"]
        else:
            raise KeyError("No label found in dataset entry")

        series.append(target)
        labels.append(float(label))

    return np.stack(series), np.array(labels, dtype=np.float32)


def baseline_predict_mean(series):
    return np.mean(series)


def baseline_predict_last(series):
    return series[-1]


def evaluate_chronos(model, tokenizer, test_arrow_path, context_length=512):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    X, y = load_arrow(test_arrow_path)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in range(len(X)):

            series = X[i]
            label = float(y[i])

            context = torch.tensor(
                series[-context_length:],
                dtype=torch.float32
            )
            input_ids, attention_mask, _ = tokenizer.context_input_transform(context)

            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            preds = outputs["logits"].detach().cpu().numpy().reshape(-1)

            all_preds.extend(preds)
            all_labels.extend([label] * len(preds))

    return {
        "rmse": rmse(all_preds, all_labels),
        "mae": mae(all_preds, all_labels),
        "n_samples": len(all_preds),
    }


def evaluate_all_baselines(ridge_model, test_arrow_path, context_length=512):

    X, y = load_arrow(test_arrow_path)

    preds_mean = []
    preds_last = []
    preds_ridge = []

    for i in range(len(X)):

        series = X[i][-context_length:]
        label = float(y[i])

        # naive baselines
        preds_mean.append(np.mean(series))
        preds_last.append(series[-1])

        # ridge baseline
        feat = extract_features(series).reshape(1, -1)
        preds_ridge.append(ridge_model.predict(feat)[0])

    y = np.asarray(y, dtype=np.float32)

    return {
        "mean": {
            "rmse": rmse(preds_mean, y),
            "mae": mae(preds_mean, y),
        },
        "last": {
            "rmse": rmse(preds_last, y),
            "mae": mae(preds_last, y),
        },
        "ridge": {
            "rmse": rmse(preds_ridge, y),
            "mae": mae(preds_ridge, y),
        },
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    idx = args.index


    conn = get_db_connection()   
    cur = conn.cursor()


    cur.execute(
        """
            SELECT model_path, eval_data
            FROM runs
            WHERE id = ?
        """,
        (idx,)
    )

    row = cur.fetchone()

    if row is None:
        raise ValueError(f"No run found for id={idx}")

    model_path, test_dataset = row
    print("model_path", model_path)

    conn.close()

    pipeline = ChronosPipeline.from_pretrained(
        model_path,
        task="tser",
    )

    model = pipeline.model

    # load regression head
    regressor_path = Path(model_path) / "regressor.pt"
    if regressor_path.exists():
        model.regressor.load_state_dict(
            torch.load(regressor_path, map_location="cpu")
        )

    tokenizer = pipeline.tokenizer

    results = evaluate_chronos(
        model=model,
        tokenizer=tokenizer,
        test_arrow_path=test_dataset,
    )

    conn = get_db_connection()
    cur = conn.cursor()

    print("RMSE", results["rmse"], "MAE", results["mae"])

    execute_db_update(
        """
        UPDATE runs
        SET rmse=?,
            mae=?
        WHERE id=?
        """,
        (
            results["rmse"],
            results["mae"],
            idx,
        ),
        description=f"updating results for idx {idx}"
    )



