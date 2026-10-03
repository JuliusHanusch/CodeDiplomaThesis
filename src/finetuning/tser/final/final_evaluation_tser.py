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

DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/tser/Final/tser_bestConfigs_time.db"

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

            pred = outputs["logits"].detach().cpu().numpy().reshape(-1)[0]

            all_preds.append(pred)
            all_labels.append(label)

    all_preds = np.asarray(all_preds, dtype=np.float32)
    all_labels = np.asarray(all_labels, dtype=np.float32)
    absolute_errors = np.abs(all_preds - all_labels)

    return {
        "rmse": rmse(all_preds, all_labels),
        "mae": mae(all_preds, all_labels),
        "n_samples": len(all_preds),
        "predictions": all_preds,
        "labels": all_labels,
        "absolute_errors": absolute_errors,
    }



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()

    idx = args.index
    seed = args.seed


    conn = get_db_connection()   
    cur = conn.cursor()


    model_path_column = f"model_path_{seed}"

    cur.execute(
        f"""
            SELECT
                dataset,
                {model_path_column},
                test_data
            FROM runs
            WHERE id = ?
        """,
        (idx,)
    )

    row = cur.fetchone()

    if row is None:
        raise ValueError(f"No run found for id={idx}")

    dataset, model_path, test_dataset = row

    conn.close()

    start_time = time.perf_counter()

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

    inference_time = time.perf_counter() - start_time

    output_dir = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/Results/Finetuning/TSER/Predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / f"{dataset}_seed_{seed}.npz",
        y_true=results["labels"],
        predictions=results["predictions"],
        absolute_errors=results["absolute_errors"],
    )



    # print("RMSE", results["rmse"], "MAE", results["mae"])

    execute_db_update(
        f"""
        UPDATE runs
        SET
            rmse_{seed} = ?,
            mae_{seed} = ?,
            inference_time_{seed} = ?
        WHERE id = ?
        """,
        (
            results["rmse"],
            results["mae"],
            inference_time,
            idx,
        ),
        description=(
            f"updating results and inference time "
            f"for idx {idx}, seed {seed}"
        )
    )



