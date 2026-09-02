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
from torch.nn.utils.rnn import pad_sequence
import time

root_dir = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi")
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"src").resolve()))  
sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos import ChronosPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/similarity/similarity_allData.db"


def get_db_connection():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=60,
    )
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


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

def execute_db_update(sql, params, max_attempts=5):
    for attempt in range(max_attempts):
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
                f"[SQLite] Database locked. "
                f"Retry {attempt + 1}/{max_attempts}..."
            )

            if attempt < max_attempts - 1:
                time.sleep(5)

    raise RuntimeError(
        f"Database remained locked after {max_attempts} attempts."
    )



@torch.no_grad()
def evaluate_similarity(
    model,
    tokenizer,
    dataset_name,
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

        if dataset_name=="ArabicSpokenDigits1" or dataset_name=="ArabicSpokenDigits2":

            ids1_list = []
            mask1_list = []
            ids2_list = []
            mask2_list = []

            for a, b in zip(
                X1[start:end],
                X2[start:end]
            ):

                i1, m1, _ = tokenizer.context_input_transform(
                    torch.tensor(a).unsqueeze(0)
                )

                i2, m2, _ = tokenizer.context_input_transform(
                    torch.tensor(b).unsqueeze(0)
                )

                ids1_list.append(i1.squeeze(0))
                mask1_list.append(m1.squeeze(0))

                ids2_list.append(i2.squeeze(0))
                mask2_list.append(m2.squeeze(0))


            ids1 = pad_sequence(
                ids1_list,
                batch_first=True,
                padding_value=0
            )

            mask1 = pad_sequence(
                mask1_list,
                batch_first=True,
                padding_value=0
            )

            ids2 = pad_sequence(
                ids2_list,
                batch_first=True,
                padding_value=0
            )

            mask2 = pad_sequence(
                mask2_list,
                batch_first=True,
                padding_value=0
            )

        else:

            c1 = torch.tensor(
                X1[start:end],
                dtype=torch.float32
            )

            c2 = torch.tensor(
                X2[start:end],
                dtype=torch.float32
            )

            ids1, mask1, _ = tokenizer.context_input_transform(c1)
            ids2, mask2, _ = tokenizer.context_input_transform(c2)
            
        ids1 = ids1.to(device)
        mask1 = mask1.to(device)

        ids2 = ids2.to(device)
        mask2 = mask2.to(device)

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

    batch_size = 32


    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        f"""
        SELECT config,
            model_path,
            eval_data,
            dataset,
            task
        FROM runs
        WHERE id = ?
        """,
        (idx,)
    )

    row = cur.fetchone()
    conn.close()

    if row is None:
        raise ValueError(
            f"No run found id={idx}"
        )

    config_json, model_path, eval_data, dataset, task = row
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

    model = pipeline.model.to(DEVICE)

    model.projection.load_state_dict(
        torch.load(
            head_path,
            map_location=DEVICE
        )
    )

    X1, X2, pair_labels = load_pairs(eval_data)

    results = evaluate_similarity(
        model,
        tokenizer,
        dataset,
        X1,
        X2,
        pair_labels,
        batch_size
    )



    execute_db_update(
        """
        UPDATE runs
        SET accuracy=?,
            f1=?,
            auroc=?
        WHERE id=?
        """,
        (
            results["accuracy"],
            results["f1"],
            results["auroc"],
            idx
        )
    )