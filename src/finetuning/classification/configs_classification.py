import sqlite3
import json
import yaml
import random
import numpy as np
import hashlib
import copy

DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/classification.db"
BASE_CONFIG_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/base_config.yaml"

def config_hash(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()


def sample_config():
    """Search space definition."""

    return {
        "num_train_epochs": int(random.choice([2, 5, 10, 20, 40])),
        "per_device_train_batch_size": int(random.choice([8, 16, 32])),
        "learning_rate": float(np.exp(
            np.random.uniform(np.log(1e-5), np.log(5e-4))
        )),
        "dropout_head": float(random.uniform(0.0, 0.3)),
        "warmup_ratio": float(random.uniform(0.0, 0.1)),
        "TrainInnerModel": bool(random.choice([True, False])),
    }

# -----------------------------
# DEFINE DATASETS HERE
# -----------------------------
datasets = [
    # EXAMPLE:
    {
        "name": "ArrowHead",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_arrow/ArrowHead_train.arrow",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/ArrowHead/ArrowHead_TEST.tsv",
        "labels": 3
    },
    {
        "name": "DistalPhalanxTW",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_arrow/DistalPhalanxTW_train.arrow",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/DistalPhalanxTW/DistalPhalanxTW_TEST.tsv",
        "labels": 6
    },
    {
        "name": "GestureMidAirD2",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_arrow/GestureMidAirD2_train.arrow",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/GestureMidAirD2/GestureMidAirD2_TEST.tsv",
        "labels": 26
    },
    {
        "name": "Wafer",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_arrow/Wafer_train.arrow",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/Wafer/Wafer_TEST.tsv",
        "labels": 2
    },
        {
        "name": "UCI-HAR",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCI_HAR/UCI_HAR.arrow",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCI_HAR/UCI HAR Dataset/test",
        "labels": 6
    }
]


def load_base():
    with open(BASE_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def make_full_config():

    base = copy.deepcopy(load_base())
    hparams = sample_config()
    base.update(hparams)

    return base


def insert(conn, cfg, ds):
    h = config_hash(cfg)
    cur = conn.cursor()

    try:
        cur.execute("""
            INSERT INTO runs (
                config_hash,
                config,
                dataset,
                train_data,
                test_data,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            h,
            json.dumps(cfg),
            ds["name"],
            ds["train"],
            ds["test"],
            "PENDING"
        ))

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False


def generate(n_configs=20):
    conn = sqlite3.connect(DB_PATH)

    inserted = 0
    attempts = 0

    while inserted < n_configs:
        cfg = make_full_config()
        for dataset in datasets:
            cfg["training_data_paths"] = [str(dataset["train"])]
            cfg["num_labels"] = int(dataset["labels"])

            insert(conn, cfg, dataset)

        inserted += 1
        print(f"Inserted config {inserted}/{n_configs}")

        attempts += 1

        if attempts > n_configs * 20:
            break

    conn.close()


if __name__ == "__main__":
    generate(20)