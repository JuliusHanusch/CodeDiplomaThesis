import sqlite3
import json
import yaml
import random
import numpy as np
import hashlib
import copy

DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/tser/tser.db"
BASE_CONFIG_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/tser/base_config.yaml"

def config_hash(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()


def sample_config():
    """Search space definition."""

    return {
        # core training
        "max_steps": random.choice([500, 1000, 2000, 3000]),
        "per_device_train_batch_size": random.choice([8, 16, 32]),
        "learning_rate": float(
            np.exp(np.random.uniform(np.log(1e-5), np.log(5e-4)))
        ),
        "loss_type": random.choice(["mae", "mse"]),
        # important extras
        "warmup_ratio": random.uniform(0.0, 0.1),
        "gradient_accumulation_steps": random.choice([1, 2, 4]),
        "shuffle_buffer_length": random.choice([1000, 5000, 10000, 20000]),
    }

# -----------------------------
# DEFINE DATASETS HERE
# -----------------------------
datasets = [
    # EXAMPLE:
    {
        "name": "FloodModeling",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/TSER/FloodModeling/train.arrow",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/TSER/FloodModeling/test.arrow",
    },
    {
        "name": "HouseHoldPowerConsumption",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/TSER/HouseHoldPowerConsumption/train.arrow",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/TSER/HouseHoldPowerConsumption/test.arrow",
    },
    {
        "name": "LiveFuelMoisture",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/TSER/LiveFuelMoisture/train.arrow",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/TSER/LiveFuelMoisture/test.arrow",
    },
    {
        "name": "NewsHeadline",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/TSER/NewsHeadline/train.arrow",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/TSER/NewsHeadline/test.arrow",
    },
        {
        "name": "PPGDalia",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/TSER/PPGDalia/train.arrow",
        "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/TSER/PPGDalia/test.arrow",
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
        cfg = make_full_config(dataset)
        for dataset in datasets:
            cfg["training_data_paths"] = [str(dataset["train"])]

            insert(conn, cfg, dataset)

        inserted += 1
        print(f"Inserted config {inserted}/{n_configs}")

        attempts += 1

        if attempts > n_configs * 20:
            break

    conn.close()


if __name__ == "__main__":
    generate(20)