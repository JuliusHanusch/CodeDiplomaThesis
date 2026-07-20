import sqlite3
import json
import yaml
import random
import numpy as np
import hashlib
import copy

DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/imputation/imputation.db"
BASE_CONFIG_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/imputation/base_config.yaml"

def config_hash(config: dict,) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()


def sample_config():
    """Search space definition."""

    return {
        "max_steps": int(random.choice([500, 1000, 2000, 5000, 10000])),
        "per_device_train_batch_size": int(random.choice([8, 16, 32])),
        "learning_rate": float(np.exp(
            np.random.uniform(np.log(1e-5), np.log(5e-4))
        )),
        "dropout_head": float(random.uniform(0.0, 0.3)),
        "warmup_ratio": float(random.uniform(0.0, 0.1)),
        "TrainInnerModel": bool(random.choice([True, False])),
    }
MASK_RATIOS = [0.15, 0.30, 0.45]
SPAN_LENGTHS = [16, 32, 64]
datasets = [
    {
        "name": "electricity_15min",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Imputation/electricity_15min.arrow",
    },
    {
        "name": "m4_daily",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Imputation/m4_daily.arrow",
    },
    {
        "name": "m4_hourly",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Imputation/m4_hourly.arrow",
    },
    {
        "name": "m4_weekly",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Imputation/m4_weekly.arrow",
    },
        {
        "name": "monash_electricity_hourly",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Imputation/monash_electricity_hourly.arrow",
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


def insert(conn, cfg, ds, mask_ratio, mean_span_length):
    h = config_hash(cfg)
    cur = conn.cursor()

    try:
        cur.execute("""
            INSERT INTO runs (
                config_hash,
                config,
                dataset,
                train_data,
                masking_ratio,
                mean_span_length,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            h,
            json.dumps(cfg),
            ds["name"],
            ds["train"],
            mask_ratio,
            mean_span_length,
            "PENDING"
        ))

        conn.commit()
        return True

    except sqlite3.IntegrityError as e:
        print(e)
        return False


def generate(n_configs=20):
    conn = sqlite3.connect(DB_PATH)

    inserted = 0
    attempts = 0

    while inserted < n_configs:
        cfg = make_full_config()
        for dataset in datasets:
            for mask_ratio in MASK_RATIOS:
                for span_length in SPAN_LENGTHS:
                    cfg["training_data_paths"] = [str(dataset["train"])]
                    cfg["masking_prob"] = mask_ratio
                    cfg["mean_span_length"] = span_length
                    insert(
                        conn,
                        cfg,
                        dataset,
                        mask_ratio,
                        span_length,
                    )

        inserted += 1
        print(f"Inserted config {inserted}/{n_configs}")

        attempts += 1
        if attempts > n_configs * 20:
            break

    conn.close()


if __name__ == "__main__":
    generate(20)