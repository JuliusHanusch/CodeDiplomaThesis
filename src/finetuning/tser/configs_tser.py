import sqlite3
import json
import random
import numpy as np
import hashlib
from pathlib import Path

DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/tser/tser_allData.db"
BASE_CONFIG_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/tser/base_config.yaml"

TSER_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/data/finetuning/TSER"
)


def config_hash(config: dict, dataset: str) -> str:
    hash_data = {
        "config": config,
        "dataset": dataset,
    }

    return hashlib.sha256(
        json.dumps(hash_data, sort_keys=True).encode()
    ).hexdigest()


def sample_config():
    """Search space definition."""

    max_gpu_batch = 32
    sampled_batch = int(random.choice([8, 16, 32, 64, 128]))
    if sampled_batch > max_gpu_batch:
        gradient_accumulation_steps = sampled_batch // max_gpu_batch
        per_device_train_batch_size = max_gpu_batch
    else:
        gradient_accumulation_steps = 1
        per_device_train_batch_size = sampled_batch

    return {
        "num_train_epochs": int(random.choice([2, 5, 10, 20, 40])),
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": float(np.exp(
            np.random.uniform(np.log(5e-6), np.log(1e-3))
        )),
        "dropout_head": float(random.uniform(0.0, 0.3)),
        "warmup_ratio": float(random.uniform(0.0, 0.1)),
        "TrainInnerModel": bool(random.choice([True, False])),
        "loss_type": random.choice(["mae", "mse"]),
    }


def find_datasets():

    datasets = []

    for out_dir in sorted(TSER_ROOT.iterdir()):

        if not out_dir.is_dir():
            continue

        train_file = out_dir / "train_small_uni.arrow"
        eval_file = out_dir / "eval_uni.arrow"

        if not train_file.exists():
            continue

        if not eval_file.exists():
            continue

        datasets.append({
            "name": out_dir.name,
            "train": str(train_file),
            "eval": str(eval_file),
        })

    return datasets


def insert(conn, cfg, ds):
    h = config_hash(cfg, ds["name"])
    cur = conn.cursor()

    try:
        cur.execute("""
            INSERT INTO runs (
                config_hash,
                config,
                dataset,
                train_data,
                eval_data,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            h,
            json.dumps(cfg),
            ds["name"],
            ds["train"],
            ds["eval"],
            "PENDING"
        ))

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False


def generate(n_configs=20):
    conn = sqlite3.connect(DB_PATH)

    datasets = find_datasets()

    # Generate the HPO configurations once
    configs = [sample_config() for _ in range(n_configs)]

    for dataset in datasets:
        print(f"Generating configs for {dataset['name']}")

        for i, hparams in enumerate(configs, 1):

            # Only sampled HPO parameters are stored in config
            cfg = hparams.copy()

            inserted = insert(conn, cfg, dataset)

            if inserted:
                print(f"Inserted config {i}/{n_configs}")
            else:
                print(f"Exists config {i}/{n_configs}")

    conn.close()


if __name__ == "__main__":
    generate(100)