import sqlite3
import json
import yaml
import random
import numpy as np
import hashlib
import copy
from pathlib import Path


DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/similarity/similarity_cv.db"

BASE_CONFIG_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/similarity/base_config.yaml"

UCR_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018"
)

ARABIC_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Similarity/ArabicSpokenDigits"
)


def config_hash(config: dict, ds: dict) -> str:
    config = config.copy()
    config["dataset"] = ds["name"]

    return hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()


def sample_config():
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
    }


datasets = []

for dataset_dir in sorted(UCR_ROOT.iterdir()):
    if not dataset_dir.is_dir():
        continue

    dataset_name = dataset_dir.name
    train_pairs = dataset_dir / "similarity/train_pairs.npz"

    if not train_pairs.exists():
        print(
            f"WARNING: {dataset_name}: "
            f"train_small_pairs.npz not found -> skipped"
        )
        continue

    datasets.append({
        "name": dataset_name,
        "train": str(train_pairs),
        "task": "similarity"
    })


datasets.extend([
    {
        "name": "ArabicSpokenDigits1",
        "train": str(
            ARABIC_ROOT
            / "similarity"
            / "univariate"
            / "digit"
            / "train_pairs.npz"
        ),
        "task": "digit"
    },
    {
        "name": "ArabicSpokenDigits2",
        "train": str(
            ARABIC_ROOT
            / "similarity"
            / "univariate"
            / "voice"
            / "train_pairs.npz"
        ),
        "task": "voice"
    },
])


def make_config():
    config = sample_config()
    return config


def insert(conn, cfg, ds):
    h = config_hash(cfg, ds)

    cur = conn.cursor()

    try:
        cur.execute("""
            INSERT INTO runs (
                config_hash,
                config,
                dataset,
                train_data,
                task,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            h,
            json.dumps(cfg),
            ds["name"],
            ds["train"],
            ds["task"],
            "PENDING"
        ))

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False


def generate(n_configs=20):

    conn = sqlite3.connect(DB_PATH)

    # SAME sampled HPO configurations for every dataset
    configs = [
        sample_config()
        for _ in range(n_configs)
    ]

    for dataset in datasets:

        print(f"Generating configs for {dataset['name']}")

        for i, hparams in enumerate(configs, 1):

            config = hparams.copy()

            config["dataset"] = dataset["name"]
            config["task"] = dataset["task"]

            inserted = insert(
                conn,
                config,
                dataset
            )

            if inserted:
                print(
                    f"Inserted config {i}/{n_configs}"
                )
            else:
                print(
                    f"Skipped existing config {i}/{n_configs}"
                )

    conn.close()


if __name__ == "__main__":
    generate(100)