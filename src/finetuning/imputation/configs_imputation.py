import sqlite3
import json
import yaml
import random
import numpy as np
import hashlib
import copy

DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/imputation/imputation_allData.db"

def config_hash(config: dict,) -> str:
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
        "warmup_ratio": float(random.uniform(0.0, 0.1)),
        "mean_span_length": int(random.choice([8, 16, 32])),
        "masking_prob": float(random.choice([0.15, 0.3, 0.45])),
    }

datasets = [
    {
        "name": "ETTh1",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Imputation/ETTh1/train_small.npz",
        "eval": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Imputation/ETTh1/eval.npz",
    },
    {
        "name": "ETTh2",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Imputation/ETTh2/train_small.npz",
        "eval": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Imputation/ETTh2/eval.npz",
    },
    {
        "name": "ETTm1",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Imputation/ETTm1/train_small.npz",
        "eval": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Imputation/ETTm1/eval.npz",
    },
    {
        "name": "ETTm2",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Imputation/ETTm2/train_small.npz",
        "eval": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/Imputation/ETTm2/eval.npz",
    },
]



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

    except sqlite3.IntegrityError as e:
        print(e)
        return False


def generate(n_configs=20):
    conn = sqlite3.connect(DB_PATH)

    # Generate the 100 random configs once
    configs = [sample_config() for _ in range(n_configs)]

    for dataset in datasets:
        print(f"Generating configs for {dataset['name']}")

        for i, hparams in enumerate(configs, 1):
            config = hparams.copy()

            insert(conn, config, dataset)

            print(f"Inserted config {i}/{n_configs}")

    conn.close()

if __name__ == "__main__":
    generate(100)