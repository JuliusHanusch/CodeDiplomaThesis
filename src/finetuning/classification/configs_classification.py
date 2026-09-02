import sqlite3
import json
import yaml
import random
import numpy as np
import hashlib
from pathlib import Path


DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/classification_allData.db"
UCR_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/"
    "data/finetuning/UCR_extracted/UCRArchive_2018"
)

def run_hash(config: dict, dataset: str) -> str:
    data = {
        "config": config,
        "dataset": dataset,
    }

    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
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

def find_ucr_datasets():
    """
    Find all UCR datasets containing:
        *_TRAIN_small.tsv
        *_EVAL.tsv
        *_TEST.tsv

    Only TRAIN_small and EVAL are stored in the DB.
    TEST remains untouched and is used separately for
    the final evaluation.
    """

    datasets = []

    for dataset_dir in sorted(UCR_ROOT.iterdir()):

        if not dataset_dir.is_dir():
            continue

        train_small_files = list(
            dataset_dir.glob("*_TRAIN_small.tsv")
        )

        eval_files = list(
            dataset_dir.glob("*_EVAL.tsv")
        )

        test_files = list(
            dataset_dir.glob("*_TEST.tsv")
        )

        if len(train_small_files) != 1:
            print(
                f"[SKIP] {dataset_dir.name}: "
                f"expected 1 TRAIN_small file, "
                f"found {len(train_small_files)}"
            )
            continue

        if len(eval_files) != 1:
            print(
                f"[SKIP] {dataset_dir.name}: "
                f"expected 1 EVAL file, "
                f"found {len(eval_files)}"
            )
            continue

        if len(test_files) != 1:
            print(
                f"[SKIP] {dataset_dir.name}: "
                f"expected 1 TEST file, "
                f"found {len(test_files)}"
            )
            continue

        train_small_file = train_small_files[0]
        eval_file = eval_files[0]

        dataset_name = train_small_file.name.replace(
            "_TRAIN_small.tsv", ""
        )

        datasets.append({
            "name": dataset_name,
            "train": str(train_small_file),
            "eval": str(eval_file),
        })

    return datasets

def get_num_labels(train_file):
    """
    Determine number of classes from the first column
    of the UCR TRAIN_small file.
    """

    labels = set()

    with open(train_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            label = line.split("\t")[0]
            labels.add(label)

    return len(labels)


def make_config(dataset):
    hparams = sample_config()

    config = hparams.copy()
    config["num_labels"] = int(dataset["labels"])

    return config


def insert(conn, cfg, ds):

    h = run_hash(cfg, ds["name"])
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


def generate(n_configs=100):

    datasets = find_ucr_datasets()

    conn = sqlite3.connect(DB_PATH)

    configs = [
        sample_config()
        for _ in range(n_configs)
    ]

    for dataset in datasets:

        num_labels = get_num_labels(
            dataset["train"]
        )

        dataset["labels"] = num_labels

        print(
            f"{dataset['name']}: "
            f"{num_labels} classes"
        )

        for i, hparams in enumerate(configs, 1):

            config = hparams.copy()

            config["num_labels"] = int(
                dataset["labels"]
            )

            inserted = insert(
                conn,
                config,
                dataset
            )

            if inserted:
                print(
                    f"  Inserted config "
                    f"{i}/{n_configs}"
                )
            else:
                print(
                    f"  Exists config "
                    f"{i}/{n_configs}"
                )

    conn.close()


if __name__ == "__main__":
    generate(100)