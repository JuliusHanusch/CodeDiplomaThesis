import json
import sqlite3
import random
import time
from pathlib import Path
import sys
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import mean_absolute_error, mean_squared_error

from gluonts.dataset.arrow import ArrowFile


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi"
)

DB_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/tser/Final/tser_bestConfigs_time.db"
)

BASE_MODEL_PATH = (
    ROOT_DIR
    / "BertModel/BertSmall/run-2/checkpoint-final"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "Results/TSER"
)

CSV_PATH = OUTPUT_DIR / "default_results.csv"


# ============================================================
# Imports from local Chronos package
# ============================================================

sys.path.append(str(ROOT_DIR.resolve()))
sys.path.append(str((ROOT_DIR / "src").resolve()))
sys.path.append(str((ROOT_DIR / "chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos import ChronosPipeline


# ============================================================
# Device / seed
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEFAULT CONFIG
# ============================================================

CONFIG = {

    # Training
    "per_device_train_batch_size": 32,
    "num_train_epochs": 20,
    "learning_rate": 1e-4,

    # Regression head
    "dropout_head": 0.1,

    # Scheduler
    "warmup_ratio": 0.1,

    # Chronos-BERT
    "TrainInnerModel": True,

    # Loss
    "loss_type": "mse",

    # Gradient accumulation
    "gradient_accumulation_steps": 1,



}


# ============================================================
# Dataset
# ============================================================

def load_arrow(path: Path):

    dataset = ArrowFile(path)

    series = []
    labels = []

    for entry in dataset:

        target = np.asarray(
            entry["target"],
            dtype=np.float32
        )

        if "label" in entry:
            label = entry["label"]

        elif "y" in entry:
            label = entry["y"]

        else:
            raise KeyError(
                f"No label found in dataset entry: {path}"
            )

        series.append(target)
        labels.append(float(label))

    return (
        np.stack(series),
        np.asarray(labels, dtype=np.float32)
    )


class TSERDataset(Dataset):

    def __init__(self, X, y, tokenizer):

        self.X = X
        self.y = y
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        series = torch.tensor(
            self.X[idx],
            dtype=torch.float32
        )

        token_ids, attention_mask, _ = (
            self.tokenizer.context_input_transform(
                series.unsqueeze(0)
            )
        )

        return {
            "input_ids": token_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": torch.tensor(
                self.y[idx],
                dtype=torch.float32
            ),
        }


# ============================================================
# Database
# ============================================================

def get_datasets_from_db():

    conn = sqlite3.connect(DB_PATH)

    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            dataset,
            train_data,
            test_data
        FROM runs
        WHERE dataset IS NOT NULL
          AND train_data IS NOT NULL
          AND test_data IS NOT NULL
        GROUP BY dataset
        ORDER BY dataset
        """
    )

    rows = cur.fetchall()

    conn.close()

    return rows


def train_model(model, tokenizer, train_path, config):

    X, y = load_arrow(Path(train_path))

    dataset = TSERDataset(X, y, tokenizer)

    batch_size = config["per_device_train_batch_size"]
    num_epochs = config["num_train_epochs"]
    learning_rate = config["learning_rate"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    warmup_ratio = config["warmup_ratio"]

    generator = torch.Generator()
    generator.manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    optimizer = AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=learning_rate,
    )

    import math

    steps_per_epoch = math.ceil(
        len(loader) / gradient_accumulation_steps
    )

    total_steps = num_epochs * steps_per_epoch

    warmup_steps = int(
        total_steps * warmup_ratio
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()

    optimizer.zero_grad()

    for epoch in range(num_epochs):

        total_loss = 0.0

        for step, batch in enumerate(loader):

            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )

            loss = outputs["loss"]

            # Keep the unscaled loss for logging
            total_loss += loss.item()

            # Scale loss before accumulating gradients
            loss = loss / gradient_accumulation_steps
            loss.backward()

            # Update weights only after accumulating gradients
            if (
                (step + 1) % gradient_accumulation_steps == 0
                or (step + 1) == len(loader)
            ):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        print(
            f"Epoch {epoch + 1}: "
            f"{total_loss / len(loader):.4f}",
            flush=True
        )


def rmse(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return np.sqrt(np.mean((preds - labels) ** 2))


def mae(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return np.mean(np.abs(preds - labels))

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



# ============================================================
# Save model
# ============================================================

def save_model(
    model,
    output_path,
    config,
):

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save BERT model
    model.model.save_pretrained(
        output_path
    )

    # Save regression head
    torch.save(
        model.regressor.state_dict(),
        output_path / "regressor.pt"
    )

    # Save config
    with open(
        output_path / "training_config.json",
        "w"
    ) as f:

        json.dump(
            config,
            f,
            indent=2
        )


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    datasets = get_datasets_from_db()

    print(
        f"Found {len(datasets)} datasets"
    )

    for dataset_name, train_path, test_path in datasets:


        print(f"Train: {train_path}")
        print(f"Eval:  {test_path}")

        start_time = time.time()

        # ----------------------------------------------------
        # Load fresh model for every dataset
        # ----------------------------------------------------

        print()
        print("Loading base model...")

        pipeline = ChronosPipeline.from_pretrained(
            str(BASE_MODEL_PATH),
            task="tser",
            loss_type=CONFIG["loss_type"],
            dropout_head=CONFIG["dropout_head"],
            TrainInnerModel=CONFIG["TrainInnerModel"],
        )

        model = pipeline.model
        tokenizer = pipeline.tokenizer

        model.to(DEVICE)

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        print()
        print("Starting training...")

        train_model(
            model=model,
            tokenizer=tokenizer,
            train_path=train_path,
            config=CONFIG,
        )


        results = evaluate_chronos(
            model=model,
            tokenizer=tokenizer,
            test_arrow_path=test_path,
            context_length=512,
        )

        mae = results["mae"]
        rmse = results["rmse"]

        elapsed = (
            time.time() - start_time
        )


        safe_dataset_name = (
            str(dataset_name)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        model_output = (
            OUTPUT_DIR
            / safe_dataset_name
        )


        result = {
            "dataset": dataset_name,

            "MAE": mae,
            "RMSE": rmse,

            "time_seconds": elapsed,
        }


        if CSV_PATH.exists():

            existing = pd.read_csv(
                CSV_PATH
            )

            existing = pd.concat(
                [
                    existing,
                    pd.DataFrame([result])
                ],
                ignore_index=True
            )

        else:

            existing = pd.DataFrame(
                [result]
            )

        existing.to_csv(
            CSV_PATH,
            index=False
        )


        # Explicitly free model before next dataset
        del model
        del pipeline
        del tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print("=" * 70)
    print("ALL DATASETS FINISHED")
    print("=" * 70)

    print(
        f"Results written to:\n{CSV_PATH}"
    )


if __name__ == "__main__":
    main()