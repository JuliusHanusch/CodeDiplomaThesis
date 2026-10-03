import json
import sqlite3
from pathlib import Path
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
import time
from gluonts.dataset.arrow import ArrowFile


root_dir = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi")
sys.path.append(str(root_dir.resolve()))
sys.path.append(str((root_dir/"src").resolve()))
sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos import ChronosPipeline


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/tser/Final/tser_multivariate.db"

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_arrow(path: Path):
    """
    Reads TS-ARROW dataset using GluonTS ArrowFile reader.
    """

    dataset = ArrowFile(path)

    series = []
    labels = []

    for entry in dataset:
        target = np.asarray(entry["target"], dtype=np.float32)

        if "label" in entry:
            label = entry["label"]
        elif "y" in entry:
            label = entry["y"]
        else:
            raise KeyError("No label found in dataset entry")

        series.append(target)
        labels.append(float(label))

    return np.stack(series), np.array(labels, dtype=np.float32)

def rmse(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return np.sqrt(np.mean((preds - labels) ** 2))


def mae(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return np.mean(np.abs(preds - labels))


def evaluate_chronos(model, tokenizer, test_arrow_path, context_length=512):

    model.eval()

    X, y = load_arrow(test_arrow_path)

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for i in range(len(X)):

            series = X[i]
            label = float(y[i])

            input_ids = []
            attention_masks = []

            for channel in range(series.shape[0]):

                context = torch.tensor(
                    series[channel][-context_length:],
                    dtype=torch.float32
                )

                token_ids, attention_mask, _ = tokenizer.context_input_transform(
                    context.unsqueeze(0)
                )

                input_ids.append(token_ids.squeeze(0))
                attention_masks.append(attention_mask.squeeze(0))

            input_ids = torch.stack(input_ids).unsqueeze(0).to(DEVICE)
            attention_mask = torch.stack(attention_masks).unsqueeze(0).to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            pred = outputs["logits"].detach().cpu().numpy().reshape(-1)[0]

            all_preds.append(pred)
            all_labels.append(label)

    all_preds = np.asarray(all_preds, dtype=np.float32)
    all_labels = np.asarray(all_labels, dtype=np.float32)

    absolute_errors = np.abs(
        all_preds - all_labels
    )

    return {
        "rmse": rmse(all_preds, all_labels),
        "mae": mae(all_preds, all_labels),
        "n_samples": len(all_preds),
        "predictions": all_preds,
        "labels": all_labels,
        "absolute_errors": absolute_errors,
    }

class TSERDatasetMulti(Dataset):

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

        input_ids = []
        attention_masks = []

        for channel in range(series.shape[0]):

            token_ids, attention_mask, _ = self.tokenizer.context_input_transform(
                series[channel].unsqueeze(0)
            )

            input_ids.append(token_ids.squeeze(0))
            attention_masks.append(attention_mask.squeeze(0))

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.tensor(
                self.y[idx],
                dtype=torch.float32
            ),
        }

if __name__ == "__main__":

    print("Start")

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()

    idx = args.index
    seed = args.seed

    set_seed(seed)

    conn = get_db_connection()
    cur = conn.cursor()

    model_path_column = f"model_path_{seed}"

    cur.execute(
        f"""
        SELECT config,
            train_data,
            test_data,
            dataset,
            {model_path_column}
        FROM runs
        WHERE id=?
        """,
        (idx,),
    )

    row = cur.fetchone()

    conn.close()
    config_json, train_data, test_data, dataset_name, model_path = row
    config = json.loads(config_json)


    # Hyperparameters
    batch_size = config["per_device_train_batch_size"]
    num_epochs = config["num_train_epochs"]
    learning_rate = config["learning_rate"]
    dropout_head = config["dropout_head"]
    warmup_ratio = config["warmup_ratio"]
    train_inner = config["TrainInnerModel"]
    loss_type = config["loss_type"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]


    # Load data
    X, y = load_arrow(
        Path(train_data)
    )
    n_channels = X.shape[1]

    # Load TSER model
    pipeline = ChronosPipeline.from_pretrained(
        "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/BertModel/BertSmall/run-2/checkpoint-final",
        task="tser",
        loss_type=loss_type,
        dropout_head=dropout_head,
        TrainInnerModel=train_inner,
        uni_multi = "multi",
        n_channels =  n_channels
    )

    model = pipeline.model.to(DEVICE)
    tokenizer = pipeline.tokenizer
    print("Model Loaded")


    dataset = TSERDatasetMulti(
        X,
        y,
        tokenizer
    )

    print("Dataset Loaded")


    generator = torch.Generator()
    generator.manual_seed(seed)

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

    print("Training Started")


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
            f"{total_loss / len(loader):.4f}"
        )

    start_time = time.time()

    results = evaluate_chronos(
        model=model,
        tokenizer=tokenizer,
        test_arrow_path=Path(test_data),
    )

    inference_time = time.time() - start_time

    output_dir = Path(
        "/data/horse/ws/juha972b-AION-BERT-Chronos/"
        "BERTi/Results/Finetuning/TSER/Predictions"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez(
        output_dir / f"{dataset_name}_seed_{seed}_multivariate.npz",
        y_true=results["labels"],
        predictions=results["predictions"],
        absolute_errors=results["absolute_errors"],
    )

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
            float(results["rmse"]),
            float(results["mae"]),
            float(inference_time),
            idx,
        ),
        description=(
            f"updating results and inference time "
            f"for idx {idx}, seed {seed}"
        )
    )


