import json
import sqlite3
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForCausalLM,
    AutoConfig,
    T5Config,
    BertConfig,
    Trainer,
    TrainingArguments,  
    AutoModelForMaskedLM,
)



root_dir = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi")
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"src").resolve()))  
sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos.chronos_classification import ChronosModelForClassification
from chronos_pkg.src.chronos import ChronosConfig, ChronosTokenizer, ChronosPipeline
from chronos_pkg.src.chronos.chronos_bolt import ChronosBoltModelForForecasting, ChronosBoltConfig


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/"
    "data/finetuning/UCR_extracted/UCRArchive_2018"
)


# Fungi is excluded because it cannot be split correctly
# while keeping every class in the training set.
EXCLUDED_DATASETS = {
    "Fungi",
}


def get_datasets():

    datasets = []

    for dataset_dir in sorted(DATA_ROOT.iterdir()):

        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name

        if dataset in EXCLUDED_DATASETS:
            print(f"Skipping {dataset}")
            continue

        train_path = dataset_dir / f"{dataset}_TRAIN.tsv"
        test_path = dataset_dir / f"{dataset}_TEST.tsv"

        if not train_path.exists():
            print(
                f"[WARNING] Missing train file for {dataset}: "
                f"{train_path}"
            )
            continue

        if not test_path.exists():
            print(
                f"[WARNING] Missing test file for {dataset}: "
                f"{test_path}"
            )
            continue

        datasets.append({
            "name": dataset,
            "train": str(train_path),
            "test": str(test_path),
        })

    return datasets



def load_ucr_tsv(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", header=None).values

    y = df[:, 0]
    X = df[:, 1:].astype(np.float32)

    y = y.astype(int)
    unique = np.unique(y)
    label_map = {v: i for i, v in enumerate(unique)}
    y = np.vectorize(label_map.get)(y)

    return X, y




class ClassificationDataset(Dataset):

    def __init__(self, X, y, tokenizer):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        series = torch.tensor(self.X[idx], dtype=torch.float32)

        token_ids, attention_mask, _ = self.tokenizer.context_input_transform(
            series.unsqueeze(0)
        )

        return {
            "input_ids": token_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": torch.tensor(self.y[idx], dtype=torch.long),
        }
    
def train_model(model, train_loader, epochs, lr, warmup_ratio):
    model.train()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    for epoch in range(epochs):
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )

            loss = outputs["loss"]
            loss.backward()

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: {total_loss/len(train_loader):.4f}")

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

@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()

    preds = []
    labels = []

    for batch in loader:

        outputs = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
        )

        pred = outputs["logits"].argmax(dim=-1)

        preds.extend(pred.cpu().numpy())
        labels.extend(batch["labels"].numpy())

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }

if __name__ == "__main__":

    results = []
    datasets = get_datasets()


    for dataset_info in datasets:

        dataset = dataset_info["name"]
        train_data = dataset_info["train"]
        test_data = dataset_info["test"]

        X_train, y_train = load_ucr_tsv(train_data)
        X_test, y_test = load_ucr_tsv(test_data)
 

        dataset = dataset_info["name"]
        train_data = dataset_info["train"]
        test_data = dataset_info["test"]


        #Set HPs
        num_labels = get_num_labels(train_data)
        num_epochs = 10
        batch_size = 32
        learning_rate = 1e-4
        dropout_head = 0.1
        warmup_ratio = 0.1
        train_inner = True

        model_id = "prajjwal1/bert-small"

        # Load ONLY the configuration
        config = AutoConfig.from_pretrained(model_id)

        # Create model from config -> RANDOM WEIGHTS
        inner_model = AutoModelForMaskedLM.from_config(config)

        for param in inner_model.parameters():
            param.requires_grad = train_inner

        # Create Chronos classification model
        from chronos_pkg.src.chronos.chronos_classification import (
            ChronosModelForClassification
        )

        chronos_config = ChronosConfig(
            model_type="mlm",
            tokenizer_class="MeanScaleUniformBins",
            tokenizer_kwargs={
                "low_limit": -15.0,
                "high_limit": 15.0,
            },
            context_length=512,
            prediction_length=64,
            n_tokens=4096,
            n_special_tokens=4,
            bos_token_id=1,
            pad_token_id=0,
            eos_token_id=2,
            use_eos_token=True,
            num_samples=20,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
        )

        random_model = ChronosModelForClassification(
            config=chronos_config,
            model=inner_model,
            num_labels=num_labels,
            dropout_head=dropout_head,
        ).to(DEVICE)

        # Chronos tokenizer
        random_tokenizer = chronos_config.create_tokenizer()


        #Default Model
        pipeline_default = ChronosPipeline.from_pretrained(
            "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/BertModel/BertSmall/run-2/checkpoint-final",
            task="classification",
            num_labels=num_labels,
            dropout_head=dropout_head,
            TrainInnerModel=train_inner,
        )

        default_model = pipeline_default.model.to(DEVICE)
        default_tokenizer = pipeline_default.tokenizer

        #Optimal Model
        pipeline_opt = ChronosPipeline.from_pretrained(
            "juliushanusch/ChronosBERT-Optimized",
            task="classification",
            num_labels=num_labels,
            dropout_head=dropout_head,
            TrainInnerModel=train_inner,
        )

        opt_model = pipeline_opt.model.to(DEVICE)
        opt_tokenizer = pipeline_opt.tokenizer


        #Default Model
        # pipeline_base = ChronosPipeline.from_pretrained(
        #     "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/BertModel/BertBase400k/run-0/checkpoint-final",
        #     task="classification",
        #     num_labels=num_labels,
        #     dropout_head=dropout_head,
        #     TrainInnerModel=train_inner,
        # )

        # base_model = pipeline_base.model.to(DEVICE)
        # base_tokenizer = pipeline_base.tokenizer

       

        models = [
            ("Dafault", default_model, default_tokenizer),
            #("Base", base_model, base_tokenizer),
            ("Optimized", opt_model, opt_tokenizer),
            ("Random", random_model, random_tokenizer),





        ]

        for model_name, model, tokenizer in models:
            train_dataset = ClassificationDataset(X_train, y_train, tokenizer)
            test_dataset = ClassificationDataset(X_test, y_test, tokenizer)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            print(f"\nTraining {model_name}")

            train_model(
                model,
                train_loader,
                num_epochs,
                learning_rate,
                warmup_ratio,
            )

            metrics = evaluate_model(model, test_loader)

            metrics["dataset"] = dataset
            metrics["model"] = model_name

            results.append(metrics)

            print(metrics)

    results_df = pd.DataFrame(results)

    # Pivot models into columns
    results_df = results_df.pivot(
        index="dataset",
        columns="model",
        values=["accuracy"]
    )

    # Flatten multi-index columns
    results_df.columns = [
        f"{model}_{metric}"
        for metric, model in results_df.columns
    ]

    results_df = results_df.reset_index()


    # Add average row
    avg_row = {"dataset": "Average"}

    for col in results_df.columns:
        if col != "dataset":
            avg_row[col] = results_df[col].mean()

    results_df = pd.concat(
        [results_df, pd.DataFrame([avg_row])],
        ignore_index=True
    )


    csv_path = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/Results/Finetuning/Classification/DefaultvsOptsRandom_10Epochs.csv"

    Path(csv_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    print(results_df)
    print(f"Saved results to {csv_path}")
                    