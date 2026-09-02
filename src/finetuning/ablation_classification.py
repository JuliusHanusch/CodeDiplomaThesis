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

datasets = [
    # EXAMPLE:
    #     {
    #     "name": "UCI-HAR",
    #     "train":"/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCI_HAR/UCI HAR Dataset/train/",
    #     "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCI_HAR/UCI HAR Dataset/test/",
    #     "labels": 6
    # },
    # {
    #     "name": "ArrowHead",
    #     "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/ArrowHead/ArrowHead_TRAIN.tsv",
    #     "test":  "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/ArrowHead/ArrowHead_TEST.tsv",
    #     "labels": 3
    # },
    # {
    #     "name": "DistalPhalanxTW",
    #     "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/DistalPhalanxTW/DistalPhalanxTW_TRAIN.tsv",
    #     "test": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/DistalPhalanxTW/DistalPhalanxTW_TEST.tsv",
    #     "labels": 6
    # },
    {
        "name": "GestureMidAirD2",
        "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/GestureMidAirD2/GestureMidAirD2_TRAIN.tsv",
        "test":  "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/GestureMidAirD2/GestureMidAirD2_TEST.tsv",
        "labels": 26
    },
    # {
    #     "name": "Wafer",
    #     "train": "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/Wafer/Wafer_TRAIN.tsv",
    #     "test":  "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018/Wafer/Wafer_TEST.tsv",
    #     "labels": 2
    # },
]


def load_ucr_tsv(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", header=None).values

    y = df[:, 0]
    X = df[:, 1:].astype(np.float32)

    y = y.astype(int)
    unique = np.unique(y)
    label_map = {v: i for i, v in enumerate(unique)}
    y = np.vectorize(label_map.get)(y)

    return X, y




def load_uci_har(train_dir, test_dir):
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)

    X_train = np.loadtxt(train_dir / "X_train.txt").astype(np.float32)
    y_train = np.loadtxt(train_dir / "y_train.txt").astype(int) - 1

    X_test = np.loadtxt(test_dir / "X_test.txt").astype(np.float32)
    y_test = np.loadtxt(test_dir / "y_test.txt").astype(int) - 1

    return X_train, y_train, X_test, y_test


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

    for dataset_info in datasets:

        print("\n==============================")
        print(dataset_info["name"])
        print("==============================")
 

        dataset = dataset_info["name"]
        train_data = dataset_info["train"]
        test_data = dataset_info["test"]


        #Set HPs
        num_labels = 26
        num_epochs = 20
        batch_size = 32
        learning_rate = 0.0005751958831639615
        dropout_head = 0.27983761176689764
        warmup_ratio = 0.09461253863631648
        train_inner = True


        #random inti
        # model_id="prajjwal1/bert-small"
        # random_init=True
        # task="classification"

        # AutoModelClass = AutoModelForMaskedLM
        # config = AutoConfig.from_pretrained(model_id)
        # inner_model = AutoModelClass.from_config(config)
        # inner_model.resize_token_embeddings(4096)

        # chronos_config = ChronosConfig(
        #     tokenizer_class="MeanScaleUniformBins",
        #     tokenizer_kwargs={
        #         "low_limit": -15.0,
        #         "high_limit": 15.0,
        #     },
        #     n_tokens=4096,
        #     n_special_tokens=4,
        #     pad_token_id=0,
        #     bos_token_id=1,
        #     eos_token_id=2,
        #     mask_token_id=3,
        #     use_eos_token=True,
        #     model_type="mlm",
        #     context_length=512,
        #     prediction_length=64,
        #     num_samples=20,
        #     temperature=1.0,
        #     top_k=50,
        #     top_p=1.0,
        # )        

        # random_model = ChronosModelForClassification(
        #     config=chronos_config,
        #     model=inner_model,
        #     num_labels=num_labels,
        #     dropout_head = dropout_head
        # )

        # random_tokenizer=chronos_config.create_tokenizer()
        # random_model.config.chronos_config = chronos_config.__dict__
        # random_model = random_model.to(DEVICE)



        # #HPO Model 
        # pipeline_optimized = ChronosPipeline.from_pretrained(
        #     "juliushanusch/ChronosBERT-Optimized",
        #     task="classification",
        #     num_labels=num_labels,
        #     dropout_head=dropout_head,
        #     TrainInnerModel=train_inner,
        # )


        # optimized_model = pipeline_optimized.model.to(DEVICE)
        # optimized_tokenizer = pipeline_optimized.tokenizer

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


        #Default Model
        pipeline_base = ChronosPipeline.from_pretrained(
            "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/BertModel/BertBase200k/run-2/checkpoint-final",
            task="classification",
            num_labels=num_labels,
            dropout_head=dropout_head,
            TrainInnerModel=train_inner,
        )

        base_model = pipeline_default.model.to(DEVICE)
        base_tokenizer = pipeline_default.tokenizer

        # #Checkpoins BERT BASE
        # Bert10kpipeline = ChronosPipeline.from_pretrained(
        #     "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/BertModel/BertBaseLong/run-0/checkpoint-10000",
        #     task="classification",
        #     num_labels=num_labels,
        #     dropout_head=dropout_head,
        #     TrainInnerModel=train_inner,
        # )

        # Bert10k = Bert10kpipeline.model.to(DEVICE)
        # k10_tokenizer = Bert10kpipeline.tokenizer

        # #30k 
        # Bert30kpipeline = ChronosPipeline.from_pretrained(
        #     "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/BertModel/BertBaseLong/run-0/checkpoint-30000",
        #     task="classification",
        #     num_labels=num_labels,
        #     dropout_head=dropout_head,
        #     TrainInnerModel=train_inner,
        # )

        # Bert30k = Bert30kpipeline.model.to(DEVICE)
        # k30_tokenizer = Bert30kpipeline.tokenizer

        # #90k
        # Bert90kpipeline = ChronosPipeline.from_pretrained(
        #     "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/BertModel/BertBaseLong/run-0/checkpoint-90000",
        #     task="classification",
        #     num_labels=num_labels,
        #     dropout_head=dropout_head,
        #     TrainInnerModel=train_inner,
        # )

        # Bert90k = Bert90kpipeline.model.to(DEVICE)
        # k90_tokenizer = Bert90kpipeline.tokenizer

        # #150k
        # Bert150kpipeline = ChronosPipeline.from_pretrained(
        #     "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/BertModel/BertBaseLong/run-0/checkpoint-150000",
        #     task="classification",
        #     num_labels=num_labels,
        #     dropout_head=dropout_head,
        #     TrainInnerModel=train_inner,
        # )

        # Bert150k = Bert150kpipeline.model.to(DEVICE)
        # k150_tokenizer = Bert150kpipeline.tokenizer

        # #300k
        # Bert300kpipeline = ChronosPipeline.from_pretrained(
        #     "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/BertModel/BertBaseLong/run-0/checkpoint-300000",
        #     task="classification",
        #     num_labels=num_labels,
        #     dropout_head=dropout_head,
        #     TrainInnerModel=train_inner,
        # )

        # Bert300k = Bert300kpipeline.model.to(DEVICE)
        # k300_tokenizer = Bert300kpipeline.tokenizer

        if dataset == "UCI-HAR":
             X_train, y_train, X_test, y_test = load_uci_har(train_data, test_data)
        else:
            X_train, y_train = load_ucr_tsv(train_data)
            X_test, y_test = load_ucr_tsv(test_data)

        models = [
            #("Random", random_model, random_tokenizer),
            #("Optimized", optimized_model, optimized_tokenizer),
            ("Default", default_model, default_tokenizer),
            ("Base", base_model, base_tokenizer),
            #("10k", Bert10k, k10_tokenizer),
            #("30k", Bert30k, k30_tokenizer),
            #("90k", Bert90k, k90_tokenizer),
            #("150k", Bert150k, k150_tokenizer),
            #("300k", Bert300k, k300_tokenizer),


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


    csv_path = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/Results/Finetuning/Classification/Experiment2.csv"

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
                    