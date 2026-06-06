from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

import logging
from pathlib import Path

# Include Parent Directory to load packages from
import sys  
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"code").resolve()))  
sys.path.append(str((Path(__file__).parent.parent / "chronos_pkg/src").resolve()))

from transformers import AutoModelForMaskedLM, AutoConfig
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)  # optional, for wide display
pd.set_option("display.max_colwidth", None)
import torch
from tqdm.auto import tqdm

sys.path.append(str(Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/chronos_pkg/src").resolve()))
from chronos import ChronosPipeline, ChronosConfig
from chronos.chronos_bolt import ChronosBoltPipeline
from chronos.chronos_bolt import ChronosBoltModelForForecasting, ChronosBoltConfig

from math import log
from utils import get_model_size
from warnings import warn


import os
import numpy as np

def load_uci_har_test(dataset_path, context_length=512):
    X_path = os.path.join(dataset_path, "X_test.txt")
    y_path = os.path.join(dataset_path, "y_test.txt")

    X = np.loadtxt(X_path).astype(np.float32)
    y = np.loadtxt(y_path).astype(int) - 1  # make labels 0-based

    # pad / truncate to context_length
    if X.shape[1] < context_length:
        pad_width = context_length - X.shape[1]
        X = np.pad(X, ((0, 0), (0, pad_width)), mode="constant")
    elif X.shape[1] > context_length:
        X = X[:, -context_length:]

    return X, y


def evaluate_model(model, tokenizer, X, y, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    num_samples = len(X)

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Evaluating"):
            batch_X = X[i:i + batch_size]
            batch_y = y[i:i + batch_size]

            # ---- tokenize batch ----
            context = torch.tensor(batch_X, dtype=torch.float32)    
            # tokenizer expects (B, T)
            input_ids, attention_mask, _ = tokenizer.context_input_transform(context)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # ---- forward pass ----
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs["logits"]
            # ---- handle shape safely ----
            if logits.ndim == 3:
                # e.g. (B, T, num_labels) → take last token
                logits = logits[:, -1, :]

            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y)

    # ---- metrics ----
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (weighted): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    # at the end of evaluate_model()

    return {
        "accuracy": acc,
        "f1": f1,
        "predictions": all_preds,
        "labels": all_labels,
    }

def save_results_to_csv(results, model_path, output_path="evaluation_results.csv"):
    import pandas as pd
    import os

    row = {
        "model_path": model_path,
        "accuracy": results["accuracy"],
        "f1_weighted": results["f1"],
    }

    df = pd.DataFrame([row])

    # append if file exists
    if os.path.exists(output_path):
        df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    model_path = "/FineTunedModels/Classification/BertDefault/run-2/checkpoint-final"
    dataset_path = "data/finetuning/UCI_HAR/UCI HAR Dataset/test"
    num_labels = 6
    batch_size = 32

    pipeline = ChronosPipeline.from_pretrained(
        model_path,
        task="classification",
        num_labels=num_labels
    )
    model = pipeline.model

    classifier_path = Path(model_path) / "classifier.pt"

    model.classifier.load_state_dict(
        torch.load(classifier_path, map_location="cpu")
    )

    tokenizer = pipeline.tokenizer

    X_test, y_test = load_uci_har_test(dataset_path)
    


    # ---- Evaluate ----
    results = evaluate_model(model, tokenizer, X_test, y_test, batch_size)

    save_results_to_csv(
        results,
        model_path,
        output_path="/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/Results/Finetuning/Classification/uci_har_eval_results.csv"
    )
