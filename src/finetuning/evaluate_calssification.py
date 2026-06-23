import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
import sys
import os
from collections import Counter


ROOT = "/content/CodeDiplomaThesis"
sys.path.append(str(Path(ROOT).resolve()))
from chronos_pkg.src.chronos import ChronosPipeline

LABEL_SHIFT = {
    "GestureMidAirD2": 1,
    "DistalPhalanxTW": 3,
}
# -----------------------------
# LOAD UCR TSV (correct label handling)
# -----------------------------
def load_ucr_tsv(tsv_path, context_length=512):
    df = pd.read_csv(tsv_path, sep="\t", header=None).values

    y = df[:, 0]
    X = df[:, 1:].astype(np.float32)

    # IMPORTANT:
    # UCR labels are often 1-based → shift to 0-based safely
    y = y.astype(int)
    y = y - y.min()

    # pad / truncate
    if X.shape[1] < context_length:
        pad = context_length - X.shape[1]
        X = np.pad(X, ((0, 0), (0, pad)), mode="constant")
    else:
        X = X[:, -context_length:]

    return X, y


# -----------------------------
# EVALUATION
# -----------------------------
def evaluate_model(model, tokenizer, X, y, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    preds_all, labels_all = [], []

    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size), desc="Evaluating"):
            batch_X = X[i:i + batch_size]
            batch_y = y[i:i + batch_size]

            context = torch.tensor(batch_X, dtype=torch.float32)

            input_ids, attention_mask, _ = tokenizer.context_input_transform(context)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs["logits"]

            # safety fix (some models return (B,T,C))
            if logits.ndim == 3:
                logits = logits[:, -1, :]

            preds = torch.argmax(logits, dim=-1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(batch_y)

    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="weighted")

    print("\n=== UCR Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (weighted): {f1:.4f}")
    print("\nReport:")
    print(classification_report(labels_all, preds_all))

    return {
        "accuracy": acc,
        "f1": f1,
        "predictions": preds_all,
        "labels": labels_all,
    }


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":

    model_path = "/content/CodeDiplomaThesis/FineTunedModels/classification/run-9/checkpoint-final/"

    #tsv_path = "/content/CodeDiplomaThesis/data/finetuning/UCR_extracted/UCRArchive_2018/DistalPhalanxTW/DistalPhalanxTW_TEST.tsv"
    #tsv_path = "/content/CodeDiplomaThesis/data/finetuning/UCR_extracted/UCRArchive_2018/ArrowHead/ArrowHead_TEST.tsv"
    tsv_path = "/content/CodeDiplomaThesis/data/finetuning/UCR_extracted/UCRArchive_2018/GestureMidAirD2/GestureMidAirD2_TEST.tsv"


    num_labels = 26
    batch_size = 32
    context_length = 512

    pipeline = ChronosPipeline.from_pretrained(
        model_path,
        task="classification",
        num_labels=num_labels
    )

    model = pipeline.model
    tokenizer = pipeline.tokenizer

    # load trained classifier head if exists
    classifier_path = Path(model_path) / "classifier.pt"
    if classifier_path.exists():
        model.classifier.load_state_dict(
            torch.load(classifier_path, map_location="cpu")
        )

    X_test, y_test = load_ucr_tsv(tsv_path, context_length)

    dataset_name = os.path.basename(tsv_path).replace(".tsv", "")
    shift = LABEL_SHIFT.get(dataset_name, 0)
    y_test = y_test - shift

    results = evaluate_model(model, tokenizer, X_test, y_test, batch_size)

    majority_class = Counter(y_test).most_common(1)[0][0]

    naive_preds = np.full_like(y_test, fill_value=majority_class)

    naive_acc = accuracy_score(y_test, naive_preds)
    naive_f1 = f1_score(y_test, naive_preds, average="weighted")



    summary_df = pd.DataFrame([{
        "dataset": dataset_name,
        "accuracy": results["accuracy"],
        "f1": results["f1"],
        "naive_accuracy": naive_acc,
        "naive_f1": naive_f1
    }])

    summary_df.to_csv(
        "/content/CodeDiplomaThesis/GestureMidAirD2.csv",
        index=False
    )