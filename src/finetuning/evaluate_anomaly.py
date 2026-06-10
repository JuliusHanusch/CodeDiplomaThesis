from pathlib import Path

# Include Parent Directory to load packages from
import sys  
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)  # optional, for wide display
pd.set_option("display.max_colwidth", None)
import torch
import ast

#colab import
ROOT = "/content/CodeDiplomaThesis"
sys.path.append(str(Path(ROOT).resolve()))

from chronos_pkg.src.chronos import ChronosPipeline
from sklearn.metrics import (
    precision_recall_fscore_support,
)


def point_adjust(pred, gt):
    pred = pred.copy()

    in_anomaly = False
    start = 0

    for i in range(len(gt)):
        if gt[i] == 1 and not in_anomaly:
            start = i
            in_anomaly = True

        if gt[i] == 0 and in_anomaly:
            end = i

            if pred[start:end].sum() > 0:
                pred[start:end] = 1

            in_anomaly = False

    if in_anomaly:
        end = len(gt)

        if pred[start:end].sum() > 0:
            pred[start:end] = 1

    return pred


def intervals_to_mask(intervals, length):
    mask = np.zeros(length, dtype=np.int32)
    for start, end in intervals:
        mask[start:end] = 1
    return mask


def load_label_file(path: Path, length: int):
    raw = open(path, "r").read().strip()

    # --------------------------
    # CASE 1: SMD (binary mask)
    # --------------------------
    if raw[0] in ["0", "1"] and "[" not in raw:
        y = np.loadtxt(path, dtype=np.int32).reshape(-1)

        if len(y) < length:
            y = np.pad(y, (0, length - len(y)))
        else:
            y = y[:length]

        return y.astype(np.int32)

    # --------------------------
    # CASE 2: SMAP / MSL (intervals)
    # --------------------------
    try:
        intervals = ast.literal_eval(raw)
    except Exception:
        raise ValueError(f"Cannot parse label file: {path}")

    if len(intervals) == 0:
        return np.zeros(length, dtype=np.int32)

    return intervals_to_mask(intervals, length)



def load_dataset(dataset_dir, dataset_name):

    test_dir = Path(dataset_dir) / "test"
    label_dir = Path(dataset_dir) / "test_labels"

    series = []
    labels = []

    if dataset_name == "SMD":

        test_files = sorted(test_dir.glob("*.txt"))
        label_files = sorted(label_dir.glob("*.txt"))

    else:

        test_files = sorted(test_dir.glob("*.npy"))
        label_files = sorted(label_dir.glob("*.txt"))

    for x_file, y_file in zip(test_files, label_files):

        # --------------------------
        # load series
        # --------------------------
        if x_file.suffix == ".txt":
            x = np.loadtxt(x_file, delimiter=",").astype(np.float32)
        else:
            x = np.load(x_file).astype(np.float32)

        if x.ndim > 1:
            x = x.mean(axis=1)

        x = x.reshape(-1)

        # --------------------------
        # load labels (FIXED)
        # --------------------------
        y = load_label_file(y_file, len(x))

        series.append(x)
        labels.append(y)

    return series, labels

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict_series(
    model,
    tokenizer,
    series,
    context_length=512,
    stride=128,
    threshold=0.5,
):
    device = next(model.parameters()).device

    scores = np.zeros(len(series))
    counts = np.zeros(len(series))

    model.eval()

    with torch.no_grad():

        for start in range(0, len(series) - context_length + 1, stride):

            window = series[start:start + context_length]

            context = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

            input_ids, attention_mask, _ = tokenizer.context_input_transform(context)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs["logits"]   # (1, T)

            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            print("logits min/max:", logits.min().item(), logits.max().item())
            print("probs min/max:", probs.min(), probs.max())
            print("mean prob:", probs.mean())

            mask = attention_mask.squeeze(0).cpu().numpy().astype(bool)

            probs = probs * mask  # safety: ignore padding

            scores[start:start + context_length] += probs
            counts[start:start + context_length] += mask

    counts = np.clip(counts, 1, None)
    scores = scores / counts

    print("score range:", scores.min(), scores.max())

    pred = (scores > threshold).astype(np.int32)

    print("pred positives:", pred.sum(), "/", len(pred))

    return pred, scores

def evaluate_dataset(
    model,
    tokenizer,
    dataset_dir,
    dataset_name,
):

    series_list, label_list = load_dataset(
        dataset_dir,
        dataset_name,
    )

    all_gt = []
    all_pred = []

    all_gt_pa = []
    all_pred_pa = []

    print(f"\n[DEBUG] Dataset: {dataset_name}")
    print(f"[DEBUG] Num series: {len(series_list)}")
    print(f"[DEBUG] Example GT shape: {label_list[0].shape}")
    print(f"[DEBUG] GT positives total (first 1000 samples): {np.sum(label_list[0][:1000])}")
    print(f"[DEBUG] GT ratio (first series): {np.mean(label_list[0])}")

    for i, (series, gt) in enumerate(zip(series_list, label_list)):

        pred, scores = predict_series(model, tokenizer, series)

        min_len = min(len(gt), len(pred))
        gt = gt[:min_len]
        pred = pred[:min_len]

        all_gt.extend(gt)
        all_pred.extend(pred)

        pred_pa = point_adjust(pred, gt)

        all_gt_pa.extend(gt)
        all_pred_pa.extend(pred_pa)

        print("\n--- Series debug ---")
        print(f"[Series {i}]")
        print("GT positives:", np.sum(gt))
        print("GT ratio:", np.mean(gt))
        print("Pred positives:", np.sum(pred))
        print("Pred ratio:", np.mean(pred))
        print("Score range:", scores.min(), scores.max())
        print("Overlap:", np.sum((gt == 1) & (pred == 1)))
    
    print("\n=== Dataset summary ===")
    print("GT anomaly ratio:", np.mean(all_gt))
    print("Pred anomaly ratio:", np.mean(all_pred))
    print("Total overlap:", np.sum((np.array(all_gt) == 1) & (np.array(all_pred) == 1)))

    _, _, f1, _ = precision_recall_fscore_support(
        all_gt,
        all_pred,
        average="binary",
        zero_division=0,
    )

    _, _, f1_pa, _ = precision_recall_fscore_support(
        all_gt_pa,
        all_pred_pa,
        average="binary",
        zero_division=0,
    )


    return {
        "F1": f1,
        "F1PA": f1_pa,
    }


if __name__ == "__main__":

    model_path = (
        "/content/CodeDiplomaThesis/FineTunedModels/Anomaly/run-0/checkpoint-final"
    )

    base_dir = Path(
        "/content/DiplomaThesis/data/finetuning/"
    )

    # --------------------------------------------------
    # Load pipeline
    # --------------------------------------------------

    pipeline = ChronosPipeline.from_pretrained(
        model_path,
        task="anomaly",
    )

    model = pipeline.model

    anomaly_path = Path(model_path) / "anomaly.pt"

    model.classifier.load_state_dict(
        torch.load(
            anomaly_path,
            map_location="cpu",
        )
    )

    tokenizer = pipeline.tokenizer

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    # --------------------------------------------------
    # Evaluate datasets
    # --------------------------------------------------

    datasets = [
        "SMAP",
        "MSL",
        "SMD",
    ]

    rows = []

    for ds in datasets:

        print(f"\n{'='*20}")
        print(f"Evaluating {ds}")
        print(f"{'='*20}")


        metrics = evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_dir=base_dir / ds,
            dataset_name=ds,
        )

        print(
            f"F1    : {metrics['F1']:.4f}"
        )

        print(
            f"F1-PA : {metrics['F1PA']:.4f}"
        )

        rows.append({
            "dataset": ds,
            "F1": metrics["F1"],
            "F1PA": metrics["F1PA"],
        })

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    results = pd.DataFrame(rows)

    print("\nResults")
    print(results)

    output_file = (
        "/Results/Finetuning/Anomaly/"
        "anomaly_eval_results.csv"
    )

    Path(output_file).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nSaved results to {output_file}"
    )