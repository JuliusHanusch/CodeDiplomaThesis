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
from sklearn.metrics import precision_recall_curve


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
        label_dir = Path(dataset_dir) / "test_label"

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

def sliding_window(arr, window_size, stride):
    """Create overlapping windows."""
    for start in range(0, len(arr) - window_size + 1, stride):
        yield start, arr[start:start + window_size]

def evaluate_dataset_no_windowing(
    model,
    dataset_dir,
    dataset_name,
    tokenizer,
    max_length: int = 512,
    threshold: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):

    import numpy as np
    import torch

    model.eval()
    model.to(device)

    series_list, label_list = load_dataset(dataset_dir, dataset_name)

    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():

        for series, labels in zip(series_list, label_list):

            series = np.asarray(series)
            labels = np.asarray(labels)

            if len(series) != len(labels):
                raise ValueError(f"Mismatch: {len(series)} vs {len(labels)}")

            preds_full = np.zeros(len(series))
            scores_full = np.zeros(len(series))

            for start in range(0, len(series), max_length):

                end = min(start + max_length, len(series))

                chunk = torch.tensor(series[start:end], dtype=torch.float32)

                input_ids, attention_mask, _ = tokenizer.context_input_transform(chunk)

                input_ids = input_ids.unsqueeze(0).to(device)
                attention_mask = attention_mask.unsqueeze(0).to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                logits = outputs["logits"].squeeze(0)
                probs = torch.sigmoid(logits).cpu().numpy()

                length = end - start

                scores_full[start:end] = probs[:length]
                preds_full[start:end] = (probs[:length] >= threshold)

            all_scores.append(scores_full)
            all_preds.append(preds_full)
            all_labels.append(labels)

    # --------------------------------------------------
    # FLATTEN (CORRECT ORDER)
    # --------------------------------------------------
    all_scores = np.concatenate(all_scores)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    tn = np.sum((all_preds == 0) & (all_labels == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    # --------------------------------------------------
    # OPTIONAL: threshold sweep (VERY IMPORTANT)
    # --------------------------------------------------
    from sklearn.metrics import precision_recall_curve

    p, r, t = precision_recall_curve(all_labels, all_scores)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pr_curve": (p, r, t),
    }

def evaluate_dataset(
    model,
    dataset_dir,
    dataset_name,
    tokenizer,
    window_size: int = 512,
    stride: int = 128,
    threshold: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):

    model.eval()
    model.to(device)

    series_list, label_list = load_dataset(dataset_dir, dataset_name)

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for series, labels in zip(series_list, label_list):

            series = np.asarray(series)
            labels = np.asarray(labels)

            if len(series) != len(labels):
                raise ValueError(
                    f"Mismatch: series={len(series)} labels={len(labels)}"
                )

            window_preds = np.zeros(len(series))
            window_counts = np.zeros(len(series))

            # --------------------------
            # window inference
            # --------------------------
            for start, window in sliding_window(series, window_size, stride):

                end = start + window_size

                context = torch.tensor(window, dtype=torch.float32)

                input_ids, attention_mask, _ = tokenizer.context_input_transform(context)

                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs["logits"].squeeze(0)
                probs = torch.sigmoid(logits)

                preds = (probs >= threshold).cpu().numpy()

                window_preds[start:end] += preds
                window_counts[start:end] += 1

            # avoid division issues
            window_counts[window_counts == 0] = 1
            final_preds = (window_preds / window_counts) >= 0.5

            # align with labels
            valid_mask = np.ones_like(labels, dtype=bool)

            all_preds.append(final_preds[valid_mask])
            all_labels.append(labels[valid_mask])

    # --------------------------
    # flatten
    # --------------------------
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # --------------------------
    # metrics
    # --------------------------
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    tn = np.sum((all_preds == 0) & (all_labels == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
    

if __name__ == "__main__":

    model_path = (
        "/content/CodeDiplomaThesis/FineTunedModels/Anomaly/run-1/checkpoint-final"
    )

    print("Model PATH:", model_path)
    base_dir = Path(
        "/content/CodeDiplomaThesis/data/finetuning/"
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
        #"SMAP",
        "MSL",
    ]

    rows = []

    for ds in datasets:

        print(f"\n{'='*20}")
        print(f"Evaluating {ds}")
        print(f"{'='*20}")

        metrics = evaluate_dataset_no_windowing(
            model=model,
            tokenizer=tokenizer,
            dataset_dir=base_dir / ds,
            dataset_name=ds,
        )

        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall    : {metrics['recall']:.4f}")
        print(f"F1        : {metrics['f1']:.4f}")
        print(f"Accuracy  : {metrics['accuracy']:.4f}")

        print(
            f"TP: {metrics['tp']} | FP: {metrics['fp']} | "
            f"FN: {metrics['fn']} | TN: {metrics['tn']}"
        )

        rows.append({
            "dataset": ds,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "F1": metrics["f1"],
            "accuracy": metrics["accuracy"],
        })

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