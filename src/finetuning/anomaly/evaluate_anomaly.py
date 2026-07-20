from pathlib import Path
import sqlite3
import json
from pathlib import Path
import pickle

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
from sklearn.metrics import precision_recall_curve

root_dir = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi")
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"src").resolve()))  
sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos import ChronosPipeline


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

    model.eval()
    model.to(device)

    series_list, label_list = load_dataset(dataset_dir, dataset_name)

    # with open(
    #     dataset_dir / "scaler.pkl",
    #     "rb"
    # ) as f:
    #     scaler = pickle.load(f)


    # series_list = [
    #     scaler.transform(
    #         s.reshape(-1,1)
    #     ).reshape(-1)
    #     for s in series_list
    # ]

    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():

        for series, labels in zip(series_list, label_list):

            series = np.asarray(series)
            labels = np.asarray(labels)

            if len(series) != len(labels):
                raise ValueError(
                    f"Mismatch: {len(series)} vs {len(labels)}"
                )

            preds_full = np.zeros(len(series))
            scores_full = np.zeros(len(series))

            for start in range(0, len(series), max_length):

                end = min(start + max_length, len(series))

                chunk = torch.tensor(
                    series[start:end],
                    dtype=torch.float32
                )

                input_ids, attention_mask, _ = tokenizer.context_input_transform(
                    chunk
                )

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
                preds_full[start:end] = (
                    probs[:length] >= threshold
                )

            all_scores.append(scores_full)
            all_preds.append(preds_full)
            all_labels.append(labels)

    # --------------------------------------------------
    # FLATTEN
    # --------------------------------------------------

    all_scores = np.concatenate(all_scores)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)


    # --------------------------------------------------
    # METRICS WITH GIVEN THRESHOLD
    # --------------------------------------------------

    tp = np.sum(
        (all_preds == 1) & (all_labels == 1)
    )

    fp = np.sum(
        (all_preds == 1) & (all_labels == 0)
    )

    fn = np.sum(
        (all_preds == 0) & (all_labels == 1)
    )

    tn = np.sum(
        (all_preds == 0) & (all_labels == 0)
    )


    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = (
        2 * precision * recall /
        (precision + recall + 1e-8)
    )

    accuracy = (
        (tp + tn) /
        (tp + tn + fp + fn + 1e-8)
    )


    # --------------------------------------------------
    # FIND OPTIMAL THRESHOLD FOR MAX F1
    # --------------------------------------------------

    from sklearn.metrics import precision_recall_curve

    precision_curve, recall_curve, thresholds = (
        precision_recall_curve(
            all_labels,
            all_scores
        )
    )

    f1_curve = (
        2
        * precision_curve[:-1]
        * recall_curve[:-1]
        /
        (
            precision_curve[:-1]
            +
            recall_curve[:-1]
            +
            1e-8
        )
    )

    best_idx = np.argmax(f1_curve)

    optimal_threshold = thresholds[best_idx]
    optimal_f1 = f1_curve[best_idx]

    optimal_precision = precision_curve[best_idx]
    optimal_recall = recall_curve[best_idx]


    print("\n========== Evaluation ==========")
    print(f"Threshold {threshold:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")

    print("\n========== Optimal Threshold ==========")
    print(f"Threshold : {optimal_threshold:.4f}")
    print(f"F1        : {optimal_f1:.4f}")
    print(f"Precision : {optimal_precision:.4f}")
    print(f"Recall    : {optimal_recall:.4f}")

    print("\nConfusion Matrix (default threshold)")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TN: {tn}")


    return {
        # default threshold metrics
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,

        # optimized threshold metrics
        "optimal_threshold": optimal_threshold,
        "optimal_f1": optimal_f1,
        "optimal_precision": optimal_precision,
        "optimal_recall": optimal_recall,

        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,

        "scores": all_scores,
        "labels": all_labels,

        "pr_curve": (
            precision_curve,
            recall_curve,
            thresholds,
        ),
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

    series_list, label_list = load_dataset(
        dataset_dir,
        dataset_name
    )

    all_scores = []
    all_labels = []

    with torch.no_grad():

        for series, labels in zip(series_list, label_list):

            series = np.asarray(series)
            labels = np.asarray(labels)

            if len(series) != len(labels):
                raise ValueError(
                    f"Mismatch: series={len(series)} labels={len(labels)}"
                )

            window_scores = np.zeros(len(series))
            window_counts = np.zeros(len(series))


            # --------------------------
            # window inference
            # --------------------------

            for start, window in sliding_window(
                series,
                window_size,
                stride
            ):

                end = min(start + window_size, len(series))

                context = torch.tensor(
                    window,
                    dtype=torch.float32
                )

                input_ids, attention_mask, _ = (
                    tokenizer.context_input_transform(
                        context
                    )
                )

                input_ids = input_ids.unsqueeze(0).to(device)
                attention_mask = attention_mask.unsqueeze(0).to(device)


                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )


                logits = outputs["logits"].squeeze(0)

                probs = torch.sigmoid(logits).cpu().numpy()


                length = end - start

                window_scores[start:end] += probs[:length]
                window_counts[start:end] += 1


            # average overlapping windows
            window_counts[window_counts == 0] = 1

            final_scores = (
                window_scores /
                window_counts
            )


            all_scores.append(final_scores)
            all_labels.append(labels)


    # --------------------------
    # flatten
    # --------------------------

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)


    # --------------------------
    # metrics with given threshold
    # --------------------------

    all_preds = (
        all_scores >= threshold
    )


    tp = np.sum(
        (all_preds == 1) &
        (all_labels == 1)
    )

    fp = np.sum(
        (all_preds == 1) &
        (all_labels == 0)
    )

    fn = np.sum(
        (all_preds == 0) &
        (all_labels == 1)
    )

    tn = np.sum(
        (all_preds == 0) &
        (all_labels == 0)
    )


    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = (
        2 * precision * recall /
        (precision + recall + 1e-8)
    )

    accuracy = (
        (tp + tn) /
        (tp + tn + fp + fn + 1e-8)
    )


    # --------------------------
    # optimal threshold search
    # --------------------------

    from sklearn.metrics import precision_recall_curve

    precision_curve, recall_curve, thresholds = (
        precision_recall_curve(
            all_labels,
            all_scores
        )
    )


    f1_curve = (
        2
        * precision_curve[:-1]
        * recall_curve[:-1]
        /
        (
            precision_curve[:-1]
            +
            recall_curve[:-1]
            +
            1e-8
        )
    )


    best_idx = np.argmax(f1_curve)

    optimal_threshold = thresholds[best_idx]
    optimal_f1 = f1_curve[best_idx]

    optimal_precision = precision_curve[best_idx]
    optimal_recall = recall_curve[best_idx]


    print("\n========== Window Evaluation ==========")
    print(f"Threshold: {threshold:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")

    print("\n========== Optimal Threshold ==========")
    print(f"Threshold: {optimal_threshold:.4f}")
    print(f"F1       : {optimal_f1:.4f}")
    print(f"Precision: {optimal_precision:.4f}")
    print(f"Recall   : {optimal_recall:.4f}")


    return {
        # default threshold
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,

        # optimized threshold
        "optimal_threshold": optimal_threshold,
        "optimal_f1": optimal_f1,
        "optimal_precision": optimal_precision,
        "optimal_recall": optimal_recall,

        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,

        "scores": all_scores,
        "labels": all_labels,

        "pr_curve": (
            precision_curve,
            recall_curve,
            thresholds,
        ),
    }
    

if __name__ == "__main__":

    model_path = (
        "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/FineTunedModels/Anomaly/run-12/checkpoint-final"
    )
    dataset_dir = Path(
        "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/SMAP"
    )

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

    rows = []


    print(f"\n{'='*20}")
    print(f"Evaluating {"SMAP"}")
    print(f"{'='*20}")

    metrics = evaluate_dataset_no_windowing(
        model=model,
        tokenizer=tokenizer,
        dataset_dir=dataset_dir,
        dataset_name="SMAP",
    )

    metrics_windowing = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset_dir=dataset_dir,
        dataset_name="SMAP",
    )


    # ==================================================
    # NO WINDOWING RESULTS
    # ==================================================

    print("\n========== No Windowing ==========")

    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")

    print(f"Optimal threshold: {metrics['optimal_threshold']:.4f}")
    print(f"Optimal F1       : {metrics['optimal_f1']:.4f}")

    print(
        f"TP: {metrics['tp']} | "
        f"FP: {metrics['fp']} | "
        f"FN: {metrics['fn']} | "
        f"TN: {metrics['tn']}"
    )

    print(
        f"PR curve points: "
        f"{len(metrics['pr_curve'][0])}"
    )


    # ==================================================
    # WINDOWING RESULTS
    # ==================================================

    print("\n========== Windowing ==========")

    print(f"Precision : {metrics_windowing['precision']:.4f}")
    print(f"Recall    : {metrics_windowing['recall']:.4f}")
    print(f"F1        : {metrics_windowing['f1']:.4f}")
    print(f"Accuracy  : {metrics_windowing['accuracy']:.4f}")

    print(
        f"Optimal threshold: "
        f"{metrics_windowing['optimal_threshold']:.4f}"
    )

    print(
        f"Optimal F1       : "
        f"{metrics_windowing['optimal_f1']:.4f}"
    )

    print(
        f"TP: {metrics_windowing['tp']} | "
        f"FP: {metrics_windowing['fp']} | "
        f"FN: {metrics_windowing['fn']} | "
        f"TN: {metrics_windowing['tn']}"
    )

    print(
        f"PR curve points: "
        f"{len(metrics_windowing['pr_curve'][0])}"
    )


    dataset_name = "SMAP"

    long_results = pd.DataFrame(
        {
            "Metric": [
                "precision",
                "recall",
                "f1",
                "accuracy",
                "optimal_threshold",
                "optimal_f1",
                "tp",
                "fp",
                "fn",
                "tn",
            ],

            "Window": [
                metrics_windowing["precision"],
                metrics_windowing["recall"],
                metrics_windowing["f1"],
                metrics_windowing["accuracy"],
                metrics_windowing["optimal_threshold"],
                metrics_windowing["optimal_f1"],
                metrics_windowing["tp"],
                metrics_windowing["fp"],
                metrics_windowing["fn"],
                metrics_windowing["tn"],
            ],

            "No Window": [
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["accuracy"],
                metrics["optimal_threshold"],
                metrics["optimal_f1"],
                metrics["tp"],
                metrics["fp"],
                metrics["fn"],
                metrics["tn"],
            ],
        }
    )


    print("\nLong Results")
    print(long_results)


    output_file = (
        "/data/horse/ws/juha972b-AION-BERT-Chronos/"
        "BERTi/Results/Finetuning/Anomaly/"
        f"{dataset_name}.csv"
    )


    long_results.to_csv(
        output_file,
        index=False
    )

    print(f"Saved results to {output_file}")

