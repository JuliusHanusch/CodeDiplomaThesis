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
    roc_auc_score
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

def predict_series(
    model,
    tokenizer,
    series,
    gt_series,
    context_length=512,
    stride=128,
):
    device = next(model.parameters()).device
    model.eval()

    scores_sum = np.zeros(len(series), dtype=np.float32)
    counts = np.zeros(len(series), dtype=np.float32)
    gt_all = np.zeros(len(series), dtype=np.float32)

    with torch.no_grad():

        for start in range(0, len(series) - context_length + 1, stride):

            window = series[start:start + context_length]
            gt_window = gt_series[start:start + context_length]

            context = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

            input_ids, attention_mask, _ = tokenizer.context_input_transform(context)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs["logits"][0]  # (T,)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

            mask = attention_mask[0].bool().cpu().numpy()

            probs = probs[mask]

            scores_sum[start:start + len(probs)] += probs
            counts[start:start + len(probs)] += 1

            gt_all[start:start + len(probs)] += np.array(gt_window)[mask]

    scores = scores_sum / np.maximum(counts, 1e-8)
    gt = (gt_all > 0).astype(np.int32)

    return scores, gt

def evaluate_dataset(
    model,
    tokenizer,
    dataset_dir,
    dataset_name,
):
    series_list, label_list = load_dataset(dataset_dir, dataset_name)

    all_scores = []
    all_gt = []

    print(f"\n[DEBUG] Dataset: {dataset_name}")
    print(f"[DEBUG] Num series: {len(series_list)}")

    for series, gt in zip(series_list, label_list):

        scores, labels = predict_series(
            model,
            tokenizer,
            series,
            gt,
        )

        min_len = min(len(scores), len(labels))

        all_scores.append(scores[:min_len])
        all_gt.append(labels[:min_len])

    all_scores = np.concatenate(all_scores)
    all_gt = np.concatenate(all_gt)

    # -------------------------
    # metrics
    # -------------------------
    roc_auc = roc_auc_score(all_gt, all_scores)

    preds = (all_scores > np.percentile(all_scores, 90)).astype(int)

    f1 = precision_recall_fscore_support(
        all_gt,
        preds,
        average="binary",
        zero_division=0,
    )[2]

    print("\n=== TOKEN LEVEL RESULTS ===")
    print("ROC-AUC:", roc_auc)
    print("F1:", f1)
    print("GT ratio:", all_gt.mean())
    print("Pred ratio:", preds.mean())

    print("\nScore stats:")
    print("pos mean:", all_scores[all_gt == 1].mean())
    print("neg mean:", all_scores[all_gt == 0].mean())

    return {
        "F1": f1,
        "ROC_AUC": roc_auc,
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

        # print(
        #     f"F1-PA : {metrics['F1PA']:.4f}"
        # )

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