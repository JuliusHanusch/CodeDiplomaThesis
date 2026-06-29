from pathlib import Path
import numpy as np
import torch
import pandas as pd
import sys
from sklearn.linear_model import Ridge

# -------------------------
# PATH SETUP
# -------------------------
ROOT = Path("/content/CodeDiplomaThesis")
sys.path.append(str(ROOT))

from chronos_pkg.src.chronos import ChronosPipeline
from gluonts.dataset.arrow import ArrowFile


# -------------------------
# METRICS
# -------------------------
def rmse(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return np.sqrt(np.mean((preds - labels) ** 2))


def mae(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return np.mean(np.abs(preds - labels))

def extract_features(series: np.ndarray):
    series = np.asarray(series)

    series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)

    mean = np.mean(series)
    std = np.std(series)
    min_v = np.min(series)
    max_v = np.max(series)
    last = series[-1]

    x = np.arange(len(series))

    if len(series) < 2 or np.all(series == series[0]):
        slope = 0.0
    else:
        try:
            slope = np.polyfit(x, series, 1)[0]
        except Exception:
            slope = 0.0

    feat = np.array([mean, std, min_v, max_v, last, slope], dtype=np.float32)

    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    return feat

def train_ridge_baseline(train_dataset, context_length=512):

    X_feat = []
    y_target = []



    X, y = load_arrow(train_dataset)

    for i in range(len(X)):

        series = X[i][-context_length:]
        label = float(y[i])

        X_feat.append(extract_features(series))
        y_target.append(label)

    X_feat = np.vstack(X_feat)
    y_target = np.asarray(y_target, dtype=np.float32)

    model = Ridge(alpha=1.0)
    model.fit(X_feat, y_target)

    return model
# -------------------------
# GLUONTS ARROW LOADER (CORRECT)
# -------------------------
def load_arrow(path: Path):
    """
    Reads TS-ARROW dataset using GluonTS ArrowFile reader.
    """

    dataset = ArrowFile(path)

    series = []
    labels = []

    for entry in dataset:
        target = np.asarray(entry["target"], dtype=np.float32)

        # handle possible label keys safely
        if "label" in entry:
            label = entry["label"]
        elif "y" in entry:
            label = entry["y"]
        else:
            raise KeyError("No label found in dataset entry")

        series.append(target)
        labels.append(float(label))

    return np.stack(series), np.array(labels, dtype=np.float32)


# -------------------------
# BASELINES
# -------------------------
def baseline_predict_mean(series):
    return np.mean(series)


def baseline_predict_last(series):
    return series[-1]


# -------------------------
# CHRONOS EVALUATION
# -------------------------
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


def evaluate_all_baselines(ridge_model, test_arrow_path, context_length=512):

    X, y = load_arrow(test_arrow_path)

    preds_mean = []
    preds_last = []
    preds_ridge = []

    for i in range(len(X)):

        series = X[i][-context_length:]
        label = float(y[i])

        # naive baselines
        preds_mean.append(np.mean(series))
        preds_last.append(series[-1])

        # ridge baseline
        feat = extract_features(series).reshape(1, -1)
        preds_ridge.append(ridge_model.predict(feat)[0])

    y = np.asarray(y, dtype=np.float32)

    return {
        "mean": {
            "rmse": rmse(preds_mean, y),
            "mae": mae(preds_mean, y),
        },
        "last": {
            "rmse": rmse(preds_last, y),
            "mae": mae(preds_last, y),
        },
        "ridge": {
            "rmse": rmse(preds_ridge, y),
            "mae": mae(preds_ridge, y),
        },
    }


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    model_path = "/content/CodeDiplomaThesis/FineTunedModels/TSER/run-6/checkpoint-final"
    test_dataset = "/content/CodeDiplomaThesis/data/finetuning/TSER/LiveFuelMoisture/test.arrow"
    train_dataset = "/content/CodeDiplomaThesis/data/finetuning/TSER/LiveFuelMoisture/train.arrow"



    pipeline = ChronosPipeline.from_pretrained(
        model_path,
        task="tser",
    )

    model = pipeline.model

    # load regression head
    regressor_path = Path(model_path) / "tser.pt"
    model.regressor.load_state_dict(
        torch.load(regressor_path, map_location="cpu")
    )

    tokenizer = pipeline.tokenizer

    rows = []


    # ---------------------------------------------------
    # iterate over datasets
    # ---------------------------------------------------

    print("Evaluating:", test_dataset)

    # -------------------------
    # Chronos model
    # -------------------------
    chronos_metrics = evaluate_chronos(
        model=model,
        tokenizer=tokenizer,
        test_arrow_path=test_dataset,
    )

    ridge_model = train_ridge_baseline(train_dataset=train_dataset)

    baseline_metrics = evaluate_all_baselines(
        ridge_model,
        test_dataset
    )

    print("\nBaselines:")
    print(f"Mean  RMSE: {baseline_metrics['mean']['rmse']:.6f}")
    print(f"Mean  MAE : {baseline_metrics['mean']['mae']:.6f}")

    print(f"Last  RMSE: {baseline_metrics['last']['rmse']:.6f}")
    print(f"Last  MAE : {baseline_metrics['last']['mae']:.6f}")

    print(f"Ridge RMSE: {baseline_metrics['ridge']['rmse']:.6f}")
    print(f"Ridge MAE : {baseline_metrics['ridge']['mae']:.6f}")


    rows.append({
        "dataset": test_dataset,

        "chronos_rmse": chronos_metrics["rmse"],
        "chronos_mae": chronos_metrics["mae"],

        "mean_rmse": baseline_metrics["mean"]["rmse"],
        "mean_mae": baseline_metrics["mean"]["mae"],

        "last_rmse": baseline_metrics["last"]["rmse"],
        "last_mae": baseline_metrics["last"]["mae"],

        "ridge_rmse": baseline_metrics["ridge"]["rmse"],
        "ridge_mae": baseline_metrics["ridge"]["mae"],

    })

results = pd.DataFrame(rows)

print("\nFinal Results")
print(results)
ds_name = Path(test_dataset).parent.name

out_file = ROOT / f"{ds_name}.csv"
results.to_csv(out_file, index=False)

print(f"\nSaved to {out_file}")