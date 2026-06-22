from pathlib import Path
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset
import sys
from datasets import get_dataset_config_names

#colab import
ROOT = "/content/CodeDiplomaThesis"
sys.path.append(str(Path(ROOT).resolve()))

from chronos_pkg.src.chronos import ChronosPipeline


# -------------------------
# RMSE
# -------------------------
def rmse(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return np.sqrt(np.mean((preds - labels) ** 2))


def mae(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return np.mean(np.abs(preds - labels))


# -------------------------
# Evaluation
# -------------------------
def evaluate_tser_dataset(
    model,
    tokenizer,
    dataset_name,
    repo="foxy-steve/monash_uea_ucr_tser",
    context_length: int = 512,
):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"


    model.eval()
    model.to(device)

    ds = load_dataset(repo, dataset_name)["test"]

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for ex in ds:

            series = np.asarray(ex["timeseries"], dtype=np.float32)
            label = float(ex["to_predict"])

            print("label", label)
            print("len series", len(series))

            context = torch.tensor(series[-context_length:], dtype=torch.float32)

            input_ids, attention_mask, _ = tokenizer.context_input_transform(context)

            print("input_ids shape:", input_ids.shape)  # debug

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            preds = outputs["logits"].detach().cpu().numpy().reshape(-1)

            all_preds.extend(preds)
            all_labels.extend([label] * len(preds))  # safer

    # -------------------------
    # FINAL METRICS
    # -------------------------
    all_preds = np.array(all_preds, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.float32)

    rmse_score = rmse(all_preds, all_labels)
    mae_score = mae(all_preds, all_labels)
    return {
        "rmse": rmse_score,
        "mae": mae_score,
        "n_samples": len(all_preds),
        "preds": all_preds,
        "labels": all_labels,
    }


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    model_path = "/content/CodeDiplomaThesis/FineTunedModels/TSER/run-10/checkpoint-final"

    print("Loading model:", model_path)

    pipeline = ChronosPipeline.from_pretrained(
        model_path,
        task="tser",
    )

    model = pipeline.model

    regressor = Path(model_path) / "tser.pt"

    model.regressor.load_state_dict(
        torch.load(regressor, map_location="cpu")
    )

    tokenizer = pipeline.tokenizer

    repo = "foxy-steve/monash_uea_ucr_tser"

    datasets = get_dataset_config_names(repo)

    print(f"Found {len(datasets)} datasets:")
    print(datasets)

    rows = []

    for ds_name in datasets:

        print("\n" + "=" * 30)
        print(f"Evaluating {ds_name}")
        print("=" * 30)

        metrics = evaluate_tser_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_name=ds_name,
        )

        print(f"RMSE     : {metrics['rmse']:.6f}")
        print(f"MAE      : {metrics['mae']:.6f}")
        print(f"Samples  : {metrics['n_samples']}")

        rows.append({
            "dataset": ds_name,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "n_samples": metrics["n_samples"],
        })

    results = pd.DataFrame(rows)

    print("\nFinal Results")
    print(results)

    output_file = "/Results/Finetuning/TSER/tser_eval_results.csv"

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file, index=False)

    print(f"\nSaved to {output_file}")