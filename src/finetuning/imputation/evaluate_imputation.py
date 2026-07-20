print("Start Imports")
import logging
from typing import Optional
import sqlite3
import argparse
import yaml
import json
import torch
import typer
from gluonts.itertools import batcher
from tqdm.auto import tqdm
from pathlib import Path
from transformers import AutoModelForMaskedLM, AutoConfig
print("Imports 1")

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)  # optional, for wide display
pd.set_option("display.max_colwidth", None)
print("Start Imports 2")

# Include Parent Directory to load packages from
import sys  
root_dir = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi")
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"src").resolve()))  
sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))
print("Imports 3")

from chronos_pkg.src.chronos import ChronosPipeline
from chronos_pkg.src.chronos import ChronosConfig
from chronos_pkg.src.chronos.chronos_bolt import ChronosBoltModelForForecasting, ChronosBoltConfig
from src.utils import load_val_data
print("Finished Imports")


app = typer.Typer(pretty_exceptions_enable=False)


def load_chronos_bert(model_path: str, device: str, torch_dtype: torch.dtype):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    if hasattr(config, "chronos_config"):
        chronos_cfg = ChronosConfig(**config.chronos_config)
    else:
        raise ValueError("No chronos_config found in model config.")
    tokenizer = chronos_cfg.create_tokenizer()
    context_length = getattr(chronos_cfg, "context_length", 512)  # fallback default
    return model, tokenizer, context_length


def load_chronos_bert_bolt(model_path: str, device: str, torch_dtype: torch.dtype, bolt: bool):
    config = AutoConfig.from_pretrained(model_path)
    if bolt == False:
        model = AutoModelForMaskedLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch_dtype,
                device_map=device,
            )
    elif bolt == True:
        model = ChronosBoltModelForForecasting(config)
    tokenizer = config.create_tokenizer()
    context_length = getattr(config, "context_length", 512)  # fallback default
    return model, tokenizer, context_length




def timeseries_level_scaled_metrics(labels_array, preds_array, mask_array):
    # Probabilistic predictions -> median
    if preds_array.ndim == 3:
        preds_median = np.median(preds_array, axis=1)
    else:
        preds_median = preds_array

    valid_mask = mask_array & ~np.isnan(labels_array)

    error_model = np.abs(preds_median - labels_array)

    mae_scaled_series = []
    mae_series = []

    baseline_scaled_series = []
    baseline_series = []

    debug_series = []   # <-- add this

    for i in tqdm(range(labels_array.shape[0]), desc="Series"):

        token_mask = valid_mask[i]

        if not np.any(token_mask):
            continue

        series = labels_array[i]

        series_scale = np.nanmean(np.abs(series[token_mask]))
        series_scale = max(series_scale, 1.0)

        mae_model = np.mean(error_model[i, token_mask])

        mae_series.append(mae_model)
        mae_scaled_series.append(mae_model / series_scale)


        # -----------------------
        # Baseline prediction
        # -----------------------
        baseline_pred = np.full_like(series, np.nan)

        mask = token_mask.copy()

        idx = 0
        while idx < len(series):

            if not mask[idx]:
                idx += 1
                continue

            start = idx

            while idx < len(series) and mask[idx]:
                idx += 1

            end = idx

            left_value = None
            right_value = None

            if start > 0:
                j = start - 1
                while j >= 0:
                    if not mask[j] and not np.isnan(series[j]):
                        left_value = series[j]
                        break
                    j -= 1

            if end < len(series):
                j = end
                while j < len(series):
                    if not mask[j] and not np.isnan(series[j]):
                        right_value = series[j]
                        break
                    j += 1

            if left_value is not None and right_value is not None:
                fill = 0.5 * (left_value + right_value)
            elif left_value is not None:
                fill = left_value
            elif right_value is not None:
                fill = right_value
            else:
                fill = series_scale

            baseline_pred[start:end] = fill


        baseline_error = np.abs(
            baseline_pred[token_mask] - series[token_mask]
        )

        mae_baseline = np.mean(baseline_error)

        baseline_series.append(mae_baseline)
        baseline_scaled_series.append(mae_baseline / series_scale)


        if len(debug_series) < 10:
            debug_series.append(
                {
                    "series_id": i,
                    "ground_truth": series.copy(),
                    "mask": mask.copy(),
                    "chronos_prediction": np.where(
                        mask,
                        preds_median[i],
                        np.nan
                    ),
                    "baseline_prediction": baseline_pred.copy(),
                    "mae_chronos": mae_model,
                    "mae_baseline": mae_baseline,
                }
            )


    rows = []

    for item in debug_series:
        for t in range(len(item["ground_truth"])):
            rows.append(
                {
                    "series_id": item["series_id"],
                    "time": t,
                    "ground_truth": item["ground_truth"][t],
                    "masked": bool(item["mask"][t]),
                    "chronos_prediction": item["chronos_prediction"][t],
                    "baseline_prediction": item["baseline_prediction"][t],
                    "mae_chronos_series": item["mae_chronos"],
                    "mae_baseline_series": item["mae_baseline"],
                }
            )

    df = pd.DataFrame(rows)

    df.to_csv(
        "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/Results/Finetuning/Imputation/imputation_debug_first10_series.csv",
        index=False
    )


    MASE_model = (
        float(np.mean(mae_scaled_series))
        if mae_scaled_series else np.nan
    )

    MAE_model = (
        float(np.mean(mae_series))
        if mae_series else np.nan
    )

    MASE_baseline = (
        float(np.mean(baseline_scaled_series))
        if baseline_scaled_series else np.nan
    )

    MAE_baseline = (
        float(np.mean(baseline_series))
        if baseline_series else np.nan
    )

    return MASE_model, MAE_model, MASE_baseline, MAE_baseline


def impute_span(
    series: np.ndarray,
    model,
    tokenizer,
    context_length: int,
    mask_ratio: float = 0.15,
    mean_span_length: float = 3.0,
    num_samples: int = 10,
    temperature: float = 1.0,
):
    """
    Impute missing values by masking spans of tokens (MLM-style span masking)
    and sampling from model predictions.

    Returns:
        samples: np.ndarray, shape (num_samples, context_length)
        mask_positions: np.ndarray, shape (context_length,), bool
    """
    device = model.device
    model.eval()

    # --- convert series to tensor ---
    series_tensor = torch.tensor(series, dtype=torch.float32, device="cpu")
    seq_len = min(len(series_tensor), context_length)

    # --- tokenize (like in training) ---
    input_ids, attention_mask, scale = tokenizer.context_input_transform(
        series_tensor[:seq_len].unsqueeze(0)
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    scale = scale.to(device)

    # --- determine mask token id ---
    mask_token_id = getattr(tokenizer.config, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError(
            "Tokenizer does not define `mask_token_id`. "
            "Ensure you use the same tokenizer used for MLM training."
        )

    special_token_cutoff = tokenizer.config.n_special_tokens
    #print("special_token_cutoff", special_token_cutoff)

    # --- identify valid (non-special) positions ---
    special_tokens_mask = input_ids < special_token_cutoff
    valid_positions = ~special_tokens_mask
    n_tokens = valid_positions.sum().detach().cpu().item()
    n_to_mask = int(mask_ratio * n_tokens)


    # --- create boolean mask for spans ---
    mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)

    valid_indices = valid_positions.nonzero(as_tuple=True)[1]
    total_masked = 0

    while total_masked < n_to_mask and len(valid_indices) > 0:
        # Sample a span length from Poisson distribution
        span_len = max(1, np.random.poisson(mean_span_length))
        if total_masked + span_len > n_to_mask:
            span_len = n_to_mask - total_masked

        # Randomly choose a valid start index
        start_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))].item()
        end_idx = min(start_idx + span_len, input_ids.size(1))

        # Skip if overlapping existing masked region
        if mask[0, start_idx:end_idx].any():
            continue

        # Apply mask
        mask[0, start_idx:end_idx] = True
        total_masked += span_len
    
    #print("n_to_mask", n_to_mask, "total_masked", total_masked)


    # Ensure no special tokens are masked
    mask &= valid_positions

    mask_positions = mask[0]  # shape: (context_length,)

    # --- create masked input ---
    masked_input = input_ids.clone()
    masked_input[mask] = mask_token_id

    # --- forward pass ---
    with torch.no_grad():
        outputs = model(input_ids=masked_input, attention_mask=attention_mask)
        logits = outputs.logits[0]  # (context_length, vocab_size)
        probs = torch.nn.functional.softmax(logits / float(temperature), dim=-1)

    # --- sample new tokens for masked positions ---
    sampled_token_ids = masked_input[0].repeat(num_samples, 1)

    masked_idx_list = mask_positions.nonzero(as_tuple=True)[0].tolist()
    for pos in masked_idx_list:
        p = probs[pos].clone()
        if torch.isnan(p).any() or p.sum() == 0:
            p = torch.ones_like(p) / p.numel()
        else:
            p /= p.sum()
        sampled_token_ids[:, pos] = torch.multinomial(p, num_samples, replacement=True)

    # --- decode back to real values ---
    sampled_ids_for_output = sampled_token_ids.unsqueeze(0)
    values = tokenizer.output_transform(sampled_ids_for_output.cpu(), scale.cpu())
    values = values.squeeze(0).cpu().numpy()  # (num_samples, context_length)

    # --- ensure unmasked tokens remain identical ---
    mask_np = mask_positions.cpu().numpy()
    unmasked_ids = input_ids[0, ~mask_positions].cpu()
    unmasked_values = tokenizer.output_transform(
        unmasked_ids.unsqueeze(0), scale.cpu()
    ).squeeze(0).cpu().numpy()
    for i in range(num_samples):
        values[i, ~mask_np] = unmasked_values

    # --- pad if series shorter than context_length ---
    if len(series) < context_length:
        values[:, len(series):] = np.nan
        mask_np[len(series):] = False

    return values.astype(np.float32), mask_np

import numpy as np




def main(
    idx: int,
    device: str = "cuda",
    torch_dtype: str = "float32",
    batch_size: int = 32,
    num_samples: int = 20,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    bolt: bool = False,
):


    DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/imputation/imputation.db"
    CONFIG_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/eval_configs/in-domain.yaml"

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT dataset, model_path, mean_span_length, masking_ratio
        FROM runs
        WHERE id = ?
    """, (idx,))

    row = cur.fetchone()

    if row is None:
        raise ValueError(f"No run found for id={idx}")

    dataset_name, model_path, mean_span_length, masking_ratio = row
    
    model, tokenizer, context_length = load_chronos_bert(model_path, device, torch_dtype)


    with open(CONFIG_PATH) as fp:
        backtest_configs = yaml.safe_load(fp)

    backtest_configs = [
        cfg for cfg in backtest_configs
        if cfg["name"] == dataset_name
    ]

    if len(backtest_configs) == 0:
        raise ValueError(f"Dataset '{dataset_name}' not found in config.")

    metrics = []

    metrics = []
    for config in backtest_configs:
        aok = True
        print("Config", config)
        targets = config.pop("targets", ["target"])
        for target in targets:
            print("target", target)
            dataset_name = config["name"]

            logger.info(f"Loading {target} from {dataset_name}")
            test_data, train_data = load_val_data(config=config, target=target)
            if test_data is None: # Catch DS Not Found
                logger.info(f"Failed locating Target {target} in {dataset_name}")
                continue 


            all_labels, all_imputed_values, all_masks_imputation, all_linear_values   = [], [], [], []

            for batch in tqdm(batcher(test_data, batch_size=batch_size), desc=dataset_name):
                for input_instance, label_instance in batch:
                    series = np.array(input_instance["target"], dtype=np.float32)
                    if len(series) < context_length:
                        pad_len = context_length - len(series)
                            # Left-pad with NaNs
                        series = np.pad(series, (pad_len, 0), constant_values=np.nan)
                    elif len(series) > context_length:
                        series = series[-context_length:]


                    imputed_values, mask_positions_imputation = impute_span(
                        series=series,
                        model=model,
                        tokenizer=tokenizer,
                        context_length=context_length,
                        mean_span_length= mean_span_length,
                        mask_ratio=masking_ratio,
                        num_samples=num_samples,
                        temperature=1.0
                    )

                    masked_series = series.copy()

                    # Hide the exact same positions Chronos saw as missing
                    masked_series[mask_positions_imputation] = np.nan

                    observed = ~np.isnan(masked_series)

                    if observed.sum() >= 2:
                        x = np.arange(len(series))

                        linear_prediction = np.interp(
                            x,
                            x[observed],
                            masked_series[observed],
                        ).astype(np.float32)

                    else:
                        # fallback if interpolation is impossible
                        linear_prediction = masked_series.copy()


                    all_labels.append(series)
                    all_imputed_values.append(imputed_values)
                    all_masks_imputation.append(mask_positions_imputation)
                    all_linear_values.append(linear_prediction)



            logger.info("Imputation and Forecast Done")       
                    
            #labels arrays
            labels_array = np.stack(all_labels, axis=0)

            #Imputation Arrays
            preds_array_imputation = np.stack(all_imputed_values, axis=0)
            mask_array_imputation = np.stack(all_masks_imputation, axis=0)
            preds_array_linear = np.stack(all_linear_values, axis=0)

            logger.info("Regular Metrics Done")


            MASE_Chronos, MAE_Chronos, MASE_baseline, MAE_baseline = timeseries_level_scaled_metrics(
                labels_array=labels_array,
                preds_array=preds_array_imputation,
                mask_array=mask_array_imputation,
                )
            
            
            print("DEBUG:")
            print("MASE_Chronos:", MASE_Chronos, type(MASE_Chronos))
            print("MAE_Chronos:", MAE_Chronos, type(MAE_Chronos))
            print("MASE_baseline:", MASE_baseline, type(MASE_baseline))
            print("MAE_baseline:", MAE_baseline, type(MAE_baseline))
            
            conn = sqlite3.connect(DB_PATH)

            conn.execute("""
                UPDATE runs
                SET
                    MAE=?,
                    MASE=?,
                    MAE_Lin=?,
                    MASE_Lin=?
                WHERE id=?
            """, (
                float(MAE_Chronos),
                float(MASE_Chronos),
                float(MAE_baseline),
                float(MASE_baseline),
                idx,
            ))

            conn.commit()

    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("Chronos Evaluation")
    logger.setLevel(logging.INFO)

    main(idx=args.index)

