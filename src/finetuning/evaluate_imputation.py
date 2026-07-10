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

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)  # optional, for wide display
pd.set_option("display.max_colwidth", None)

# Include Parent Directory to load packages from
import sys  
root_dir = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi")
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"src").resolve()))  
sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos import ChronosPipeline
from chronos_pkg.src.chronos import ChronosConfig
from chronos_pkg.src.chronos.chronos_bolt import ChronosBoltModelForForecasting, ChronosBoltConfig
from src.utils import load_val_data

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

    error_model = preds_median - labels_array
    abs_error_model = np.abs(error_model)

    mae_scaled_series = []
    rmse_scaled_series = []

    for i in tqdm(range(labels_array.shape[0]), desc="Series"):
        token_mask = valid_mask[i]

        if not np.any(token_mask):
            continue

        # Series scale based on ground truth values
        series_scale = np.nanmean(np.abs(labels_array[i, token_mask]))

        #Prevent division by zero
        series_scale = max(series_scale, 1.0)

        mae_model = np.mean(abs_error_model[i, token_mask])
        #rmse_model = np.sqrt(np.mean(error_model[i, token_mask] ** 2))

        mae_scaled_series.append(mae_model / series_scale)
        #rmse_scaled_series.append(rmse_model / series_scale)

    mae_scaled_mean = (
        float(np.mean(mae_scaled_series))
        if mae_scaled_series
        else np.nan
    )

    # rmse_scaled_mean = (
    #     float(np.mean(rmse_scaled_series))
    #     if rmse_scaled_series
    #     else np.nan
    # )

    return mae_scaled_mean, mae_model

# def timeseries_level_scaled_metrics(labels_array, preds_array, mask_array):
#     n_series, seq_len = labels_array.shape

#     # Probabilistic predictions -> median
#     if preds_array.ndim == 3:
#         preds_median = np.median(preds_array, axis=1)
#     else:
#         preds_median = preds_array

#     # Naive baseline
#     naive_preds = np.full_like(labels_array, np.nan, dtype=np.float32)
#     fallback_flag = np.zeros_like(labels_array, dtype=bool)

#     for i in tqdm(range(n_series), desc="Series"):
#         for t in range(seq_len):

#             if not mask_array[i, t]:
#                 continue

#             prev_val = np.nan
#             next_val = np.nan

#             # search up to 10 tokens to the left
#             for k in range(1, 11):
#                 pos = t - k
#                 if pos < 0:
#                     break

#                 if (
#                     not np.isnan(labels_array[i, pos])
#                     and not mask_array[i, pos]   # cannot use masked targets
#                 ):
#                     prev_val = labels_array[i, pos]
#                     break

#             # search up to 10 tokens to the right
#             for k in range(1, 11):
#                 pos = t + k
#                 if pos >= seq_len:
#                     break

#                 if (
#                     not np.isnan(labels_array[i, pos])
#                     and not mask_array[i, pos]   # cannot use masked targets
#                 ):
#                     next_val = labels_array[i, pos]
#                     break


#             prev_valid = not np.isnan(prev_val)
#             next_valid = not np.isnan(next_val)

#             if prev_valid and next_valid:
#                 naive_preds[i, t] = 0.5 * (prev_val + next_val)
#             elif prev_valid:
#                 naive_preds[i, t] = prev_val
#             elif next_valid:
#                 naive_preds[i, t] = next_val
#             else:
#                 # No valid tokens within ±10 positions
#                 naive_preds[i, t] = np.nan
#                 fallback_flag[i, t] = True

#     valid_mask = mask_array & ~np.isnan(labels_array)

#     error_model = preds_median - labels_array
#     error_naive = naive_preds - labels_array
#     abs_error_model = np.abs(error_model)
#     abs_error_naive = np.abs(error_naive)

#     series_rows = []
#     mae_scaled_series = []
#     rmse_scaled_series = []

#     # Compute series-level metrics
#     for i in tqdm(range(n_series), desc="Series"):
#         token_mask = valid_mask[i]

#         if not np.any(token_mask):
#             continue

#         # Apply fallback for tokens with missing neighbors
#         fallback_tokens = fallback_flag[i] & token_mask
#         if np.any(fallback_tokens):
#             series_scale = np.nanmean(np.abs(labels_array[i, token_mask]))
#             abs_error_naive[i, fallback_tokens] = max(series_scale, 1)
#             error_naive[i, fallback_tokens] = abs_error_naive[i, fallback_tokens]

#         # MAE & RMSE
#         mae_model = np.mean(abs_error_model[i, token_mask])
#         mae_naive = np.mean(abs_error_naive[i, token_mask])
#         rmse_model = np.sqrt(np.mean(error_model[i, token_mask]**2))
#         rmse_naive = np.sqrt(np.mean(error_naive[i, token_mask]**2))

#         # Safe floor if naive metrics too small
#         mae_naive = max(mae_naive, 1)
#         rmse_naive = max(rmse_naive, 1)

#         mae_scaled = mae_model / mae_naive
#         rmse_scaled = rmse_model / rmse_naive

#         mae_scaled_series.append(mae_scaled)
#         rmse_scaled_series.append(rmse_scaled)

#         # series_row = {
#         #     "series_index": i,
#         #     "mae_model": mae_model,
#         #     "mae_naive": mae_naive,
#         #     "mae_scaled": mae_scaled,
#         #     "rmse_model": rmse_model,
#         #     "rmse_naive": rmse_naive,
#         #     "rmse_scaled": rmse_scaled,
#         #     "n_tokens_used": int(np.sum(token_mask)),
#         #     "fallback_used": np.any(fallback_tokens),

#         #     # NEW
#         #     "fallback_positions": np.where(fallback_tokens)[0].tolist() if np.any(fallback_tokens) else None,
#         #     "masked_positions": np.where(mask_array[i])[0].tolist(),
#         #     "naive_preds_masked": naive_preds[i, mask_array[i]].tolist(),
#         #     "abs_error_naive_masked": abs_error_naive[i, mask_array[i]].tolist(),
#         #     "labels_series": labels_array[i].tolist()
#         #     }        
#         #print(series_row)
#         #series_rows.append(series_row)

#     #df = pd.DataFrame(series_rows)
#     #df.to_csv("evaluation_results.csv", index=False)
    
#     mae_scaled_mean = float(np.mean(mae_scaled_series)) if mae_scaled_series else np.nan
#     rmse_scaled_mean = float(np.mean(rmse_scaled_series)) if rmse_scaled_series else np.nan


#     return mae_scaled_mean, rmse_scaled_mean


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

def impute_span_bolt():
    return 0



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


            all_labels, all_imputed_values, all_masks_imputation  = [], [], []

            for batch in tqdm(batcher(test_data, batch_size=batch_size), desc=dataset_name):
                for input_instance, label_instance in batch:
                    series = np.array(input_instance["target"], dtype=np.float32)
                    if len(series) < context_length:
                        pad_len = context_length - len(series)
                            # Left-pad with NaNs
                        series = np.pad(series, (pad_len, 0), constant_values=np.nan)
                    elif len(series) > context_length:
                        series = series[-context_length:]

                    if bolt: # TODO Fusion into one clean version
                        forecasts, mask_positions = impute_span_bolt()

                    else:
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
                        all_labels.append(series)
                        all_imputed_values.append(imputed_values)
                        all_masks_imputation.append(mask_positions_imputation)

            logger.info("Imputation and Forecast Done")       
                    
            #labels arrays
            labels_array = np.stack(all_labels, axis=0)

            #Imputation Arrays
            preds_array_imputation = np.stack(all_imputed_values, axis=0)
            mask_array_imputation = np.stack(all_masks_imputation, axis=0)

            logger.info("Regular Metrics Done")


            MASE_Chronos, MAE_Chronos = timeseries_level_scaled_metrics(
                labels_array=labels_array,
                preds_array=preds_array_imputation,
                mask_array=mask_array_imputation,
                )
            
            print("DEBUG:")
            print("MASE_Chronos:", MASE_Chronos, type(MASE_Chronos))
            print("MAE_Chronos:", MAE_Chronos, type(MAE_Chronos))
            
            conn = sqlite3.connect(DB_PATH)

            conn.execute("""
                UPDATE runs
                SET
                    MAE=?,
                    MASE=?
                WHERE id=?
            """, (
                float(MAE_Chronos),
                float(MASE_Chronos),
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

