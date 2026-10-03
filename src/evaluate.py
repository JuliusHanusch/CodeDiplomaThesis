import logging
from pathlib import Path
from typing import Iterable, Optional

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
import typer
import yaml
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss, RMSE, MAE, NRMSE, SumQuantileLoss, MSE, MAPE, SMAPE
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import SampleForecast
from tqdm.auto import tqdm
from functools import cache
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from utils import load_val_data
import json
from chronos_pkg.src.chronos import ChronosPipeline, ChronosConfig
from chronos_pkg.src.chronos.chronos_bolt import ChronosBoltPipeline
from chronos_pkg.src.chronos.chronos_bolt import ChronosBoltModelForForecasting, ChronosBoltConfig

import re
from math import log
from utils import get_model_size
from warnings import warn
import math

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


def timeseries_level_scaled_metrics(labels_array, preds_array, mask_array, csv_path: Path = None):
    n_series, seq_len = labels_array.shape

    # Probabilistic predictions -> median
    if preds_array.ndim == 3:
        preds_median = np.median(preds_array, axis=1)
    else:
        preds_median = preds_array

    # Naive baseline
    naive_preds = np.full_like(labels_array, np.nan, dtype=np.float32)
    fallback_flag = np.zeros_like(labels_array, dtype=bool)

    for i in range(n_series):
        for t in range(seq_len):

            if not mask_array[i, t]:
                continue

            prev_val = np.nan
            next_val = np.nan

            # search up to 10 tokens to the left
            for k in range(1, 11):
                pos = t - k
                if pos < 0:
                    break
                if not np.isnan(labels_array[i, pos]):
                    prev_val = labels_array[i, pos]
                    break

            # search up to 10 tokens to the right
            for k in range(1, 11):
                pos = t + k
                if pos >= seq_len:
                    break
                if not np.isnan(labels_array[i, pos]):
                    next_val = labels_array[i, pos]
                    break

            prev_valid = not np.isnan(prev_val)
            next_valid = not np.isnan(next_val)

            if prev_valid and next_valid:
                naive_preds[i, t] = 0.5 * (prev_val + next_val)
            elif prev_valid:
                naive_preds[i, t] = prev_val
            elif next_valid:
                naive_preds[i, t] = next_val
            else:
                # No valid tokens within ±10 positions
                naive_preds[i, t] = np.nan
                fallback_flag[i, t] = True

    valid_mask = mask_array & ~np.isnan(labels_array)

    error_model = preds_median - labels_array
    error_naive = naive_preds - labels_array
    abs_error_model = np.abs(error_model)
    abs_error_naive = np.abs(error_naive)

    series_rows = []
    mae_scaled_series = []
    rmse_scaled_series = []

    # Compute series-level metrics
    for i in range(n_series):
        token_mask = valid_mask[i]

        if not np.any(token_mask):
            continue

        # Apply fallback for tokens with missing neighbors
        fallback_tokens = fallback_flag[i] & token_mask
        if np.any(fallback_tokens):
            series_scale = np.nanmean(np.abs(labels_array[i, token_mask]))
            abs_error_naive[i, fallback_tokens] = max(0.01 * series_scale, 1)
            error_naive[i, fallback_tokens] = abs_error_naive[i, fallback_tokens]

        # MAE & RMSE
        mae_model = np.mean(abs_error_model[i, token_mask])
        mae_naive = np.mean(abs_error_naive[i, token_mask])
        rmse_model = np.sqrt(np.mean(error_model[i, token_mask]**2))
        rmse_naive = np.sqrt(np.mean(error_naive[i, token_mask]**2))

        # Safe floor if naive metrics too small
        mae_naive = max(mae_naive, 1)
        rmse_naive = max(rmse_naive, 1)

        mae_scaled = mae_model / mae_naive
        rmse_scaled = rmse_model / rmse_naive

        mae_scaled_series.append(mae_scaled)
        rmse_scaled_series.append(rmse_scaled)

        # if csv_path is not None:
        #     series_rows.append({
        #         "series_index": i,
        #         "mae_model": mae_model,
        #         "mae_naive": mae_naive,
        #         "mae_scaled": mae_scaled,
        #         "rmse_model": rmse_model,
        #         "rmse_naive": rmse_naive,
        #         "rmse_scaled": rmse_scaled,
        #         "n_tokens_used": int(np.sum(token_mask)),
        #         "fallback_used": np.any(fallback_tokens),

        #         # NEW
        #         "fallback_positions": np.where(fallback_tokens)[0].tolist() if np.any(fallback_tokens) else None,
        #         "masked_positions": np.where(mask_array[i])[0].tolist(),
        #         "labels_series": labels_array[i].tolist()
        #     })
    mae_scaled_mean = float(np.mean(mae_scaled_series)) if mae_scaled_series else np.nan
    rmse_scaled_mean = float(np.mean(rmse_scaled_series)) if rmse_scaled_series else np.nan

    # if csv_path is not None:
    #     df = pd.DataFrame(series_rows)
    #     csv_path.parent.mkdir(parents=True, exist_ok=True)
    #     df.to_csv(csv_path, index=False)

    return mae_scaled_mean, rmse_scaled_mean


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
    vocab_size = tokenizer.config.n_tokens

    # --- identify valid (non-special) positions ---
    special_tokens_mask = input_ids < special_token_cutoff
    valid_positions = ~special_tokens_mask
    n_tokens = valid_positions.sum().item()
    n_to_mask = max(1, int(mask_ratio * n_tokens))

    # --- create boolean mask for spans ---
    mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)

    valid_indices = valid_positions.nonzero(as_tuple=True)[1]
    total_masked = 0

    while total_masked < n_to_mask and len(valid_indices) > 0:
        # Sample a span length from Poisson distribution
        span_len = max(1, int(torch.poisson(torch.tensor(mean_span_length)).item()))
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

def generate_forecasts_bolt():
    return 0



def generate_forecasts(
    series: np.ndarray,
    model,
    tokenizer,
    context_length: int,
    prediction_length: int,
    num_samples: int = 10,
    temperature: float = 1.0,
):
    """
    Predict the next `prediction_length` positions using MLM-style masking.
    
    Args:
        series: np.ndarray, input series (1D)
        model: HuggingFace AutoModelForMaskedLM
        tokenizer: ChronosTokenizer
        context_length: maximum context length for model input
        prediction_length: number of positions to predict (mask)
        num_samples: number of stochastic samples
        temperature: sampling temperature

    Returns:
        values: np.ndarray of shape (num_samples, context_length)
        mask_positions: np.ndarray of shape (context_length,), bool
    """
    device = model.device
    model.eval()

    # --- convert series to tensor ---
    series_tensor = torch.tensor(series, dtype=torch.float32, device="cpu")

    # --- tokenize ---
    input_ids, attention_mask, scale = tokenizer.context_input_transform(
        series_tensor.unsqueeze(0)
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    scale = scale.to(device)

    # --- mask last `prediction_length` positions ---
    mask_token_id = tokenizer.config.mask_token_id
    mask_positions = torch.zeros(context_length, dtype=torch.bool, device=device)
    mask_positions[-prediction_length:] = True
    #print("mask_positions",mask_positions)

    masked_input = input_ids.clone()
    masked_input[0, mask_positions] = mask_token_id

    #print("masked_input", masked_input)

    # --- forward pass ---
    with torch.no_grad():
        outputs = model(input_ids=masked_input, attention_mask=attention_mask)
        logits = outputs.logits[0]  # (context_length, vocab_size)
        probs = torch.nn.functional.softmax(logits / float(temperature), dim=-1)

    # --- sample new tokens for masked positions ---
    sampled_token_ids = masked_input[0].repeat(num_samples, 1)

    masked_idx_list = mask_positions.nonzero(as_tuple=True)[0].tolist()

    #print("masked_idx_list", masked_idx_list)
    for pos in masked_idx_list:
        p = probs[pos].clone()
        if torch.isnan(p).any() or p.sum() == 0:
            p = torch.ones_like(p) / p.numel()
        else:
            p /= p.sum()
        for s in range(num_samples):
            sampled_token_ids[s, pos] = torch.multinomial(p, 1).item()

    # --- decode back to real values ---
    sampled_ids_for_output = sampled_token_ids.unsqueeze(0)  # (1, num_samples, context_length)
    values = tokenizer.output_transform(sampled_ids_for_output.cpu(), scale.cpu())
    values = values.squeeze(0).cpu().numpy()  # (num_samples, context_length)

    for i in range(num_samples):
        values[i, ~mask_positions.cpu().numpy()] = tokenizer.output_transform(
            input_ids[0:1, ~mask_positions].cpu(), scale.cpu()
        ).squeeze(0).cpu().numpy()

    return values.astype(np.float32), mask_positions.cpu().numpy()

@app.command()
def main(
    config_path: Path,
    chronos_model_id: str = "amazon/chronos-t5-small",
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    batch_size: int = 32,
    num_samples: int = 20,
    mask_ratio: float = 0.15,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    bolt: bool = False,
) -> pd.DataFrame:
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)

    # if bolt:
    #      PipelineClass = ChronosBoltPipeline
    # else:
    #     PipelineClass = ChronosPipeline

    # pipeline = PipelineClass.from_pretrained(
    #      chronos_model_id,
    #      device_map=device,
    #      torch_dtype=torch_dtype,
    #  )
    
    model, tokenizer, context_length = load_chronos_bert(chronos_model_id, device, torch_dtype)


    # Load backtest configs
    with open(config_path) as fp:
        backtest_configs = yaml.safe_load(fp)

    metrics = []
    for config in backtest_configs:
        aok = True
        print("Config", config)
        targets = config.pop("targets", ["target"])
        for target in targets:
            print("target", target)
            dataset_name = config["name"]
            prediction_length = config["prediction_length"]
            #print("dataset_name", dataset_name ,"   prediction_length", prediction_length, )

            # Check If Trivial Time Series by seeing how AG behaves (Very Quick through Caching)
            #baseline_score_sample = eval_ag(config_hashable=json.dumps(config, sort_keys=True), target=target, metric="MAE")
            #if baseline_score_sample <= 0.0000001 or pd.isna(baseline_score_sample) or math.isinf(baseline_score_sample):
                # Skip Trivial Time Series (Here already to save compute)
            #    continue

            logger.info(f"Loading {target} from {dataset_name}")
            test_data, train_data = load_val_data(config=config, target=target)
            if test_data is None: # Catch DS Not Found
                logger.info(f"Failed locating Target {target} in {dataset_name}")
                continue 

            logger.info(
                f"Generating forecasts for {dataset_name} "
                f"({len(test_data.input)} time series)"
            )

            #print("context_length", context_length)

            all_forecasts, all_labels, all_masks_forecast, all_imputed_values, all_masks_imputation  = [], [], [], [], []

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
                        forecasts, mask_positions = generate_forecasts_bolt()

                    else:
                        forecasts, mask_positions_forecast = generate_forecasts(
                            series=series,
                            model=model,
                            tokenizer=tokenizer,
                            context_length=context_length,
                            prediction_length = prediction_length,
                            num_samples=num_samples,
                            temperature=1.0
                        )
                        all_forecasts.append(forecasts)
                        all_labels.append(series)
                        all_masks_forecast.append(mask_positions_forecast)


                        imputed_values, mask_positions_imputation = impute_span(
                            series=series,
                            model=model,
                            tokenizer=tokenizer,
                            context_length=context_length,
                            mask_ratio=mask_ratio,
                            num_samples=num_samples,
                            temperature=1.0
                        )

                        all_imputed_values.append(imputed_values)
                        all_masks_imputation.append(mask_positions_imputation)

            logger.info("Imputation and Forecast Done")

                    
                    
            #labels arrays
            labels_array = np.stack(all_labels, axis=0)

            #Forecast arrays
            preds_array_forecast = np.stack(all_forecasts, axis=0)
            mask_array_forecast = np.stack(all_masks_forecast, axis=0)

            #Imputation Arrays
            preds_array_imputation = np.stack(all_imputed_values, axis=0)
            mask_array_imputation = np.stack(all_masks_imputation, axis=0)



            # Expand labels to match preds shape
            #Forecast arrays
            labels_array_exp_forecst = np.expand_dims(labels_array, 1).repeat(preds_array_forecast.shape[1], axis=1)
            mask_array_exp_forecast = np.expand_dims(mask_array_forecast, 1).repeat(preds_array_forecast.shape[1], axis=1)

            #Imputation Arrays
            labels_array_exp_imputation = np.expand_dims(labels_array, 1).repeat(preds_array_imputation.shape[1], axis=1)
            mask_array_exp_imputation = np.expand_dims(mask_array_imputation, 1).repeat(preds_array_imputation.shape[1], axis=1)

            #Valid Forevast
            valid_eval_forecast = mask_array_exp_forecast & ~np.isnan(labels_array_exp_forecst)

            #Valid Imputation
            valid_eval_imputation = mask_array_exp_imputation & ~np.isnan(labels_array_exp_imputation)



            # Safe evaluation of metrics
            # if valid_eval_forecast.any():
            #     mae_forecast = float(np.mean(np.abs(preds_array_forecast[valid_eval_forecast] - labels_array_exp_forecst[valid_eval_forecast])))
            #     rmse_forecast = float(np.sqrt(np.mean((preds_array_forecast[valid_eval_forecast] - labels_array_exp_forecst[valid_eval_forecast]) ** 2)))
            # else:
            #     mae_forecast = float('inf')
            #     rmse_forecast = float('inf')

            # if valid_eval_imputation.any():
            #     mae_imputation = float(np.mean(np.abs(preds_array_imputation[valid_eval_imputation] - labels_array_exp_imputation[valid_eval_imputation])))
            #     rmse_imputation = float(np.sqrt(np.mean((preds_array_imputation[valid_eval_imputation] - labels_array_exp_imputation[valid_eval_imputation]) ** 2)))
            # else:
            #     mae_imputation = float('inf')
            #     rmse_imputation = float('inf')

            logger.info("Regular Metrics Done")


            MASE_Imputation, Scaled_RMSE_Imputation = timeseries_level_scaled_metrics(
                labels_array=labels_array,
                preds_array=preds_array_imputation,
                mask_array=mask_array_imputation,
                csv_path = Path("./metrics") / f"{dataset_name}_imputation_series.csv"
                )

            MASE_Forecast, Scaled_RMSE_Forecast = timeseries_level_scaled_metrics(
                labels_array=labels_array,
                preds_array=preds_array_forecast,
                mask_array=mask_array_forecast,
                csv_path = Path("./metrics") / f"{dataset_name}_forecast_series.csv"
            )
            dataset_metrics = pd.DataFrame([{
                "Model_vs_Naive_MAE_Imputation_Series": MASE_Imputation,
                "Model_vs_Naive_RMSE_Imputation_series": Scaled_RMSE_Imputation,

                "Model_vs_Naive_MAE_Forecast_Sereis": MASE_Forecast,
                "Model_vs_Naive_RMSE_Forecast_Sereies": Scaled_RMSE_Forecast,
            }])



            logger.info(f"Metrics:\n{dataset_metrics}")

            metrics.append({
                "dataset": dataset_name,
                "target": target,

                "MASE_Imputation":MASE_Imputation,
                "Scaled_RMSE_Imputation":Scaled_RMSE_Imputation,

                "MASE_Forecast":MASE_Forecast,
                "Scaled_RMSE_Forecast":Scaled_RMSE_Forecast,
                })

    results_df = pd.DataFrame(metrics)
    return results_df


@cache
def eval_ag(
        config_hashable: str,
        target: str,
        metric: str,
) -> float:
    """
    Cacheable Function to train a ZeroShot baseline model quickly via AutoGluon
    AG is used as it automates preprocessing and model selection delivering strong baselines across varied datasets

    Args:
        _config_hashable_ is the dataset config as a json-string (for caching it is important that configs are hashable)
        _target_ the target TS to use of the given dataset 
        _metric_ AutoGluon Metric to use
    Raises:
        "IndexError('single positional indexer is out-of-bounds')" ~ When Metric does not exist
    Returns the value of the best found model using the given metric on the given dataset as a float 
    """
    config = json.loads(config_hashable)
    # Note: Uses double caching (outer cache avoids read from CSV, inner cache avoids retraining for each metric and each worker)
    cache_path = Path("./cache/AG_Scores.csv")
    dataset_name = config["name"]
    prediction_length = config["prediction_length"]

    def get_relevant_scores(df, dataset_name, target, metric) -> pd.DataFrame:
        """Just a simple df lookup that is used repeatedly"""
        return df[
            (df["ds_name"] == dataset_name) &
            (df["target"] == str(target)) &
            (df["metric"] == metric)
        ]

    def chop_tail(df, pred_len):
        return df.groupby("item_id").apply(lambda g: g.iloc[:-pred_len]).reset_index(drop=True)
    
    # Check Cache
    if cache_path.exists():
        cached_scores = pd.read_csv(cache_path)
        relevant_scores = get_relevant_scores(cached_scores, dataset_name=dataset_name, target=target, metric=metric)
        if len(relevant_scores) > 0:
            return relevant_scores["value"].iloc[0]
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
    # load data
    test_data = load_val_data(config=config, target=target, autogluon_format=True)
    test_data['target'] = pd.to_numeric(test_data['target'], errors='coerce')

    train_like = chop_tail(test_data.reset_index(drop=False), pred_len=prediction_length)
    train_like = TimeSeriesDataFrame(
            data=train_like,
            id_column="item_id",
            timestamp_column="timestamp",
        )
    
    # Train AG (Note: Does not need to be perfect, indication how diff to predict a given target suffices)
    # TODO pass eval metric as param and optimize for each independently else MASE is especially good while others are meh
    predictor = TimeSeriesPredictor(
        target="target",
        prediction_length=prediction_length,
        eval_metric=metric,
    )

    # Train and Select best Zeroshot/Statisitical Model 
    predictor.fit(
        train_data=train_like,
        presets="medium_quality",
        enable_ensemble=False,
        time_limit=600,
        hyperparameters={
            "Naive": {},
            "SeasonalNaive": {},
            "ETS": {},
            "Theta": {},
            "Chronos": [
                {"model_path": "bolt_small", "fine_tune": False},
                {"model_path": "tiny", "fine_tune": False},
            ],
        }
    )

    # Eval AG
    scores: dict = predictor.evaluate(
        test_data,
        metrics=[metric], #["SQL", "WQL", "MAE", "MASE", "WAPE", "MSE", "RMSE", "RMSLE", "RMSSE", "MAPE", "SMAPE"],
        use_cache=False
        )
    
    # Write ALL metrics to tall table (for caching)
    results = []
    for my_metric in scores.keys():
        results.append({
            "ds_name": dataset_name, 
            "target": str(target), 
            "metric": my_metric, 
            "value": -1*scores[my_metric], # AG inverts all metrics s.t. larger is better we don't
            "model": predictor.model_best # if Chronos is best (we should be able to beat it) else it might be a weak point of the old chr 
        })
    results = pd.DataFrame(results)
    if cache_path.exists():
        cached_scores = pd.read_csv(cache_path)
        results = pd.concat([cached_scores, results], ignore_index=True)
    results.to_csv(cache_path, index=False)

    relevant_scores = get_relevant_scores(results, dataset_name=dataset_name, target=target, metric=metric)
    if relevant_scores.empty:
        warn(f"Evaluation failed somehow for {dataset_name}, {target}, {metric}")
        return 0

    return relevant_scores["value"].iloc[0]


@cache
def normalize_metric_name(metric_name) -> str:
    """
    Convert convoluted metric name into simple standardized format
    """
    name_map = {
        "MEAN_WEIGHTED_SUM_QUANTILE_LOSS": "WQL",
        "WQL": "WQL",
        "MASE": "MASE",
        "MAE": "MAE",
        "RMSE": "RMSE",  
        "NRMSE": "NRMSE",  
        "WAPE": "WAPE",
        "MSE": "MSE",
        "RMSLE": "RMSLE",
        "RMSSE": "RMSSE",
        "MAPE": "MAPE",
        "SMAPE": "SMAPE",
    }

    base = re.split(r"[\[\]]", metric_name)[0].upper()  # remove brackets and suffixes
    if base in name_map:
        return name_map[base]
    elif metric_name in name_map:
        return name_map[metric_name]
    else:
        raise Exception(f"Metric {metric_name} not found!")
        return metric_name
    
def naive_baseline_metrics(labels_array, mask_array):
    """
    Compute dataset-level MAE and RMSE for the naive baseline:
        - Middle tokens: mean(prev + next)
        - First token: next
        - Last token: prev
    """

    n_series, seq_len = labels_array.shape

    naive_preds = np.full_like(labels_array, np.nan, dtype=np.float32)

    for i in range(n_series):
        for t in range(seq_len):

            if not mask_array[i, t]:
                continue

            prev_val = labels_array[i, t-1] if t > 0 else np.nan
            next_val = labels_array[i, t+1] if t < seq_len - 1 else np.nan

            prev_valid = not np.isnan(prev_val)
            next_valid = not np.isnan(next_val)

            if prev_valid and next_valid:
                naive_preds[i, t] = 0.5 * (prev_val + next_val)

            elif prev_valid:
                naive_preds[i, t] = prev_val

            elif next_valid:
                naive_preds[i, t] = next_val

            else:
                naive_preds[i, t] = np.nan

    # valid positions
    valid_mask = mask_array & ~np.isnan(labels_array) & ~np.isnan(naive_preds)

    abs_error = np.abs(naive_preds - labels_array)

    errors = abs_error[valid_mask]

    errors = errors[np.isfinite(errors)]

    if errors.size == 0:
        return 1e9, 1e9

    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    return mae, rmse




if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Chronos Evaluation")
    logger.setLevel(logging.INFO)
    app()

