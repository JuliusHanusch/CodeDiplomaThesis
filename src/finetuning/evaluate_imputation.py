import logging
from pathlib import Path
from typing import Iterable, Optional

# Include Parent Directory to load packages from
import sys  
# root_dir = Path(__file__).parent.parent
# sys.path.append(str(root_dir.resolve()))  
# sys.path.append(str((root_dir/"code").resolve()))  
# sys.path.append(str((Path(__file__).parent.parent / "chronos_pkg/src").resolve()))

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
from gluonts.itertools import batcher
from tqdm.auto import tqdm

#ColabImport
ROOT = "/content/CodeDiplomaThesis"
sys.path.append(str(Path(ROOT).resolve()))
from src.utils import load_val_data
from chronos_pkg.src.chronos import ChronosConfig
from chronos_pkg.src.chronos import ChronosPipeline
from chronos_pkg.src.chronos.chronos_bolt import ChronosBoltModelForForecasting, ChronosBoltConfig

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

                if (
                    not np.isnan(labels_array[i, pos])
                    and not mask_array[i, pos]   # cannot use masked targets
                ):
                    prev_val = labels_array[i, pos]
                    break

            # search up to 10 tokens to the right
            for k in range(1, 11):
                pos = t + k
                if pos >= seq_len:
                    break

                if (
                    not np.isnan(labels_array[i, pos])
                    and not mask_array[i, pos]   # cannot use masked targets
                ):
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

    mae_scaled_mean = float(np.mean(mae_scaled_series)) if mae_scaled_series else np.nan
    rmse_scaled_mean = float(np.mean(rmse_scaled_series)) if rmse_scaled_series else np.nan


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



@app.command()
def main(
    config_path: Path,
    chronos_model_id: str = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/chronos_models/chronos_-6611716250758806883/run-0/checkpoint-final",
    device: str = "cuda",
    torch_dtype: str = "float32",
    batch_size: int = 32,
    num_samples: int = 20,
    mean_span_length = 3,
    mask_ratio: int = 0.15,
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
                            mask_ratio=mask_ratio,
                            num_samples=num_samples,
                            temperature=1.0
                        )

                        all_imputed_values.append(imputed_values)
                        all_masks_imputation.append(mask_positions_imputation)

            logger.info("Imputation and Forecast Done")       
                    
            #labels arrays
            labels_array = np.stack(all_labels, axis=0)

            #Imputation Arrays
            preds_array_imputation = np.stack(all_imputed_values, axis=0)
            mask_array_imputation = np.stack(all_masks_imputation, axis=0)

            logger.info("Regular Metrics Done")


            MASE_Imputation, Scaled_RMSE_Imputation = timeseries_level_scaled_metrics(
                labels_array=labels_array,
                preds_array=preds_array_imputation,
                mask_array=mask_array_imputation,
                csv_path = Path("./metrics") / f"{dataset_name}_imputation_series.csv"
                )

            dataset_metrics = pd.DataFrame([{
                "Model_vs_Naive_MAE_Imputation_Series": MASE_Imputation,
                "Model_vs_Naive_RMSE_Imputation_series": Scaled_RMSE_Imputation,
            }])



            logger.info(f"Metrics:\n{dataset_metrics}")

            metrics.append({
                "dataset": dataset_name,
                "target": target,

                "MASE_Imputation":MASE_Imputation,
                "Scaled_RMSE_Imputation":Scaled_RMSE_Imputation,
                })

    results_df = pd.DataFrame(metrics)
    csv_path = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/Results/Finetuning/Imputation/MSL_3_results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)

    return



if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Chronos Evaluation")
    logger.setLevel(logging.INFO)
    app()

