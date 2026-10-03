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
import time
import random

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

from chronos_pkg.src.chronos import ChronosConfig
from chronos_pkg.src.chronos.chronos_bolt import ChronosBoltModelForForecasting, ChronosBoltConfig
from src.utils import load_val_data

DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/imputation/imputation_allData_new.db"

app = typer.Typer(pretty_exceptions_enable=False)

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_db_connection(DB_PATH):
    conn = sqlite3.connect(
        DB_PATH,
        timeout=60,
    )
    conn.execute("PRAGMA busy_timeout=60000")
    return conn

def execute_db_update(sql, params, description="database update"):
    """
    Execute a SQLite write with up to 5 attempts if the
    database is locked.
    """

    for attempt in range(5):

        conn = None

        try:
            conn = get_db_connection(DB_PATH)

            conn.execute(sql, params)
            conn.commit()
            conn.close()

            return

        except sqlite3.OperationalError as e:

            if conn is not None:
                conn.close()

            if "database is locked" not in str(e):
                raise

            print(
                f"[SQLite] Database locked during {description} "
                f"- retry {attempt + 1}/5",
                flush=True
            )

            if attempt < 4:
                time.sleep(5)

    raise RuntimeError(
        f"[SQLite] Database remained locked during "
        f"{description} after 5 attempts."
    )

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


def timeseries_level_scaled_metrics(
    labels_array,
    preds_array,
    mask_array
):
    # Probabilistic predictions -> median
    if preds_array.ndim == 3:
        preds_median = np.median(preds_array, axis=1)
    else:
        preds_median = preds_array

    valid_mask = mask_array & ~np.isnan(labels_array)

    error_model = np.abs(
        preds_median - labels_array
    )

    mae_series = []
    mase_series = []

    mae_linear_series = []
    mase_linear_series = []


    for i in tqdm(
        range(labels_array.shape[0]),
        desc="Series"
    ):

        token_mask = valid_mask[i]

        if not np.any(token_mask):
            continue

        series = labels_array[i]


        mae_model = np.mean(
            error_model[i, token_mask]
        )

        mae_series.append(mae_model)

        # =====================================================
        # 2. midpoint interpolation baseline
        # =====================================================

        baseline_pred = np.full_like(
            series,
            np.nan
        )

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

            # Find closest observed value on the left
            left_value = None

            if start > 0:
                j = start - 1

                while j >= 0:

                    if (
                        not mask[j]
                        and not np.isnan(series[j])
                    ):
                        left_value = series[j]
                        break

                    j -= 1

            # Find closest observed value on the right
            right_value = None

            if end < len(series):
                j = end

                while j < len(series):

                    if (
                        not mask[j]
                        and not np.isnan(series[j])
                    ):
                        right_value = series[j]
                        break

                    j += 1

            # Midpoint interpolation
            if (
                left_value is not None
                and right_value is not None
            ):
                fill = 0.5 * (
                    left_value + right_value
                )

            elif left_value is not None:
                fill = left_value

            elif right_value is not None:
                fill = right_value

            else:
                fill = np.nan

            baseline_pred[start:end] = fill

        # Linear interpolation MAE
        baseline_error = np.abs(
            baseline_pred[token_mask]
            - series[token_mask]
        )

        mae_baseline = np.mean(
            baseline_error
        )

        mae_linear_series.append(
            mae_baseline
        )

        observed_mask = (
            ~mask_array[i]
            & ~np.isnan(series)
        )

        if not np.any(observed_mask):
            continue

        season_mean = np.mean(
            series[observed_mask]
        )

        season_mean_error = np.abs(
            season_mean - series[token_mask]
        )

        mae_seasonal = np.mean(
            season_mean_error
        )


        mase_model = mae_model / (
            mae_seasonal + 1e-8
        )

        mase_linear = mae_baseline / (
            mae_seasonal + 1e-8
        )

        mase_series.append(
            mase_model
        )

        mase_linear_series.append(
            mase_linear
        )



    MASE_model = (
        float(np.mean(mase_series))
        if mase_series
        else np.nan
    )

    MAE_model = (
        float(np.mean(mae_series))
        if mae_series
        else np.nan
    )

    MASE_baseline = (
        float(np.mean(mase_linear_series))
        if mase_linear_series
        else np.nan
    )

    MAE_baseline = (
        float(np.mean(mae_linear_series))
        if mae_linear_series
        else np.nan
    )

    return (
        MASE_model,
        MAE_model,
        MASE_baseline,
        MAE_baseline,
    )


def impute_span(
    series: np.ndarray,
    mask_positions: np.ndarray,
    model,
    tokenizer,
    num_samples: int = 10,
    temperature: float = 1.0,
):
    device = model.device
    model.eval()

    series = np.asarray(series, dtype=np.float32)
    mask_positions = np.asarray(mask_positions, dtype=bool)

    if len(series) != len(mask_positions):
        raise ValueError(
            f"Series has {len(series)} values, "
            f"but mask has {len(mask_positions)} positions."
        )

    series_tensor = torch.tensor(
        series,
        dtype=torch.float32,
        device="cpu",
    )

    input_ids, attention_mask, scale = tokenizer.context_input_transform(
        series_tensor.unsqueeze(0)
    )

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    scale = scale.to(device)

    special_token_cutoff = tokenizer.config.n_special_tokens
    special_tokens_mask = input_ids < special_token_cutoff
    valid_positions = ~special_tokens_mask

    valid_token_indices = valid_positions[0].nonzero(as_tuple=True)[0]
    n_valid_tokens = len(valid_token_indices)

    # Map the value-level mask to Chronos token positions
    n_values = len(mask_positions)
    token_mask = np.zeros(n_valid_tokens, dtype=bool)

    for token_idx in range(n_valid_tokens):
        value_start = int(np.floor(token_idx * n_values / n_valid_tokens))
        value_end = int(np.floor((token_idx + 1) * n_values / n_valid_tokens))
        value_end = min(max(value_end, value_start + 1), n_values)

        token_mask[token_idx] = np.any(
            mask_positions[value_start:value_end]
        )
    mask = torch.zeros_like(input_ids, dtype=torch.bool)

    token_mask_tensor = torch.tensor(
        token_mask,
        dtype=torch.bool,
        device=device,
    )

    mask[0, valid_token_indices] = token_mask_tensor
    mask &= valid_positions
    mask_positions_token = mask[0]

    mask_token_id = getattr(
        tokenizer.config,
        "mask_token_id",
        None,
    )

    if mask_token_id is None:
        raise ValueError("Tokenizer does not define mask_token_id.")

    masked_input = input_ids.clone()
    masked_input[mask] = mask_token_id

    with torch.no_grad():
        outputs = model(
            input_ids=masked_input,
            attention_mask=attention_mask,
        )

        logits = outputs.logits[0]
        probs = torch.nn.functional.softmax(
            logits / float(temperature),
            dim=-1,
        )

    sampled_token_ids = masked_input[0].repeat(num_samples, 1)

    masked_idx_list = mask_positions_token.nonzero(
        as_tuple=True
    )[0].tolist()

    for pos in masked_idx_list:
        p = probs[pos].clone()

        if torch.isnan(p).any() or p.sum() == 0:
            p = torch.ones_like(p) / p.numel()
        else:
            p /= p.sum()

        sampled_token_ids[:, pos] = torch.multinomial(
            p,
            num_samples,
            replacement=True,
        )

    sampled_ids_for_output = sampled_token_ids.unsqueeze(0)
    values = tokenizer.output_transform(
        sampled_ids_for_output.cpu(),
        scale.cpu(),
    )

    values = values.squeeze(0).cpu().numpy()
    unmasked_ids = input_ids[0, ~mask_positions_token].cpu()
    unmasked_values = tokenizer.output_transform(
        unmasked_ids.unsqueeze(0),
        scale.cpu(),
    ).squeeze(0).cpu().numpy()

    mask_np = mask_positions_token.cpu().numpy()

    for i in range(num_samples):
        values[i, ~mask_np] = unmasked_values

    return values.astype(np.float32), mask_np



def main(
    idx: int,
    device: str = "cuda",
    torch_dtype: str = "float32",
    batch_size: int = 32,
    num_samples: int = 20,
):


    set_seed(SEED)

    conn = get_db_connection(DB_PATH)
    cur = conn.cursor()


    cur.execute(f"""
        SELECT
            eval_data,
            model_path,
            config
        FROM runs
        WHERE id=?
    """, (idx,))

    row = cur.fetchone()
    conn.close()

    if row is None:
        raise ValueError(f"No run with id={idx}")

    eval_path, model_path, config_json = row

    model, tokenizer, context_length = load_chronos_bert(
        model_path,
        device,
        torch_dtype,
    )

    data = np.load(
        eval_path,
        allow_pickle=True,
    )

    eval_series = data["series"]

    eval_path_obj = Path(eval_path)

    mask_path = (
        eval_path_obj.parent / "val_masks.npz"
    )

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask file not found: {mask_path}"
        )

    mask_data = np.load(
        mask_path,
        allow_pickle=True,
    )

    masks = {
        0.125: mask_data["mask_0125"],
        0.25:  mask_data["mask_025"],
        0.375: mask_data["mask_0375"],
        0.5:   mask_data["mask_050"],
    }

    MASKING_RATIOS = [
        0.125,
        0.25,
        0.375,
        0.5,
    ]

    results = {}

    for masking_ratio in MASKING_RATIOS:

        print(f"\nEvaluating masking ratio: {masking_ratio}")

        all_labels = []
        all_imputed_values = []
        all_masks_imputation = []

        series_indices = range(len(eval_series))

        for series_idx in tqdm(
            series_indices,
            desc=f"Mask {masking_ratio}"
        ):

            series = np.asarray(
                eval_series[series_idx],
                dtype=np.float32,
            )

            mask_full = masks[masking_ratio]

            n_windows = len(series) // context_length

            for window_idx in range(n_windows):

                start = window_idx * context_length
                end = start + context_length

                window = series[start:end]
                window_mask = mask_full[start:end]

                imputed_values, mask_positions = impute_span(
                    series=window,
                    mask_positions=window_mask,
                    model=model,
                    tokenizer=tokenizer,
                    num_samples=num_samples,
                    temperature=1.0,
                )

                all_labels.append(window)
                all_imputed_values.append(imputed_values)
                all_masks_imputation.append(mask_positions)



        # Convert all series to arrays
        labels_array = np.stack(
            all_labels,
            axis=0
        )

        preds_array_imputation = np.stack(
            all_imputed_values,
            axis=0
        )

        mask_array_imputation = np.stack(
            all_masks_imputation,
            axis=0
        )


        # Calculate metrics for this masking ratio
        MASE_Chronos, MAE_Chronos, MASE_baseline, MAE_baseline = (
            timeseries_level_scaled_metrics(
                labels_array=labels_array,
                preds_array=preds_array_imputation,
                mask_array=mask_array_imputation,
            )
        )


        results[masking_ratio] = {
            "MAE": MAE_Chronos,
            "MASE": MASE_Chronos,
            "MAE_Lin": MAE_baseline,
            "MASE_Lin": MASE_baseline,
        }


        print(
            f"Mask {masking_ratio}: "
            f"MAE={MAE_Chronos:.5f}, "
            f"MASE={MASE_Chronos:.5f}"
        )


    avg_MAE = np.mean(
        [
            x["MAE"]
            for x in results.values()
        ]
    )

    avg_MASE = np.mean(
        [
            x["MASE"]
            for x in results.values()
        ]
    )

    avg_MAE_Lin = np.mean(
        [
            x["MAE_Lin"]
            for x in results.values()
        ]
    )

    avg_MASE_Lin = np.mean(
        [
            x["MASE_Lin"]
            for x in results.values()
        ]
    )

    execute_db_update(
        f"""
        UPDATE runs
        SET
            MAE_0125=?,
            MASE_0125=?,
            MAE_Lin_0125=?,
            MASE_Lin_0125=?,

            MAE_025=?,
            MASE_025=?,
            MAE_Lin_025=?,
            MASE_Lin_025=?,

            MAE_0375=?,
            MASE_0375=?,
            MAE_Lin_0375=?,
            MASE_Lin_0375=?,

            MAE_050=?,
            MASE_050=?,
            MAE_Lin_050=?,
            MASE_Lin_050=?,

            MAE_avg=?,
            MASE_avg=?,
            MAE_Lin_avg=?,
            MASE_Lin_avg=?

        WHERE id=?
        """,
        (
            results[0.125]["MAE"],
            results[0.125]["MASE"],
            results[0.125]["MAE_Lin"],
            results[0.125]["MASE_Lin"],

            results[0.25]["MAE"],
            results[0.25]["MASE"],
            results[0.25]["MAE_Lin"],
            results[0.25]["MASE_Lin"],

            results[0.375]["MAE"],
            results[0.375]["MASE"],
            results[0.375]["MAE_Lin"],
            results[0.375]["MASE_Lin"],

            results[0.5]["MAE"],
            results[0.5]["MASE"],
            results[0.5]["MAE_Lin"],
            results[0.5]["MASE_Lin"],

            avg_MAE,
            avg_MASE,
            avg_MAE_Lin,
            avg_MASE_Lin,

            idx,
        ),
        description=f"results update (IDX {idx})"
    )

    print("Finished evaluation")

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

    main(
        idx=args.index,
    )

