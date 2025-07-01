import logging
from pathlib import Path
from typing import Iterable, Optional

# Include Parent Directory to load packages from
import sys  
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"code").resolve()))  
sys.path.append(str((Path(__file__).parent.parent / "chronos_pkg/src").resolve()))

import datasets
import numpy as np
import pandas as pd
import torch
import typer
import yaml
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss, RMSE, MAE, NRMSE
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import SampleForecast
from tqdm.auto import tqdm
from functools import cache
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from utils import load_val_data
import json
from chronos import ChronosPipeline
import re

app = typer.Typer(pretty_exceptions_enable=False)

def generate_sample_forecasts(
    test_data_input: Iterable,
    pipeline: ChronosPipeline,
    prediction_length: int,
    batch_size: int,
    num_samples: int,
    **predict_kwargs,
):
    # Generate forecast samples
    forecast_samples = []
    for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
        context = [torch.tensor(entry["target"]) for entry in batch]
        forecast_samples.append(
            pipeline.predict(
                context,
                prediction_length=prediction_length,
                num_samples=num_samples,
                **predict_kwargs,
            ).numpy()
        )
    forecast_samples = np.concatenate(forecast_samples)

    # Convert forecast samples into gluonts SampleForecast objects
    sample_forecasts = []
    for item, ts in zip(forecast_samples, test_data_input):
        forecast_start_date = ts["start"] + len(ts["target"])
        sample_forecasts.append(
            SampleForecast(samples=item, start_date=forecast_start_date)
        )

    return sample_forecasts


@app.command()
def main(
    config_path: Path,
    metrics_path: Path,
    chronos_model_id: str = "amazon/chronos-t5-small",
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    batch_size: int = 32,
    num_samples: int = 20,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
):
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)

    # Load Chronos
    pipeline = ChronosPipeline.from_pretrained(
        chronos_model_id,
        device_map=device,
        torch_dtype=torch_dtype,
    )

    # Load backtest configs
    with open(config_path) as fp:
        backtest_configs = yaml.safe_load(fp)

    result_rows = []
    for config in backtest_configs:
        targets = config.pop("targets", ["target"])
        for target in targets:
            dataset_name = config["name"]
            prediction_length = config["prediction_length"]

            logger.info(f"Loading {dataset_name}")
            test_data = load_val_data(config=config, target=target)

            logger.info(
                f"Generating forecasts for {dataset_name} "
                f"({len(test_data.input)} time series)"
            )
            sample_forecasts = generate_sample_forecasts(
                test_data.input,
                pipeline=pipeline,
                prediction_length=prediction_length,
                batch_size=batch_size,
                num_samples=num_samples,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            logger.info(f"Evaluating forecasts for {dataset_name}")
            metrics = (
                evaluate_forecasts(
                    sample_forecasts,
                    test_data=test_data,
                    metrics=[
                        MASE(),
                        MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
                        MAE(),
                        NRMSE()
                    ],
                    batch_size=5000,
                )
                .reset_index(drop=True)
                .to_dict(orient="records")
            )

            logger.info(f"Metrics:\n{metrics}") 
            print(f"Metrics:\n{metrics}") 
            # logger.error(f"Metrics:\n{metrics}") 

            # Get Baseline for Normaliasation & Comparability
            metrics = metrics[0]
            results = {}
            for metric_name, value in metrics.items():
                print(metric_name)
                metric_name_norm = normalize_metric_name(metric_name=metric_name)
                print(metric_name_norm)
                print(value)
                result_rows[metric_name_norm] = value / eval_ag(config_hashable=json.dumps(config, sort_keys=True), target=target, metric=metric_name_norm.upper())

            result_rows.append(
                {"dataset": dataset_name, "model": chronos_model_id, **results}
            )

    # Save results to a CSV file
    results_df = (
        pd.DataFrame(result_rows).sort_values(by="dataset")
    )
    #results_df.to_csv(metrics_path, index=False)
    return results_df


@cache
def eval_ag(
        config_hashable: str,
        target: str,
        metric: str
):
    config = json.loads(config_hashable)
    # Note Use double caching (outer cache avoids read from CSV, inner cache avoids retraining for each metric)
    cache_path = Path("./cache/AG_Scores.csv")
    dataset_name = config["name"]
    prediction_length = config["prediction_length"]
    
    # Check Cache
    if cache_path.exists():
        cached_scores = pd.read_csv(cache_path)
        relevant_scores = cached_scores[cached_scores[["ds_name", "target", "metric"]] == (dataset_name, target, metric)]
        if len(relevant_scores) > 0:
            return relevant_scores["value"][0]
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
    # TODO load data
    test_data = load_val_data(config=config, target=target, autogluon_format=True)
    test_data['target'] = pd.to_numeric(test_data['target'], errors='coerce')
    def chop_tail(df, pred_len):
        return df.groupby("item_id").apply(lambda g: g.iloc[:-pred_len]).reset_index(drop=True)

    train_like = chop_tail(test_data.reset_index(drop=False), pred_len=prediction_length)
    train_like = TimeSeriesDataFrame(
            data=train_like,
            id_column="item_id",
            timestamp_column="timestamp",
        )
    
    # Train AG (Note: Does not need to be perfect, indication how diff to predict a given target suffices)
    predictor = TimeSeriesPredictor(
        target="target",
        prediction_length=prediction_length,
        eval_metric="MASE",
    )

    # Train Ensemble of Zeroshot and Statisitical Models they require less train data and are robust to test set leakage due to overlap btwn samples
    predictor.fit(
        train_data=train_like,
        presets="best_quality",
        time_limit=600,
        hyperparameters={
            "Naive": {},
            "SeasonalNaive": {},
            "ETS": {},
            "Theta": {},
            # "RecursiveTabular": {},
            # "DirectTabular": {},
            # "TemporalFusionTransformer": {},
            "Chronos": [
                {"model_path": "bolt_small", "fine_tune": False},
                {"model_path": "tiny", "fine_tune": False},
            ],
        }
    )
    # Eval AG
    scores: dict = predictor.evaluate(
        test_data,
        metrics=["SQL", "WQL", "MAE", "MASE", "WAPE", "MSE", "RMSE", "RMSLE", "RMSSE", "MAPE", "SMAPE"],
        use_cache=False
        )
    
    # Write ALL metrics to tall table
    results = []
    for my_metric in scores.keys():
        results.append({
            "ds_name": dataset_name, 
            "target": target, 
            "metric": my_metric, 
            "value": -1*scores[my_metric], # AG inverts all metrics s.t. larger is better we don't
        })
    results = pd.DataFrame(results)
    if cache_path.exists():
        cached_scores = pd.read_csv(cache_path)
        results = pd.concat([cached_scores, results])
    results.to_csv(cache_path, index=False)

    # Return Value only for selected metric though
    return scores[metric]


def normalize_metric_name(metric_name) -> str:
    """
    Convert convoluted metric name into simple standardized format
    """
    name_map = {
        "mean_weighted_sum_quantile_loss": "WQL",
        "weighted_sum_quantile_loss": "SQL",
        "MASE": "MASE",
        "MAE": "MAE",
        "NRMSE": "RMSE",  # Assuming this is what you mean
        "WAPE": "WAPE",
        "MSE": "MSE",
        "RMSLE": "RMSLE",
        "RMSSE": "RMSSE",
        "MAPE": "MAPE",
        "SMAPE": "SMAPE",
    }

    base = re.split(r"[\[\]_]", metric_name)[0]  # remove brackets and suffixes
    for name in name_map:
        if base in name or name in base:
            return name_map[name]



if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Chronos Evaluation")
    logger.setLevel(logging.INFO)
    app()