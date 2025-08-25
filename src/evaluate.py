import logging
from pathlib import Path
from typing import Iterable, Optional

# Include Parent Directory to load packages from
import sys  
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"code").resolve()))  
sys.path.append(str((Path(__file__).parent.parent / "chronos_pkg/src").resolve()))

import numpy as np
import pandas as pd
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
from chronos import ChronosPipeline
from chronos_pkg.src.chronos.chronos_bolt import ChronosBoltPipeline
import re
from math import log
from utils import get_model_size

app = typer.Typer(pretty_exceptions_enable=False)

def generate_sample_forecasts(
    test_data_input: Iterable,
    pipeline: ChronosPipeline,
    prediction_length: int,
    batch_size: int,
    #num_samples: int,
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
                #num_samples=num_samples,
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
    chronos_model_id: str = "amazon/chronos-t5-small",
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    batch_size: int = 32,
    num_samples: int = 20,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    bolt: bool = False,
):
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)

    if bolt:
        PipelineClass = ChronosBoltPipeline
    else:
        PipelineClass = ChronosPipeline
    # Load Chronos
    pipeline = PipelineClass.from_pretrained(
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

            logger.info(f"Loading {target} from {dataset_name}")
            test_data = load_val_data(config=config, target=target)
            if test_data is None: # Catch DS Not Found
                logger.info(f"Failed locating Target {target} in {dataset_name}")
                continue 

            logger.info(
                f"Generating forecasts for {dataset_name} "
                f"({len(test_data.input)} time series)"
            )
            if bolt: # TODO Fusion into one clean version
                sample_forecasts = generate_sample_forecasts(
                    test_data.input,
                    pipeline=pipeline,
                    prediction_length=prediction_length,
                    batch_size=batch_size,
                )
            else:
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
                        #NRMSE(),
                        RMSE(),
                        #MSE(), 
                        #MAPE(), 
                        SMAPE()
                    ],
                    batch_size=5000,
                )
                .reset_index(drop=True)
                .to_dict(orient="records")
            )

            logger.info(f"Metrics:\n{metrics}") 

            # Get Baseline for Normaliasation & Comparability
            metrics = metrics[0]
            results = {"parameters": get_model_size(pipeline.model)}
            for metric_name, value in metrics.items():
                metric_name_norm = normalize_metric_name(metric_name=metric_name)
                baseline_score = eval_ag(config_hashable=json.dumps(config, sort_keys=True), target=target, metric=metric_name_norm.upper())
                results[metric_name_norm] = log(value / baseline_score)
                logger.info(f"\nMetric: {metric_name} -> {metric_name_norm}\nOriginal: {value}\nBaseline: {baseline_score}\nNew: {results[metric_name_norm]}")

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
        metric: str,
        force_return: bool = False, # Returns 1 if not found (for metrics not supported by AG)
) -> float:
    config = json.loads(config_hashable)
    # Note Use double caching (outer cache avoids read from CSV, inner cache avoids retraining for each metric)
    cache_path = Path("./cache/AG_Scores.csv")
    dataset_name = config["name"]
    prediction_length = config["prediction_length"]
    
    # Check Cache
    if cache_path.exists():
        cached_scores = pd.read_csv(cache_path)
        relevant_scores = cached_scores[
            (cached_scores["ds_name"] == dataset_name) &
            (cached_scores["target"] == target) &
            (cached_scores["metric"] == metric)
        ]
        if len(relevant_scores) > 0:
            return relevant_scores["value"].iloc[0]
        elif force_return: # Return default if metric not supported
            logger.warning(f"Metric {metric} not found for {dataset_name} and {target} AG might not support the metric. Return 1 due to force_return")
            # Cache Default
            if cache_path.exists():
                cached_scores = pd.read_csv(cache_path)
                def_row = pd.DataFrame([{
                    "ds_name": dataset_name, 
                    "target": target, 
                    "metric": metric, 
                    "value": 1.0, # AG inverts all metrics s.t. larger is better we don't
                }])
                results = pd.concat([cached_scores, def_row], ignore_index=True)
            results.to_csv(cache_path, index=False)
            return 1.0
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
    # load data
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

    # Select and Train best Zeroshot and Statisitical Models they require less train data and are robust to test set leakage due to overlap btwn samples
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
            "model": predictor.model_best # if Chronos is best (we should be able to beat it) else it might be a weak point of the old chr 
        })
    results = pd.DataFrame(results)
    if cache_path.exists():
        cached_scores = pd.read_csv(cache_path)
        results = pd.concat([cached_scores, results], ignore_index=True)
    print(results)
    results.to_csv(cache_path, index=False)

    # Call again now that entry exists (if still not exists return 1)
    return eval_ag(config_hashable, target=target, metric=metric, force_return=True)


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



if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Chronos Evaluation")
    logger.setLevel(logging.INFO)
    app()