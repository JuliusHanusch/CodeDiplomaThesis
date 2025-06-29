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

from chronos import ChronosPipeline

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
        dataset_name = config["name"]
        prediction_length = config["prediction_length"]

        logger.info(f"Loading {dataset_name}")
        test_data = load_val_data(config=config)

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

        # TODO Get Baseline for Normaliasation & Comparability

        result_rows.append(
            {"dataset": dataset_name, "model": chronos_model_id, **metrics[0]}
        )

    # Save results to a CSV file
    results_df = (
        pd.DataFrame(result_rows)
        .rename(
            {
                "MASE[0.5]": "MASE",
                "mean_weighted_sum_quantile_loss": "WQL",
                "MAE[0.5]": "MAE",
                "NRMSE[mean]": "NRMSE",


            },
            axis="columns",
        )
        .sort_values(by="dataset")
    )
    #results_df.to_csv(metrics_path, index=False)
    return results_df


@cache
def eval_ag(
        test_data_path: str, # we need a string to be hashable and cashable 
        config: dict,
):
    # TODO
    # Setup predictor
    dataset_name = config["name"]
    prediction_length = config["prediction_length"]
    hf_repo = config["hf_repo"]
    trust_remote_code = True if hf_repo == "autogluon/chronos_datasets_extra" else False


    train_data = datasets.load_dataset(
        hf_repo, dataset_name, split="train", trust_remote_code=trust_remote_code
    )

    test_data = datasets.load_dataset(
        hf_repo, dataset_name, split="test", trust_remote_code=trust_remote_code
    )


    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        path=f"autogluon-predictor-{dataset_name}",
        target="target",
        eval_metric="MASE"
    )    

    predictor.fit(
        train_data,
        presets="medium_quality",
        time_limit=300
    )

    # Evaluate on all metrics
    # (Cache)
    # return all metrics
    pass # Return MASE, WQL, NRMSE


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Chronos Evaluation")
    logger.setLevel(logging.INFO)
    app()