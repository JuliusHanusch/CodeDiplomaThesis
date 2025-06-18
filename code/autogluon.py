import pandas as pd
import numpy as np
from datasets import load_dataset
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from gluonts.evaluation import make_evaluation_predictions
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss, MAE, NRMSE

def normalize_time_series_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw time series dataframe to AutoGluon's expected format:
    Columns: item_id, timestamp, target
    """
    cols = df.columns[:3]
    id_col, ts_col, target_col = cols[0], cols[1], cols[2]

    if isinstance(df.iloc[0][ts_col], (list, np.ndarray)):
        # Flatten nested format
        new_rows = []
        for _, row in df.iterrows():
            item_id = row[id_col]
            timestamps = row[ts_col]    
            targets = row[target_col]
            for t, y in zip(timestamps, targets):
                new_rows.append((item_id, t, y))
        df_flat = pd.DataFrame(new_rows, columns=["item_id", "timestamp", "target"])
    else:
        df_flat = df.rename(columns={id_col: "item_id", ts_col: "timestamp", target_col: "target"})[
            ["item_id", "timestamp", "target"]
        ]

    df_flat["timestamp"] = pd.to_datetime(df_flat["timestamp"])
    return df_flat

print("Start processing datasets")

DATASET_CONFIGS = {
    "monash_electricity_hourly": {"prediction_length": 24, "freq": "h"},
}

results = []

for dataset_name, config in DATASET_CONFIGS.items():
    print(f"\n=== Processing dataset: {dataset_name} ===")

    ds = load_dataset("autogluon/chronos_datasets", dataset_name)
    df_train_raw = ds["train"].to_pandas()
    df_test_raw = ds["test"].to_pandas()

    print(df_train_raw.head())

    df_train = normalize_time_series_dataframe(df_train_raw)
    df_test = normalize_time_series_dataframe(df_test_raw)

    ts_train = TimeSeriesDataFrame.from_data_frame(df_train, id_column="item_id", timestamp_column="timestamp")
    ts_test = TimeSeriesDataFrame.from_data_frame(df_test, id_column="item_id", timestamp_column="timestamp")

    # Optional sanity check
    print(f"Train shape: {ts_train.shape}, Test shape: {ts_test.shape}")

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        path=f"autogluon-predictor-{dataset_name}",
        target="target",
        eval_metric="MASE"
    )

    predictor.fit(ts_train, presets="medium_quality", time_limit=300)

    evaluation = predictor.evaluate(ts_test)



    mase = evaluation["MASE"]
    wql = evaluation.get("MeanWeightedSumQuantileLoss")
    mae = evaluation.get("MAE")
    nrmse = evaluation.get("NRMSE")

    print(f"MASE: {mase:.4f}, WQL: {wql:.4f}, MAE: {mae:.4f}, NRMSE: {nrmse:.4f}")

    results.append({
        "dataset": dataset_name,
        "MASE": mase,
        "WQL": wql,
        "MAE": mae,
        "NRMSE": nrmse,
    })

results_df = pd.DataFrame(results)
print("\n=== Evaluation Results ===")
print(results_df.to_string(index=False))
