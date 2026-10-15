import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler



MASK_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/data/finetuning/Imputation"
)

DATA_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/data/finetuning/Imputation/raw"
)

RESULT_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/Results/Finetuning/Imputation"
)

RESULT_ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]

SEQ_LEN = 512
STRIDE = 512
BATCH_SIZE = 32

EVAL_MASK_RATIOS = [0.125, 0.25, 0.375, 0.5]

MASK_KEYS = {
    0.125: "mask_0125",
    0.25: "mask_025",
    0.375: "mask_0375",
    0.5: "mask_050",
}


SEASONAL_PERIODS = {
    "ETTh1": 24,
    "ETTh2": 24,
    "ETTm1": 96,
    "ETTm2": 96,
}


class InformerDataset:

    def __init__(
        self,
        data,
        data_split="train",
        data_stride_len=512,
    ):
        self.seq_len = SEQ_LEN
        self.full_file_path_and_name = data
        self.data_split = data_split
        self.data_stride_len = data_stride_len

        self._read_data()

    def _get_borders(self):
        name = Path(self.full_file_path_and_name).stem

        if name.startswith("ETTh"):
            n_train = 12 * 30 * 24
            n_val = 4 * 30 * 24
            n_test = 4 * 30 * 24

        elif name.startswith("ETTm"):
            n_train = 12 * 30 * 24 * 4
            n_val = 4 * 30 * 24 * 4
            n_test = 4 * 30 * 24 * 4

        else:
            raise ValueError(f"Unknown dataset: {name}")

        train_end = n_train
        val_end = n_train + n_val

        test_start = val_end - self.seq_len
        test_end = test_start + n_test + self.seq_len

        return (
            slice(0, train_end),
            slice(train_end, val_end),
            slice(test_start, test_end),
        )

    def _read_data(self):
        df = pd.read_csv(self.full_file_path_and_name)

        df.drop(columns=["date"], inplace=True)
        df = df[["OT"]]

        df = df.infer_objects(copy=False).interpolate(
            method="cubic"
        )

        self.original_data = df.to_numpy(dtype=np.float64)

        splits = self._get_borders()

        train_data = self.original_data[splits[0]]

        self.scaler = StandardScaler()
        self.scaler.fit(train_data)

        if self.data_split == "train":
            self.data = self.scaler.transform(
                self.original_data[splits[0]]
            )

        elif self.data_split == "val":
            self.data = self.scaler.transform(
                self.original_data[splits[1]]
            )

        elif self.data_split == "test":
            self.data = self.scaler.transform(
                self.original_data[splits[2]]
            )

        else:
            raise ValueError(
                f"Unknown split: {self.data_split}"
            )

        self.train_mean = float(np.mean(train_data))
        self.train_median = float(np.median(train_data))

    def __getitem__(self, index):
        start = self.data_stride_len * index
        end = start + self.seq_len

        timeseries = self.data[start:end, :].T
        input_mask = np.ones(self.seq_len, dtype=np.float32)

        return timeseries, input_mask

    def __len__(self):
        return (
            (len(self.data) - self.seq_len)
            // self.data_stride_len
            + 1
        )


def get_test_loader(dataset_name):
    data_path = DATA_ROOT / f"{dataset_name}.csv"

    dataset = InformerDataset(
        data=str(data_path),
        data_split="test",
        data_stride_len=STRIDE,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    return loader, dataset


def impute_linear(x, missing, train_mean, period):
    """Linear interpolation with edge extrapolation."""
    result = x.copy()
    observed = ~missing

    if not np.any(observed):
        result[:] = train_mean
        return result

    positions = np.arange(len(x))

    result[missing] = np.interp(
        positions[missing],
        positions[observed],
        x[observed],
    )

    return result


def impute_forward_fill(x, missing, train_mean, period):
    """Carry the last observed value forward."""
    result = x.copy()
    last_value = train_mean

    for i in range(len(x)):
        if not missing[i]:
            last_value = x[i]
        else:
            result[i] = last_value

    return result


def impute_backward_fill(x, missing, train_mean, period):
    """Carry the next observed value backward."""
    result = x.copy()
    next_value = train_mean

    for i in range(len(x) - 1, -1, -1):
        if not missing[i]:
            next_value = x[i]
        else:
            result[i] = next_value

    return result


def impute_mean(x, missing, train_mean, period):
    """Use the global training-set mean."""
    result = x.copy()
    result[missing] = train_mean
    return result


def impute_median(x, missing, train_mean, period):
    """Use the global training-set median."""
    result = x.copy()

    result[missing] = train_mean
    return result


def impute_seasonal_mean(x, missing, train_mean, period):
    """
    Estimate missing values using observed values at the
    same seasonal position within the current window.

    If no seasonal observations are available for a position,
    use the global training-set mean.
    """
    result = x.copy()
    observed = ~missing

    if not np.any(observed):
        result[:] = train_mean
        return result

    for i in np.flatnonzero(missing):
        same_phase = (
            np.arange(len(x)) % period == i % period
        )

        candidates = x[observed & same_phase]

        if len(candidates) > 0:
            result[i] = np.mean(candidates)
        else:
            result[i] = train_mean

    return result


IMPUTERS = {
    "LinearInterpolation": impute_linear,
    "ForwardFill": impute_forward_fill,
    "BackwardFill": impute_backward_fill,
    "Mean": impute_mean,
    "SeasonalMean": impute_seasonal_mean,
}


def calculate_metrics(trues, preds, masks):
    missing = masks.astype(bool)

    y_true = trues[missing]
    y_pred = preds[missing]

    if len(y_true) == 0:
        raise ValueError("No masked values found.")

    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    scale = np.mean(np.abs(np.diff(trues, axis=-1)))
    mase = mae / (scale + 1e-8)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MASE": float(mase),
    }


def evaluate_baseline(
    loader,
    dataset,
    stored_mask,
    mask_ratio,
    dataset_name,
):
    method_results = {
        name: {"trues": [], "preds": [], "masks": []}
        for name in IMPUTERS
    }


    method_results["Median"] = {
        "trues": [],
        "preds": [],
        "masks": [],
    }

    window_idx = 0
    period = SEASONAL_PERIODS[dataset_name]

    train_mean_scaled = float(
        dataset.scaler.transform([[dataset.train_mean]])[0, 0]
    )
    train_median_scaled = float(
        dataset.scaler.transform([[dataset.train_median]])[0, 0]
    )

    for batch_x, _ in tqdm(
        loader,
        desc=f"{dataset_name}, mask={mask_ratio}",
        leave=False,
    ):
        batch_size = batch_x.shape[0]

        batch_masks = np.stack([
            stored_mask[
                (window_idx + i) * SEQ_LEN:
                (window_idx + i + 1) * SEQ_LEN
            ]
            for i in range(batch_size)
        ])

        window_idx += batch_size

        batch_x = batch_x.numpy()[:, 0, :]

        for i in range(batch_size):
            x_scaled = batch_x[i].astype(np.float64)
            missing = batch_masks[i].astype(bool)

            x = dataset.scaler.inverse_transform(
                x_scaled.reshape(-1, 1)
            ).ravel()

            x[missing] = np.nan


            observed = ~missing

            for method_name in IMPUTERS:
                if method_name == "Mean":
                    imputed = x.copy()
                    imputed[missing] = dataset.train_mean

                elif method_name == "SeasonalMean":
                    imputed = impute_seasonal_mean(
                        x=np.nan_to_num(
                            x,
                            nan=dataset.train_mean,
                        ),
                        missing=missing,
                        train_mean=dataset.train_mean,
                        period=period,
                    )

                else:
                    safe_x = np.nan_to_num(
                        x,
                        nan=dataset.train_mean,
                    )

                    imputed = IMPUTERS[method_name](
                        x=safe_x,
                        missing=missing,
                        train_mean=dataset.train_mean,
                        period=period,
                    )

                method_results[method_name]["trues"].append(
                    dataset.scaler.inverse_transform(
                        batch_x[i].reshape(-1, 1)
                    ).ravel()
                )
                method_results[method_name]["preds"].append(
                    imputed
                )
                method_results[method_name]["masks"].append(
                    missing
                )

            median_imputed = x.copy()
            median_imputed[missing] = dataset.train_median

            method_results["Median"]["trues"].append(
                dataset.scaler.inverse_transform(
                    batch_x[i].reshape(-1, 1)
                ).ravel()
            )
            method_results["Median"]["preds"].append(
                median_imputed
            )
            method_results["Median"]["masks"].append(missing)

    if window_idx * SEQ_LEN > len(stored_mask):
        raise ValueError(
            f"{dataset_name}: mask array is too short."
        )

    results = []

    for method_name, values in method_results.items():
        trues = np.stack(values["trues"])[:, None, :]
        preds = np.stack(values["preds"])[:, None, :]
        masks = np.stack(values["masks"])[:, None, :]

        metrics = calculate_metrics(
            trues=trues,
            preds=preds,
            masks=masks,
        )

        results.append({
            "dataset": dataset_name,
            "method": method_name,
            "masking_ratio": mask_ratio,
            "n_windows": len(values["trues"]),
            **metrics,
        })

    return results




def main():
    all_results = []

    for dataset_name in DATASETS:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 60}")

        loader, dataset = get_test_loader(dataset_name)

        mask_path = (
            MASK_ROOT / dataset_name / "test_masks.npz"
        )

        with np.load(mask_path, allow_pickle=True) as mask_data:
            dataset_results = []

            for ratio in EVAL_MASK_RATIOS:
                mask_key = MASK_KEYS[ratio]
                stored_mask = np.asarray(
                    mask_data[mask_key]
                ).reshape(-1)


                required_length = len(dataset) * SEQ_LEN

                if len(stored_mask) < required_length:
                    raise ValueError(
                        f"{dataset_name}, {mask_key}: "
                        f"mask length {len(stored_mask)} is less "
                        f"than required length {required_length}."
                    )

                results = evaluate_baseline(
                    loader=loader,
                    dataset=dataset,
                    stored_mask=stored_mask,
                    mask_ratio=ratio,
                    dataset_name=dataset_name,
                )

                dataset_results.extend(results)
                all_results.extend(results)

        dataset_df = pd.DataFrame(dataset_results)

        dataset_df.to_csv(
            RESULT_ROOT
            / f"Classical_{dataset_name}_ALL_MASK_RATIOS.csv",
            index=False,
        )

        print(
            dataset_df[
                ["method", "masking_ratio", "MAE", "RMSE", "MASE"]
            ].to_string(index=False)
        )

        del loader, dataset
        gc.collect()

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        RESULT_ROOT / "Classical_Imputation_ALL_RESULTS.csv",
        index=False,
    )

    mean_results = (
        results_df
        .groupby(["dataset", "method"], as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            MASE=("MASE", "mean"),
        )
    )

    mean_results.to_csv(
        RESULT_ROOT / "Classical_Imputation_MEAN_MASK_RATIOS.csv",
        index=False,
    )

    overall_results = (
        mean_results
        .groupby("method", as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            MASE=("MASE", "mean"),
        )
        .sort_values("MAE")
    )

    overall_results.to_csv(
        RESULT_ROOT / "Classical_Imputation_OVERALL.csv",
        index=False,
    )

    print("\nOverall results, averaged across datasets:")
    print(overall_results.to_string(index=False))

    print(f"\nResults saved to: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
