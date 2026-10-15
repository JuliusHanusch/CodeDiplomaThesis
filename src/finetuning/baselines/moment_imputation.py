from momentfm.utils.utils import control_randomness
from momentfm import MOMENTPipeline
from momentfm.utils.masking import Masking

import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader

import random
import gc
from pathlib import Path
from sklearn.preprocessing import StandardScaler



control_randomness(seed=13)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MASK_ROOT = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/data/finetuning/Imputation"
)

DATA_ROOT = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/data/finetuning/Imputation/raw"
)

RESULT_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/Results/Finetuning/Imputation"
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


DATASETS = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2"
]

SEQ_LEN = 512
N_CONFIGS = 100
SEED = 13

# Evaluation masking ratios
EVAL_MASK_RATIOS = [0.125,0.25,0.375,0.5]

BATCH_SIZES = [8,16,32,64,128]

EPOCH_OPTIONS = [2,5,10,20,40]

MASKING_RATIOS = [0.15,0.3,0.45]

MASK_PATCH_LEN_OPTIONS = [8, 16, 32]

LR_MIN = 5e-6
LR_MAX = 1e-3


MAX_GPU_BATCH_SIZE = 32


def sample_config():

    sampled_batch = int(random.choice(BATCH_SIZES))
    num_train_epochs = int(random.choice(EPOCH_OPTIONS))
    learning_rate = float(
        np.exp(
            np.random.uniform(
                np.log(LR_MIN),
                np.log(LR_MAX)
            )
        )
    )
    masking_ratio = float(random.choice(MASKING_RATIOS))
    mask_len = int(random.choice(MASK_PATCH_LEN_OPTIONS))

    return {
        "batch_size": sampled_batch,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "masking_ratio": masking_ratio,
        "mask_len": mask_len

    }


class InformerDataset:
    def __init__(
        self,
        data,
        forecast_horizon=192,
        data_split="train",
        data_stride_len=1,
        task_name="forecasting",
        random_seed=42,
    ):

        self.seq_len = 512
        self.forecast_horizon = forecast_horizon

        self.full_file_path_and_name = data
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.random_seed = random_seed

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

        train = slice(0,train_end)

        val = slice(train_end, val_end)

        test = slice(test_start,test_end)

        return train, val, test


    def _read_data(self):
        self.scaler = StandardScaler()

        df = pd.read_csv(self.full_file_path_and_name)

        self.length_timeseries_original = df.shape[0]

        df.drop(columns=["date"], inplace=True)

        df = df[["OT"]]

        df = df.infer_objects(copy=False).interpolate(method="cubic")

        data_splits = self._get_borders()

        train_data = df[data_splits[0]]

        self.scaler.fit(train_data.values)

        df = self.scaler.transform(df.values)

        if self.data_split == "train":
            self.data = df[data_splits[0], :]
        elif self.data_split == "val":
            self.data = df[data_splits[1], :]
        elif self.data_split == "test":
            self.data = df[data_splits[2], :]

        self.length_timeseries = self.data.shape[0]

        self.n_channels = self.data.shape[1]


    def __getitem__(self, index):

        seq_start = (
            self.data_stride_len * index
        )

        seq_end = (
            seq_start + self.seq_len
        )

        input_mask = np.ones(
            self.seq_len
        )

        if self.task_name == "imputation":

            if seq_end > self.length_timeseries:

                seq_end = (
                    self.length_timeseries
                    - self.seq_len
                )

            timeseries = self.data[
                seq_start:seq_end,
                :
            ].T

            return (
                timeseries,
                input_mask
            )

        raise NotImplementedError


    def __len__(self):

        if self.task_name == "imputation":

            return (
                self.length_timeseries
                - self.seq_len
            ) // self.data_stride_len + 1

        raise NotImplementedError



def get_dataloaders(
    dataset_name,
    batch_size
):

    data = (
        f"{DATA_ROOT}/{dataset_name}.csv"
    )

    train_dataset = InformerDataset(
        data=data,
        data_split="train",
        data_stride_len=512,
        task_name="imputation"
    )

    val_dataset = InformerDataset(
        data=data,
        data_split="val",
        data_stride_len=512,
        task_name="imputation"
    )

    test_dataset = InformerDataset(
        data=data,
        data_split="test",
        data_stride_len=512,
        task_name="imputation"
    )

    gpu_batch_size = min(
        batch_size,
        MAX_GPU_BATCH_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=gpu_batch_size,
        shuffle=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=gpu_batch_size,
        shuffle=False,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=gpu_batch_size,
        shuffle=False,
        drop_last=False
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        test_dataset.scaler
    )


def load_model():

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-base",
        model_kwargs={
            "task_name": "reconstruction",
        }
    )

    model.init()
    model.to(DEVICE)
    model.float()
    return model


def train_epoch(
    model,
    loader,
    optimizer,
    masking_ratio,
    mask_len,
    accumulation_steps
):

    model.train()

    criterion = torch.nn.MSELoss(reduction="none")

    mask_generator = Masking(
        mask_ratio=masking_ratio,
        patch_len = mask_len
    )  # Mask patches randomly

    losses = []

    optimizer.zero_grad()

    for step, (
        batch_x,
        batch_masks
    ) in enumerate(
        tqdm(
            loader,
            desc="Training",
            leave=False
        )
    ):

        original = batch_x.clone()

        n_channels = batch_x.shape[1]

        batch_x = batch_x.reshape(
            (-1, 1, 512)
        )

        original = original.reshape(
            (-1, 1, 512)
        ).to(DEVICE).float()

        batch_x = batch_x.to(DEVICE).float()

        batch_masks = batch_masks.to(DEVICE).long()
        batch_masks = batch_masks.repeat_interleave(
            n_channels,
            axis=0
        )

        mask = mask_generator.generate_mask(
            x=batch_x,
            input_mask=batch_masks
        ).to(DEVICE).long()

        output = model(
            x_enc=batch_x,
            input_mask=batch_masks,
            mask=mask
        )

        recon_loss = criterion(
            output.reconstruction,
            original
        )

        observed_mask = batch_masks * (1 - mask)
        masked_loss = observed_mask * recon_loss

        loss = masked_loss.nansum() / (
            observed_mask.nansum() + 1e-7
        )
        loss = loss / accumulation_steps

        loss.backward()

        if (
            (step + 1) % accumulation_steps == 0
            or step == len(loader) - 1
        ):
            optimizer.step()
            optimizer.zero_grad()

        losses.append(
            loss.item() * accumulation_steps
        )

    return np.mean(losses)


@torch.no_grad()
def evaluate(model, loader, mask_data, mask_ratio, scaler):
    model.eval()

    mask_key = {
        0.125: "mask_0125",
        0.25: "mask_025",
        0.375: "mask_0375",
        0.5: "mask_050",
    }[mask_ratio]

    stored_mask = mask_data[mask_key]

    trues = []
    preds = []
    masks = []

    window_idx = 0

    for batch_x, batch_masks in tqdm(
        loader,
        desc=f"Evaluation mask={mask_ratio}",
        leave=False
    ):
        batch_size = batch_x.shape[0]
        n_channels = batch_x.shape[1]

        batch_mask = np.stack([
            stored_mask[
                (window_idx + i) * SEQ_LEN:
                (window_idx + i + 1) * SEQ_LEN
            ]
            for i in range(batch_size)
        ])

        window_idx += batch_size

        trues.append(batch_x.numpy())

        batch_x = batch_x.to(DEVICE).float()
        batch_x = batch_x.reshape(-1, 1, SEQ_LEN)

        batch_mask = torch.tensor(
            batch_mask,
            dtype=torch.long,
            device=DEVICE
        )

        batch_mask = batch_mask.repeat_interleave(
            n_channels,
            dim=0
        )

        input_mask = batch_masks.to(DEVICE).long()

        input_mask = input_mask.repeat_interleave(
            n_channels,
            dim=0
        )

        moment_mask = 1 - batch_mask

        output = model(
            x_enc=batch_x,
            input_mask=input_mask,
            mask=moment_mask
        )

        reconstruction = output.reconstruction.cpu().numpy()

        reconstruction = reconstruction.reshape(
            -1,
            n_channels,
            SEQ_LEN
        )

        batch_mask = batch_mask.cpu().numpy()

        batch_mask = batch_mask.reshape(
            -1,
            n_channels,
            SEQ_LEN
        )

        preds.append(reconstruction)
        masks.append(batch_mask)

    trues = np.concatenate(trues, axis=0)
    preds = np.concatenate(preds, axis=0)
    masks = np.concatenate(masks, axis=0)

    trues = np.transpose(trues, (0, 2, 1))
    preds = np.transpose(preds, (0, 2, 1))

    shape = trues.shape

    trues = scaler.inverse_transform(
        trues.reshape(-1, shape[-1])
    ).reshape(shape)

    preds = scaler.inverse_transform(
        preds.reshape(-1, shape[-1])
    ).reshape(shape)

    trues = np.transpose(trues, (0, 2, 1))
    preds = np.transpose(preds, (0, 2, 1))

    return trues, preds, masks


def calculate_metrics(trues, preds, masks):
    missing = masks == 1
    y_true = trues[missing]
    y_pred = preds[missing]
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    return {
        "MAE": mae,
        "RMSE": rmse,
    }



def generate_configs(
    n_configs=50
):

    configs = []

    seen = set()

    while len(configs) < n_configs:

        config = sample_config()

        key = (
            config["batch_size"],
            config["num_train_epochs"],
            round(
                config["learning_rate"],
                12
            ),
            config["masking_ratio"],
            config["mask_len"]

        )

        if key in seen:
            continue

        seen.add(key)

        config["config_id"] = (
            len(configs) + 1
        )

        configs.append(
            config
        )

    return configs

all_results = []

for dataset_name in DATASETS:

    configs = generate_configs(
        N_CONFIGS
    )

    dataset_results = []

    best_mae = float("inf")
    best_model_state = None
    best_config = None

    for config in configs:
        config_id = config["config_id"]
        batch_size = config["batch_size"]
        epochs = config["num_train_epochs"]
        learning_rate = config["learning_rate"]
        masking_ratio = config["masking_ratio"]
        mask_len = config["mask_len"]


        train_loader, val_loader, test_loader, scaler = (
            get_dataloaders(
                dataset_name,
                batch_size
            )
        )

        accumulation_steps = max(1,batch_size // MAX_GPU_BATCH_SIZE)

        print(
            f"GPU batch size: "
            f"{min(batch_size, MAX_GPU_BATCH_SIZE)}"
        )

        print(
            f"Gradient accumulation: "
            f"{accumulation_steps}"
        )

        model = load_model()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

        training_losses = []

        for epoch in range(epochs):

            loss = train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                masking_ratio=masking_ratio,
                mask_len=mask_len,
                accumulation_steps=accumulation_steps
            )

            training_losses.append(
                loss
            )

            print(
                f"Epoch "
                f"{epoch + 1}/{epochs} | "
                f"loss={loss:.6f}"
            )


        val_mask_path = (
            f"{MASK_ROOT}/"
            f"{dataset_name}/"
            f"val_masks.npz"
        )

        val_mask_data = np.load(
            val_mask_path,
            allow_pickle=True
        )

        test_mask_path = (
            f"{MASK_ROOT}/"
            f"{dataset_name}/"
            f"test_masks.npz"
        )
        
        test_mask_data = np.load(
            test_mask_path,
            allow_pickle=True
        )


        val_results = {}
        test_results = {}

        for eval_ratio in EVAL_MASK_RATIOS:

            val_trues, val_preds, val_masks = evaluate(
                model=model,
                loader=val_loader,
                mask_data=val_mask_data,
                mask_ratio=eval_ratio,
                scaler=scaler
            )

            val_metrics = calculate_metrics(
                val_trues,
                val_preds,
                val_masks
            )

            val_results[eval_ratio] = val_metrics


            test_trues, test_preds, test_masks = evaluate(
                model=model,
                loader=test_loader,
                mask_data=test_mask_data,
                mask_ratio=eval_ratio,
                scaler=scaler
            )

            test_metrics = calculate_metrics(
                test_trues,
                test_preds,
                test_masks
            )

            test_results[eval_ratio] = test_metrics


        val_mean_mae = np.mean([
            val_results[r]["MAE"]
            for r in EVAL_MASK_RATIOS
        ])

        val_mean_rmse = np.mean([
            val_results[r]["RMSE"]
            for r in EVAL_MASK_RATIOS
        ])


        test_mean_mae = np.mean([
            test_results[r]["MAE"]
            for r in EVAL_MASK_RATIOS
        ])

        test_mean_rmse = np.mean([
            test_results[r]["RMSE"]
            for r in EVAL_MASK_RATIOS
        ])


        result = {
            "dataset": dataset_name,
            "config_id": config_id,

            "batch_size": batch_size,
            "gpu_batch_size": min(
                batch_size,
                MAX_GPU_BATCH_SIZE
            ),
            "gradient_accumulation_steps": accumulation_steps,

            "num_train_epochs": epochs,
            "learning_rate": learning_rate,
            "masking_ratio": masking_ratio,
            "mask_len": mask_len,

            "train_loss": training_losses[-1],

            "val_MAE_mean": val_mean_mae,
            "val_RMSE_mean": val_mean_rmse,

            "test_MAE_mean": test_mean_mae,
            "test_RMSE_mean": test_mean_rmse,
        }

        for ratio in EVAL_MASK_RATIOS:

            result[f"val_MAE_{ratio}"] = (
                val_results[ratio]["MAE"]
            )
            result[f"val_RMSE_{ratio}"] = (
                val_results[ratio]["RMSE"]
            )
            result[f"val_MASE_{ratio}"] = (
                val_results[ratio]["MASE"]
            )

            result[f"test_MAE_{ratio}"] = (
                test_results[ratio]["MAE"]
            )
            result[f"test_RMSE_{ratio}"] = (
                test_results[ratio]["RMSE"]
            )
            result[f"test_MASE_{ratio}"] = (
                test_results[ratio]["MASE"]
            )

        dataset_results.append(result)
        all_results.append(result)

        current_df = pd.DataFrame(
            dataset_results
        )

        current_df = current_df.sort_values(
            "val_MAE_mean"
        )

        current_df.to_csv(
            RESULT_ROOT /
            f"MOMENT_HPO_{dataset_name}_{N_CONFIGS}configs.csv",
            index=False
        )


        best = current_df.iloc[0]

        del model
        del optimizer
        del train_loader
        del val_loader
        del test_loader
        del val_mask_data
        del test_mask_data

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    dataset_df = pd.DataFrame(
        dataset_results
    )

    dataset_df = dataset_df.sort_values(
        "val_MAE_mean"
    )

    best_config = dataset_df.iloc[0]

    best_config_df = pd.DataFrame(
        [best_config]
    )

    best_config_df.to_csv(
        RESULT_ROOT /
        f"MOMENT_HPO_{dataset_name}_BEST.csv",
        index=False
    )


all_results_df = pd.DataFrame(
    all_results
)

all_results_df = all_results_df.sort_values(
    [
        "dataset",
        "val_MAE_mean"
    ]
)

all_results_df.to_csv(
    RESULT_ROOT /
    f"MOMENT_HPO_ALL_{N_CONFIGS}_PER_DATASET.csv",
    index=False
)

best_per_dataset = (
    all_results_df
    .sort_values("val_MAE_mean")
    .groupby("dataset")
    .first()
    .reset_index()
)

best_per_dataset.to_csv(
    RESULT_ROOT /
    "MOMENT_HPO_BEST_PER_DATASET.csv",
    index=False
)
