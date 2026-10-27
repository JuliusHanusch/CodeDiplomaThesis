from pathlib import Path
import numpy as np


BASE_DIR = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/"
    "data/finetuning/Imputation"
)

DATASETS = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
]

#masking ratios like MOMENT
MASKING_RATIOS = [
    0.125,
    0.25,
    0.375,
    0.5,
]

PATCH_LEN = 8
SEED = 13

def generate_mask(
    series_length,
    patch_len,
    mask_ratio,
    rng,
):

    n_patches = series_length // patch_len

    mask = np.zeros(
        series_length,
        dtype=bool,
    )

    n_masked_patches = int(
        np.ceil(n_patches * mask_ratio)
    )

    selected_patches = rng.choice(
        n_patches,
        size=n_masked_patches,
        replace=False,
    )

    for patch_idx in selected_patches:

        start = patch_idx * patch_len
        end = start + patch_len

        mask[start:end] = True

    return mask



rng = np.random.default_rng(SEED)

for dataset_name in DATASETS:

    print(dataset_name)

    dataset_dir = BASE_DIR / dataset_name

    #use test or val files to create masks
    eval_path = dataset_dir / "test.npz"

    if not eval_path.exists():
        raise FileNotFoundError(eval_path)


    data = np.load(
        eval_path,
        allow_pickle=True,
    )

    test_series = data["series"]

    if len(test_series) != 1:
        raise ValueError(
            f"Expected one test series for {dataset_name}, "
            f"but found {len(test_series)}."
        )

    series = np.asarray(
        test_series[0],
        dtype=np.float32,
    )

    series_length = len(series)
    masks = {}

    for ratio in MASKING_RATIOS:

        masks[ratio] = generate_mask(
            series_length=series_length,
            patch_len=PATCH_LEN,
            mask_ratio=ratio,
            rng=rng,
        )

    mask_path = dataset_dir / "test_masks.npz"

    np.savez_compressed(
        mask_path,
        mask_0125=masks[0.125],
        mask_025=masks[0.25],
        mask_0375=masks[0.375],
        mask_050=masks[0.5],
    )

    print(
        f"Saved: {mask_path}"
    )