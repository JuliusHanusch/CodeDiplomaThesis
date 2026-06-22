from datasets import get_dataset_config_names, load_dataset
from gluonts.dataset.arrow import ArrowWriter
import numpy as np
import os


REPO = "foxy-steve/monash_uea_ucr_tser"
#OUTPUT_DIR = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/tser"
#Colab
OUTPUT_DIR = "/content/CodeDiplomaThesis/data/finetuning/tser"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_example(ex, idx):
    ts = np.asarray(ex["timeseries"], dtype=np.float32)

    # -----------------------------
    # HARD FIX: force flatten safely
    # -----------------------------
    ts = np.array(ts)

    # remove singleton dimensions
    ts = np.squeeze(ts)

    print("ts.ndim",ts.ndim)

    # if still 2D (rare but possible), flatten explicitly
    if ts.ndim > 1:
        ts = ts.reshape(-1)

    print("ts.ndim",ts.ndim)


    # final safety check (VERY IMPORTANT for GluonTS)
    assert ts.ndim == 1, f"Bad TSER shape: {ts.shape}"

    return {
        "start": ex["start"],
        "target": ts.astype(np.float32),   # 🔥 enforce dtype + shape
        "item_id": str(ex.get("item_id", idx)),
        "feat_static_cat": ex.get("feat_static_cat", None),
        "to_predict": float(ex["to_predict"]),
    }


def process_dataset(name):
    print(f"\n=== Processing {name} ===")

    ds = load_dataset(REPO, name)["train"]

    records = []
    for i, ex in enumerate(ds):
        records.append(convert_example(ex, i))

    output_path = os.path.join(OUTPUT_DIR, f"{name}_train.arrow")

    ArrowWriter(compression="lz4").write_to_file(
        dataset=records,
        path=output_path,
    )

    print(f"Saved: {output_path} ({len(records)} samples)")


def main():
    configs = get_dataset_config_names(REPO)

    print(f"Found {len(configs)} datasets")

    for name in configs:
        try:
            process_dataset(name)
        except Exception as e:
            print(f"FAILED {name}: {e}")


if __name__ == "__main__":
    main()