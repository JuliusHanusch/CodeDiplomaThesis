import json
import sys
from pathlib import Path

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

USED_FIELDS = {
    "num_train_epochs",
    "per_device_train_batch_size",
    "learning_rate",
    "dropout_head",
    "warmup_ratio",
    "TrainInnerModel",
    "num_labels",
    "output_dir",
}

with open(INPUT_PATH, "r") as f:
    config = json.load(f)

clean_config = {
    key: config[key]
    for key in USED_FIELDS
    if key in config
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(clean_config, f, indent=2)

print(f"Saved cleaned config to: {OUTPUT_PATH}")
print("\nRemoved fields:")
for key in config:
    if key not in USED_FIELDS:
        print(f"  - {key}")