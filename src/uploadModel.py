import sys
import os
import time
from pathlib import Path
print("Start")
# root_dir = Path(__file__).parent.parent
# sys.path.append(str(root_dir.resolve()))
# sys.path.append(str((root_dir / "src").resolve()))
# sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))

# from chronos_pkg.src.chronos import ChronosBoltPipeline

#Colab Import
ROOT = "/content/CodeDiplomaThesis/"
sys.path.append(str(Path(ROOT).resolve()))
from chronos_pkg.src.chronos import ChronosBoltPipeline

from transformers import AutoConfig
import json
from pathlib import Path
MODEL_PATH = "/content/CodeDiplomaThesis/output/run-0/checkpoint-final"

config_path = Path(MODEL_PATH) / "config.json"

print(config_path.exists())

with open(config_path) as f:
    cfg = json.load(f)

print(cfg.get("model_type"))

config = AutoConfig.from_pretrained(MODEL_PATH)
print(config)

print("Finished Imports")

print("🚀 Loading pipeline...")
pipeline = ChronosBoltPipeline.from_pretrained(MODEL_PATH)
print("✅ Pipeline loaded")

model = pipeline.model.model

print("Starting upload to Hugging Face Hub...")
start_time = time.time()

model.push_to_hub("ChronosBERT-Classification")

elapsed = time.time() - start_time
print(f"Upload finished in {elapsed:.1f} seconds")