import sys
import os
import time
from pathlib import Path
print("Start")
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir.resolve()))
sys.path.append(str((root_dir / "src").resolve()))
sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos_bolt import ChronosBoltPipeline

print("Finished Imports")
MODEL_PATH = "/content/CodeDiplomaThesis/BoltModel/run-1/checkpoint-final"
#MODEL_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/chronos_models/chronos_-6611716250758806883/run-0/checkpoint-final"

print("🚀 Loading pipeline...")
pipeline = ChronosBoltPipeline.from_pretrained(MODEL_PATH)
print("✅ Pipeline loaded")

model = pipeline.model.model

print("Starting upload to Hugging Face Hub...")
start_time = time.time()

model.push_to_hub("ChronosBERT-Classification")

elapsed = time.time() - start_time
print(f"Upload finished in {elapsed:.1f} seconds")