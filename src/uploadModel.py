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

MODEL_PATH = "/content/CodeDiplomaThesis/output/run-12/checkpoint-final"

print("Finished Imports")

print("🚀 Loading pipeline...")
pipeline = ChronosBoltPipeline.from_pretrained(MODEL_PATH)
print("✅ Pipeline loaded")

model = pipeline.model

print("Starting upload to Hugging Face Hub...")
start_time = time.time()

model.push_to_hub("ChronosBoltBERT-20K")

elapsed = time.time() - start_time
print(f"Upload finished in {elapsed:.1f} seconds")