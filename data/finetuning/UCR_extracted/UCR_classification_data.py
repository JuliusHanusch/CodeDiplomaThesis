import zipfile
from pathlib import Path
import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from tqdm import tqdm
from datetime import datetime


BASE = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning")

ARCHIVE_PATH = BASE / "UCRArchive.zip"
EXTRACT_ROOT = BASE / "UCR_extracted"
DATA_ROOT = EXTRACT_ROOT / "UCRArchive_2018"

PASSWORD = b"someone"

#unzip files
def extract_zip():
    if DATA_ROOT.exists():
        print("Already extracted.")
        return

    print("Extracting...")

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as z:
        z.extractall(EXTRACT_ROOT, pwd=PASSWORD)


def main():
    extract_zip()

if __name__ == "__main__":
    main()