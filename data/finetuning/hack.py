import shutil
import ast
from pathlib import Path

BASE_DIR = Path("/content/CodeDiplomaThesis/data/finetuning/MSL")  # adjust if needed
TEST_DIR = BASE_DIR / "test"
LABEL_DIR = BASE_DIR / "test_labels"
BACKUP_DIR = BASE_DIR / "backup"

THRESHOLD = 100


def backup_data():
    print("Creating backup...")

    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)

    shutil.copytree(TEST_DIR, BACKUP_DIR / "test")
    shutil.copytree(LABEL_DIR, BACKUP_DIR / "test_labels")

    print("Backup completed.")


def parse_label_file(path: Path):
    with open(path, "r") as f:
        content = f.read().strip()

    if not content:
        return []

    try:
        return ast.literal_eval(content)
    except Exception as e:
        print(f"Parse error: {path} -> {e}")
        return None


def has_long_anomaly(intervals) -> bool:
    if not isinstance(intervals, list):
        return False

    for item in intervals:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = item
            if (end - start) > THRESHOLD:
                return True

    return False


def corresponding_test_file(label_file: Path) -> Path:
    # C-1.txt -> C-1.npy
    return TEST_DIR / (label_file.stem + ".npy")


def clean_dataset():
    removed = 0

    for label_file in LABEL_DIR.glob("*.txt"):
        intervals = parse_label_file(label_file)

        if intervals is None:
            continue

        if has_long_anomaly(intervals):
            test_file = corresponding_test_file(label_file)

            print(f"Removing {label_file.name} + {test_file.name}")

            label_file.unlink(missing_ok=True)

            if test_file.exists():
                test_file.unlink()
            else:
                print(f"Warning: missing {test_file}")

            removed += 1

    print(f"Done. Removed {removed} samples.")


if __name__ == "__main__":
    backup_data()
    clean_dataset()