import numpy as np
from pathlib import Path
from tqdm import tqdm
from gluonts.dataset.arrow import ArrowWriter

# =====================================================
# CONFIG
# =====================================================

DATA_ROOT = Path("data/anomaly_train_only")
CONTEXT_LENGTH = 512
STRIDE = 128
START_TIME = np.datetime64("2000-01-01", "s")

np.random.seed(42)

# =====================================================
# ANOMALY FUNCTIONS (YOUR EXACT LOGIC)
# =====================================================

def soft_replacement(x, pool):
    x = x.copy()

    L = np.random.randint(16, 64)
    s = np.random.randint(0, len(x) - L)

    ext_start = np.random.randint(0, len(pool) - L)
    ext = pool[ext_start:ext_start + L]

    alpha = np.random.uniform(0.2, 0.8)
    x[s:s+L] = alpha * x[s:s+L] + (1 - alpha) * ext

    return x


def uniform_replacement(x):
    x = x.copy()

    L = np.random.randint(16, 64)
    s = np.random.randint(0, len(x) - L)

    val = np.random.uniform(np.min(x), np.max(x))
    x[s:s+L] = val

    return x


def length_adjustment(x):
    x = x.copy()

    L = np.random.randint(32, 128)
    s = np.random.randint(0, len(x) - L)

    seg = x[s:s+L]
    factor = np.random.choice([0.5, 0.75, 1.5, 2.0])

    new_len = max(8, int(L * factor))

    warped = np.interp(
        np.linspace(0, L - 1, new_len),
        np.arange(L),
        seg,
    )

    warped = np.interp(
        np.linspace(0, new_len - 1, L),
        np.arange(new_len),
        warped,
    )

    x[s:s+L] = warped

    return x


def peak_noise(x):
    x = x.copy()

    idx = np.random.randint(0, len(x))
    scale = np.std(x) if np.std(x) > 0 else 1.0

    x[idx] += np.random.choice([-1, 1]) * np.random.uniform(3, 8) * scale

    return x


ANOMALIES = [
    soft_replacement,
    uniform_replacement,
    length_adjustment,
    peak_noise,
]


def inject(x, pool):
    fn = np.random.choice(ANOMALIES)

    if fn == soft_replacement:
        return fn(x, pool), 1, 1

    return fn(x), 1, ANOMALIES.index(fn) + 1

# =====================================================
# DATA LOADING
# =====================================================

def load_smap_msl(name):
    base = DATA_ROOT / name

    train = np.load(base / "train.npy").astype(np.float32)
    test = np.load(base / "test.npy").astype(np.float32)
    labels = np.load(base / "test_label.npy").astype(np.int32)

    return train.reshape(-1), test.reshape(-1), labels


def load_smd():
    base = DATA_ROOT / "ServerMachineDataset"

    train_files = sorted((base / "train").glob("machine-*.txt"))
    test_files = sorted((base / "test").glob("machine-*.txt"))
    label_files = sorted((base / "test_label").glob("machine-*.txt"))

    def load(files):
        series = []
        for f in files:
            x = np.loadtxt(f).astype(np.float32).reshape(-1)
            series.append(x)
        return np.concatenate(series)

    train = load(train_files)
    test = load(test_files)
    labels = load(label_files)

    return train, test, labels


def load_dataset(name):
    if name in ["SMAP", "MSL"]:
        return load_smap_msl(name)
    elif name == "SMD":
        return load_smd()
    else:
        raise ValueError(f"Unknown dataset {name}")

# =====================================================
# WINDOWING
# =====================================================

def make_windows(series):
    entries = []

    pool = series

    for i in range(0, len(series) - CONTEXT_LENGTH, STRIDE):

        w = series[i:i + CONTEXT_LENGTH]

        if len(w) < CONTEXT_LENGTH:
            continue

        label = 0
        anomaly_type = 0

        # 50% chance synthetic anomaly
        if np.random.rand() < 0.5:
            w, label, anomaly_type = inject(w, pool)

        entries.append({
            "start": START_TIME,
            "target": w.astype(np.float32),
            "label": label,
            "anomaly_type": anomaly_type,
        })

    return entries

# =====================================================
# ARROW WRITER
# =====================================================

def write_arrow(path, data):
    ArrowWriter(compression="lz4").write_to_file(data, path)

# =====================================================
# PROCESS DATASET
# =====================================================

def process(name):
    print(f"\nProcessing {name}")

    train, test, labels = load_dataset(name)

    train_entries = make_windows(train)

    out_path = DATA_ROOT / f"{name}_train.arrow"

    write_arrow(out_path, train_entries)

    print(f"{name}: train windows = {len(train_entries)}")
    print(f"{name}: test size = {len(test)} (not used for training)")

# =====================================================
# MAIN
# =====================================================

def main():
    for name in ["SMAP", "MSL", "SMD"]:
        process(name)

    print("\nDONE")

if __name__ == "__main__":
    main()