import numpy as np
from pathlib import Path
from tqdm import tqdm
from gluonts.dataset.arrow import ArrowWriter


BASE_DIR = Path("/content/DiplomaThesis/data/finetuning")

CONTEXT_LENGTH = 512
STRIDE = 128
START_TIME = np.datetime64("2000-01-01", "s")

np.random.seed(42)


def soft_replacement(x, pool):
    x = x.copy()
    L = np.random.randint(16, 64)
    s = np.random.randint(0, len(x) - L)

    ext_start = np.random.randint(0, len(pool) - L)
    ext = pool[ext_start:ext_start + L]

    alpha = np.random.uniform(0.2, 0.8)
    x[s:s+L] = alpha * x[s:s+L] + (1 - alpha) * ext

    return x, s, L


def uniform_replacement(x):
    x = x.copy()
    L = np.random.randint(16, 64)
    s = np.random.randint(0, len(x) - L)

    val = np.random.uniform(np.min(x), np.max(x))
    x[s:s+L] = val

    return x, s, L


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
    return x, s, L


def peak_noise(x):
    x = x.copy()
    idx = np.random.randint(0, len(x))
    scale = np.std(x) if np.std(x) > 0 else 1.0

    x[idx] += np.random.choice([-1, 1]) * np.random.uniform(3, 8) * scale

    return x, idx, 1


ANOMALIES = [
    soft_replacement,
    uniform_replacement,
    length_adjustment,
    peak_noise,
]


def inject(x, pool):
    fn = np.random.choice(ANOMALIES)
    return fn(x, pool) if fn == soft_replacement else fn(x)



def safe_load_smd_file(f):
    try:
        return np.loadtxt(f, delimiter=",").astype(np.float32)
    except:
        import pandas as pd
        return pd.read_csv(f, header=None).values.astype(np.float32)


def load_train_series(dataset_path: Path, dataset_name: str):
    series = []
    train_path = dataset_path / "train"

    if dataset_name == "SMD":
        files = sorted(train_path.glob("*.txt"))
        for f in files:
            arr = safe_load_smd_file(f)
            arr = np.asarray(arr)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            series.append(arr.astype(np.float32))
    else:
        files = sorted(train_path.glob("*.npy"))
        for f in files:
            arr = np.load(f).astype(np.float32).reshape(-1)
            series.append(arr)

    return series



def build_train(series_list):
    entries = []
    pool = np.concatenate(series_list)

    for s in tqdm(series_list, desc="windowing"):
        for i in range(0, len(s) - CONTEXT_LENGTH, STRIDE):

            w = s[i:i+CONTEXT_LENGTH].copy()
            mask = np.zeros(CONTEXT_LENGTH, dtype=np.float32)

            # 50% chance anomaly injection
            if np.random.rand() < 0.5:

                fn = np.random.choice(ANOMALIES)

                if fn == soft_replacement:
                    w, pos, L = soft_replacement(w, pool)

                elif fn == peak_noise:
                    w, pos, L = peak_noise(w)

                else:
                    w, pos, L = fn(w)

                # LABELING RULES
                if fn == peak_noise:
                    # OPTION 1: single timestep
                    mask[pos] = 1.0
                else:
                    start = pos
                    end = min(pos + L, CONTEXT_LENGTH)
                    mask[start:end] = 1.0

            entries.append({
                "start": START_TIME,
                "target": w.astype(np.float32),
                "anomaly_mask": mask
            })

    return entries


# =====================================================
# WRITE ARROW
# =====================================================

def write_arrow(path, data):
    ArrowWriter(compression="lz4").write_to_file(data, path)



def process(dataset_name):
    print(f"\n=== {dataset_name} ===")

    dataset_path = BASE_DIR / dataset_name

    series = load_train_series(dataset_path, dataset_name)
    data = build_train(series)

    out_file = dataset_path / f"{dataset_name}_train.arrow"

    write_arrow(out_file, data)

    print(f"{dataset_name}: {len(data)} windows → {out_file}")



def main():
    for ds in ["SMAP", "MSL", "SMD"]:
        process(ds)

    print("\nDONE")


if __name__ == "__main__":
    main()