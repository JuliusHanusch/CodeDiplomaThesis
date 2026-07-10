import sqlite3
import json
import subprocess
import sys
from pathlib import Path

DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/tser/tser.db"

def load_config_by_idx(conn, idx):
    cur = conn.cursor()

    cur.execute("""
        SELECT config, config_hash
        FROM runs
        WHERE id = ?
    """, (idx,))

    row = cur.fetchone()

    if row is None:
        return None, None

    cfg_json, h = row

    return json.loads(cfg_json), h


def run_train(cfg_path):
    subprocess.run([
        "python3",
        "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/train.py",
        "--config", cfg_path
    ], check=True)


def run_eval(idx):
    subprocess.run([
        "python3",
        "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/evaluate_tser.py",
        "--index", str(idx)  
    ], check=True)


def main():

    idx = int(sys.argv[1]) 
    conn = sqlite3.connect(DB_PATH)
    cfg, h = load_config_by_idx(conn, idx)

    if cfg is None:
        print(f"No config for idx {idx}")
        return

    output_dir = Path(f"./FineTunedModels/TSER/{h}")
    cfg["output_dir"] = str(output_dir)

    cfg_path = output_dir / "config.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    # save config temporarily for train.py
    import yaml
    tmp_yaml = output_dir / "train_config.yaml"

    with open(tmp_yaml, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print(f"[IDX {idx}] Running config {h}")

    run_train(str(tmp_yaml))

    model_path = output_dir / "run-0" / "checkpoint-final"


    conn.execute("""
        UPDATE runs
        SET model_path=?,
            status='TRAIN_DONE'
        WHERE id=?
    """, (str(model_path), idx))

    conn.commit()

    run_eval(idx)

    conn.execute("""
        UPDATE runs
        SET status='DONE'
        WHERE id=?
    """, (idx,))

    conn.commit()

if __name__ == "__main__":
    main()