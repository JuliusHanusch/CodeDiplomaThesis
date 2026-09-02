import json
import sqlite3
import subprocess
import sys
from pathlib import Path


DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/classification.db"

SEEDS = [42, 43, 44, 45, 46]


def get_db_connection():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=60,
    )
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


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


def run_train(idx, seed):
    subprocess.run([
        "python3",
        "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/finetune_classification.py",
        "--index", str(idx),
        "--seed", str(seed)
    ], check=True)


def run_eval(idx, seed):
    subprocess.run([
        "python3",
        "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/evaluate_calssification.py",
        "--index", str(idx),
        "--seed", str(seed)
    ], check=True)


def main():

    idx = int(sys.argv[1])

    conn = get_db_connection()
    cfg, h = load_config_by_idx(conn, idx)
    conn.close()

    if cfg is None:
        print(f"No config for idx {idx}")
        return

    for seed in SEEDS:

        print(f"[IDX {idx}] Starting seed {seed}")


        output_dir = Path(
            f"./FineTunedModels/Classification/Small/{h}/seed-{seed}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = output_dir / "checkpoint-final"

        model_path_column = f"model_path_{seed}"

        conn = get_db_connection()

        conn.execute(
            f"""
            UPDATE runs
            SET {model_path_column}=?
            WHERE id=?
            """,
            (str(model_path), idx)
        )
        conn.commit()
        conn.close()


        run_train(idx, seed)

        run_eval(idx, seed)

        print(f"[IDX {idx}] Finished seed {seed}")


    conn = get_db_connection()

    conn.execute(
        """
        UPDATE runs
        SET status='Finished'
        WHERE id=?
        """,
        (idx,)
    )
    conn.commit()
    conn.close()

    print(f"[IDX {idx}] All seeds finished")


if __name__ == "__main__":
    main()