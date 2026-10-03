import json
import sqlite3
import subprocess
import sys
from pathlib import Path
import time


DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/tser/Final/tser_multivariate.db"
SEEDS = [42,43,44,45,46]

def execute_db_update(sql, params, description="database update"):
    for attempt in range(5):

        conn = None

        try:
            conn = get_db_connection()

            conn.execute(sql, params)
            conn.commit()
            conn.close()

            return

        except sqlite3.OperationalError as e:

            if conn is not None:
                conn.close()

            if "database is locked" not in str(e):
                raise

            print(
                f"[SQLite] Database locked during {description} "
                f"- retry {attempt + 1}/5",
                flush=True
            )

            if attempt < 4:
                time.sleep(5)

    raise RuntimeError(
        f"[SQLite] Database remained locked during "
        f"{description} after 5 attempts."
    )


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
        "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/tser/Final/final_training_tser.py",
        "--index", str(idx),
        "--seed", str(seed),
    ], check=True)


def run_eval(idx, seed):
    subprocess.run([
        "python3",
        "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/tser/Final/final_evaluation_tser.py",
        "--index", str(idx),
        "--seed", str(seed),
    ], check=True)

def run_multivariate(idx, seed):
    subprocess.run([
        "python3",
        "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/tser/Final/multivariate_tser.py",
        "--index", str(idx),
        "--seed", str(seed),
    ], check=True)


def main():

    idx = int(sys.argv[1])

    conn = get_db_connection()

    # cfg, h = load_config_by_idx(conn, idx)

    # conn.close()


    cur = conn.cursor()

    for seed in SEEDS:

        mae_column = f"mae_{seed}"

        cur.execute(
            f"""
            SELECT {mae_column}
            FROM runs
            WHERE id = ?
            """,
            (idx,),
        )

        row = cur.fetchone()

        if row is not None and row[0] is not None:
            print(f"Skipping idx={idx}, seed={seed}: {mae_column} already exists.")
            continue

        print(f"Running idx={idx}, seed={seed}...")

        # model_path = Path(
        #     f"/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/FineTunedModels/TSER/Small/Final/{h}/seed-{seed}/checkpoint-final"
        # )

        # model_path.parent.mkdir(
        #     parents=True,
        #     exist_ok=True
        # )

        # execute_db_update(
        #     f"""
        #     UPDATE runs
        #     SET model_path_{seed}=?
        #     WHERE id=?
        #     """,
        #     (str(model_path), idx),
        #     description=f"setting model_path_{seed} for idx {idx}"
        # )

        # run_train(idx, seed)

        #run_eval(idx, seed)

        run_multivariate(idx, seed)

    print(f"[IDX {idx}] Finished")


if __name__ == "__main__":
    main()