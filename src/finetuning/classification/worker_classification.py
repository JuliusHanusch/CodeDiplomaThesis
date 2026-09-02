import json
import sqlite3
import subprocess
import sys
from pathlib import Path
import time


DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/classification_allData.db"



def get_db_connection():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=60,
    )
    conn.execute("PRAGMA busy_timeout=60000")
    return conn

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


def run_train(idx):
    subprocess.run([
        "python3",
        "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/finetune_classification.py",
        "--index", str(idx),
    ], check=True)


def run_eval(idx):
    subprocess.run([
        "python3",
        "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/evaluate_calssification.py",
        "--index", str(idx),
    ], check=True)


def main():

    if len(sys.argv) != 2:
        print("Usage: python worker.py <idx>")
        sys.exit(1)

    idx = int(sys.argv[1])



    conn = get_db_connection()

    cfg, h = load_config_by_idx(conn, idx)

    cur = conn.cursor()
    cur.execute(
        """
        SELECT accuracy
        FROM runs
        WHERE id = ?
        """,
        (idx,)
    )
    row = cur.fetchone()
    conn.close()

    if row is not None and row[0] is not None:
        print(f"[IDX {idx}] Accuracy already exists ({row[0]}), skipping")
        return

    print(f"[IDX {idx}] Starting")
    print(f"[IDX {idx}] Config hash: {h}")

    output_dir = Path(
        f"./FineTunedModels/Classification/Small/"
        f"{h}/checkpoint-final"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = output_dir

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
            """
            SELECT model_path
            FROM runs
            WHERE id = ?
            """,
            (idx,)
    )

    existing_model_path = cur.fetchone()[0]

    conn.close()

    if existing_model_path is None:
        execute_db_update(
            """
            UPDATE runs
            SET model_path=?
            WHERE id=?
            """,
            (str(model_path), idx),
            description=f"setting model_path for idx {idx}"
        )

    print(f"[IDX {idx}] Model path: {model_path}")

    run_train(idx)

    run_eval(idx)


    conn = get_db_connection()

    conn.execute(
        """
        UPDATE runs
        SET status = 'Finished'
        WHERE id = ?
        """,
        (idx,)
    )

    conn.commit()
    conn.close()

    print(f"[IDX {idx}] Finished")


if __name__ == "__main__":
    main()