import sqlite3
import json
import subprocess
import sys
from pathlib import Path
import time


DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/imputation/imputation_allData.db"


def get_db_connection():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=60,
    )
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def execute_db_update(sql, params, description="database update"):
    """
    Execute a SQLite write.
    Retry up to 5 times if the database is locked.
    """

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
        "-u",
        "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/imputation/finetune_imputation.py",
        "--index", str(idx),
    ], check=True)


def run_eval(idx):

    subprocess.run([
        "python3",
        "-u",
        "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/imputation/evaluate_imputation.py",
        "--index", str(idx),
    ], check=True)


def main():

    idx = int(sys.argv[1])

    conn = get_db_connection()

    cfg, h = load_config_by_idx(conn, idx)

    if cfg is None:
        conn.close()
        print(f"No config for idx {idx}")
        return

    print(f"[IDX {idx}] Running config {h}", flush=True)

    cur = conn.cursor()

    cur.execute("""
        SELECT MAE_avg
        FROM runs
        WHERE id = ?
    """, (idx,))

    row = cur.fetchone()

    conn.close()

    if row is None:
        print(f"No database row for idx {idx}")
        return

    mae_avg = row[0]

    if mae_avg is not None:
        print(
            f"[IDX {idx}] Already complete "
            f"(MAE_avg={mae_avg}) - SKIPPING",
            flush=True
        )
        return

    output_dir = Path(
        f"./FineTunedModels/Imputation/Small/{h}/"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = output_dir / "checkpoint-final"



    execute_db_update(
        """
        UPDATE runs
        SET model_path=?
        WHERE id=?
        """,
        (
            str(model_path),
            idx
        ),
        description="model path update "
                    
    )

    run_train(idx)
    run_eval(idx)

    print(f"[IDX {idx}] Finished")


    execute_db_update(
        """
        UPDATE runs
        SET status='DONE'
        WHERE id=?
        """,
        (idx,),
        description=f"final status update (IDX {idx})"
    )


if __name__ == "__main__":
    main()