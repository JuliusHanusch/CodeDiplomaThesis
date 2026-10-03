import sqlite3
import json
import subprocess
import sys
from pathlib import Path
import time


DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/imputation/final/imputation_defaultConfigs.db"
SEEDS = [42, 43, 44, 45, 46]


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


def run_train(idx, seed):

    subprocess.run([
        "python3",
        "-u",
        "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/imputation/final/final_finetune_imputation.py",
        "--index", str(idx),
        "--seed", str(seed)
    ], check=True)


def run_eval(idx, seed):

    subprocess.run([
        "python3",
        "-u",
        "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/imputation/final/final_evaluate_imputation.py",
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

    print(f"[IDX {idx}] Running config {h}", flush=True)

    for seed in SEEDS:

        print(
            f"[IDX {idx}] Starting seed {seed}",
            flush=True
        )

        output_dir = Path(
            f"/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/FineTunedModels/Imputation/final/{h}/seed-{seed}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = output_dir / "checkpoint-final"

        model_path_column = f"model_path_{seed}"


        execute_db_update(
            f"""
            UPDATE runs
            SET {model_path_column}=?
            WHERE id=?
            """,
            (
                str(model_path),
                idx
            ),
            description=f"model path update "
                        f"(IDX {idx}, seed {seed})"
        )



        run_train(idx, seed)
        run_eval(idx, seed)

        print(
            f"[IDX {idx}] Finished seed {seed}",
            flush=True
        )


    execute_db_update(
        """
        UPDATE runs
        SET status='DONE'
        WHERE id=?
        """,
        (idx,),
        description=f"final status update (IDX {idx})"
    )

    print(
        f"[IDX {idx}] All seeds finished",
        flush=True
    )


if __name__ == "__main__":
    main()