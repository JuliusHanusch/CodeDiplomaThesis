import sqlite3
import json
import hashlib
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

HPO_DB_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/src/finetuning/tser/tser_allData.db"
)

FINAL_DB_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/src/finetuning/tser/Final/tser_bestConfigs.db"
)

DATA_ROOT = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/data/finetuning/TSER"
)

SEEDS = [42, 43, 44, 45, 46]


# ============================================================
# HELPERS
# ============================================================

def config_hash(config, dataset):
    return hashlib.sha256(
        json.dumps(
            {
                "config": config,
                "dataset": dataset,
            },
            sort_keys=True
        ).encode()
    ).hexdigest()


# ============================================================
# CREATE FINAL DB
# ============================================================

def create_final_db():

    hpo_conn = sqlite3.connect(HPO_DB_PATH)
    hpo_cur = hpo_conn.cursor()

    final_conn = sqlite3.connect(FINAL_DB_PATH)

    final_conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            config_hash TEXT UNIQUE,
            config JSON,

            dataset TEXT,

            train_data TEXT,
            test_data TEXT,

            source_hpo_id INTEGER,

            -- HPO selection metrics
            hpo_mae REAL,
            hpo_rmse REAL,

            -- Final model paths
            model_path_42 TEXT,
            model_path_43 TEXT,
            model_path_44 TEXT,
            model_path_45 TEXT,
            model_path_46 TEXT,

            -- Final test metrics
            mae_42 REAL,
            mae_43 REAL,
            mae_44 REAL,
            mae_45 REAL,
            mae_46 REAL,

            rmse_42 REAL,
            rmse_43 REAL,
            rmse_44 REAL,
            rmse_45 REAL,
            rmse_46 REAL,

            -- Final averages
            mae_avg REAL,
            mae_std REAL,

            rmse_avg REAL,
            rmse_std REAL,

            status TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    final_conn.commit()

    # ========================================================
    # Get all datasets
    # ========================================================

    hpo_cur.execute("""
        SELECT DISTINCT dataset
        FROM runs
        ORDER BY dataset
    """)

    datasets = [
        row[0]
        for row in hpo_cur.fetchall()
    ]

    # ========================================================
    # Select best HPO configuration per dataset
    # ========================================================

    for dataset in datasets:

        hpo_cur.execute("""
            SELECT
                id,
                config,
                mae,
                rmse
            FROM runs
            WHERE dataset = ?
              AND mae IS NOT NULL
              AND rmse IS NOT NULL
            ORDER BY mae ASC
            LIMIT 1
        """, (dataset,))

        row = hpo_cur.fetchone()

        if row is None:
            print(
                f"[WARNING] No completed HPO result "
                f"for {dataset} - skipping"
            )
            continue

        hpo_id, config_json, hpo_mae, hpo_rmse = row

        config = json.loads(config_json)

        # ====================================================
        # Final training/test data
        # ====================================================

        dataset_dir = Path(DATA_ROOT) / dataset

        train_data = dataset_dir / "train_uni.arrow"
        test_data = dataset_dir / "test_uni.arrow"

        if not train_data.exists():
            print(
                f"[WARNING] Missing full train data for "
                f"{dataset}: {train_data}"
            )
            continue

        if not test_data.exists():
            print(
                f"[WARNING] Missing test data for "
                f"{dataset}: {test_data}"
            )
            continue

        # ====================================================
        # Hash
        # ====================================================

        h = config_hash(config, dataset)

        # ====================================================
        # Insert one row for the configuration
        # ====================================================

        final_conn.execute("""
            INSERT OR IGNORE INTO runs (
                config_hash,
                config,
                dataset,
                train_data,
                test_data,
                source_hpo_id,

                hpo_mae,
                hpo_rmse,

                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            h,
            json.dumps(config),
            dataset,
            str(train_data),
            str(test_data),
            hpo_id,

            hpo_mae,
            hpo_rmse,

            "PENDING"
        ))

        final_conn.commit()

        # ====================================================
        # Get final row ID
        # ====================================================

        cur = final_conn.cursor()

        cur.execute("""
            SELECT id
            FROM runs
            WHERE config_hash = ?
        """, (h,))

        final_id = cur.fetchone()[0]

        print(
            f"{dataset}: "
            f"Final ID={final_id}, "
            f"HPO ID={hpo_id}, "
            f"HPO MAE={hpo_mae:.6f}, "
            f"HPO RMSE={hpo_rmse:.6f}"
        )

    hpo_conn.close()
    final_conn.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    create_final_db()