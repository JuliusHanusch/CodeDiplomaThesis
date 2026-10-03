import sqlite3
import json
import hashlib
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

HPO_DB_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/classification_cv_new.db"
)

FINAL_DB_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/final/classification_cv_best.db"
)

DATA_ROOT = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/data/finetuning/UCR_extracted/UCRArchive_2018"
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

    Path(FINAL_DB_PATH).parent.mkdir(
        parents=True,
        exist_ok=True
    )

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

            -- HPO selection metric
            hpo_accuracy REAL,

            -- Final model paths
            model_path_42 TEXT,
            model_path_43 TEXT,
            model_path_44 TEXT,
            model_path_45 TEXT,
            model_path_46 TEXT,

            -- Final test metrics
            accuracy_42 REAL,
            accuracy_43 REAL,
            accuracy_44 REAL,
            accuracy_45 REAL,
            accuracy_46 REAL,

            f1_42 REAL,
            f1_43 REAL,
            f1_44 REAL,
            f1_45 REAL,
            f1_46 REAL,

            -- Final averages
            accuracy_avg REAL,
            accuracy_std REAL,

            f1_avg REAL,
            f1_std REAL,

            -- Inference time per seed
            inference_time_42 REAL,
            inference_time_43 REAL,
            inference_time_44 REAL,
            inference_time_45 REAL,
            inference_time_46 REAL,

            -- Inference time averages
            inference_time_avg REAL,
            inference_time_std REAL,

            status TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    final_conn.commit()



    hpo_cur.execute("""
        SELECT DISTINCT dataset
        FROM runs
        ORDER BY dataset
    """)

    datasets = [
        row[0]
        for row in hpo_cur.fetchall()
    ]


    for dataset in datasets:

        hpo_cur.execute("""
            SELECT
                id,
                config,
                accuracy
            FROM runs
            WHERE dataset = ?
              AND accuracy IS NOT NULL
            ORDER BY accuracy DESC
            LIMIT 1
        """, (dataset,))

        row = hpo_cur.fetchone()

        if row is None:
            print(
                f"[WARNING] No completed HPO result "
                f"for {dataset} - skipping"
            )
            continue

        hpo_id, config_json, hpo_accuracy = row

        config = json.loads(config_json)

        # ====================================================
        # Final training/test data
        # ====================================================

        dataset_dir = Path(DATA_ROOT) / dataset

        train_data = dataset_dir / f"{dataset}_TRAIN.tsv"
        test_data = dataset_dir / f"{dataset}_TEST.tsv"

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

                hpo_accuracy,

                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            h,
            json.dumps(config),
            dataset,
            str(train_data),
            str(test_data),
            hpo_id,

            hpo_accuracy,

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
            f"HPO accuracy={hpo_accuracy:.6f}"
        )

    hpo_conn.close()
    final_conn.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    create_final_db()