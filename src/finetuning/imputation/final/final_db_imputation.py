import sqlite3
import json
import hashlib
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

HPO_DB_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/src/finetuning/imputation/imputation_allData_new.db"
)

FINAL_DB_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/src/finetuning/imputation/final/"
    "imputation_bestConfigs_new.db"
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
# CREATE TABLE SCHEMA
# ============================================================

def create_table(final_conn):

    columns = [
        # ====================================================
        # Basic information
        # ====================================================

        "id INTEGER PRIMARY KEY AUTOINCREMENT",

        "config_hash TEXT UNIQUE",
        "config JSON",

        "dataset TEXT",

        "train_data TEXT",
        "test_data TEXT",

        "source_hpo_id INTEGER",

        # ====================================================
        # HPO selection metrics
        # ====================================================

        "hpo_MAE_avg REAL",
        "hpo_MASE_avg REAL",
        "hpo_MAE_Lin_avg REAL",
        "hpo_MASE_Lin_avg REAL",
    ]

    # ========================================================
    # Per-seed columns
    # ========================================================

    for seed in SEEDS:

        columns.extend([

            # ------------------------------------------------
            # Model path
            # ------------------------------------------------

            f"model_path_{seed} TEXT",

            # ------------------------------------------------
            # 12.5% masking
            # ------------------------------------------------

            f"MAE_0125_{seed} REAL",
            f"MASE_0125_{seed} REAL",
            f"MAE_Lin_0125_{seed} REAL",
            f"MASE_Lin_0125_{seed} REAL",

            # ------------------------------------------------
            # 25% masking
            # ------------------------------------------------

            f"MAE_025_{seed} REAL",
            f"MASE_025_{seed} REAL",
            f"MAE_Lin_025_{seed} REAL",
            f"MASE_Lin_025_{seed} REAL",

            # ------------------------------------------------
            # 37.5% masking
            # ------------------------------------------------

            f"MAE_0375_{seed} REAL",
            f"MASE_0375_{seed} REAL",
            f"MAE_Lin_0375_{seed} REAL",
            f"MASE_Lin_0375_{seed} REAL",

            # ------------------------------------------------
            # 50% masking
            # ------------------------------------------------

            f"MAE_050_{seed} REAL",
            f"MASE_050_{seed} REAL",
            f"MAE_Lin_050_{seed} REAL",
            f"MASE_Lin_050_{seed} REAL",

            # ------------------------------------------------
            # Average across the four masking ratios
            # for this seed
            # ------------------------------------------------

            f"MAE_avg_{seed} REAL",
            f"MASE_avg_{seed} REAL",
            f"MAE_Lin_avg_{seed} REAL",
            f"MASE_Lin_avg_{seed} REAL",
        ])

    # ========================================================
    # Overall statistics across all five seeds
    # ========================================================

    columns.extend([

        # ----------------------------------------------------
        # MAE
        # ----------------------------------------------------

        "MAE_avg REAL",
        "MAE_std REAL",

        # ----------------------------------------------------
        # MASE
        # ----------------------------------------------------

        "MASE_avg REAL",
        "MASE_std REAL",

        # ----------------------------------------------------
        # Linear interpolation MAE
        # ----------------------------------------------------

        "MAE_Lin_avg REAL",
        "MAE_Lin_std REAL",

        # ----------------------------------------------------
        # Linear interpolation MASE
        # ----------------------------------------------------

        "MASE_Lin_avg REAL",
        "MASE_Lin_std REAL",

        # ====================================================
        # Inference time
        # ====================================================

        "inference_time_42 REAL",
        "inference_time_43 REAL",
        "inference_time_44 REAL",
        "inference_time_45 REAL",
        "inference_time_46 REAL",

        "inference_time_avg REAL",
        "inference_time_std REAL",

        # ====================================================
        # Status
        # ====================================================

        "status TEXT",

        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    ])

    # ========================================================
    # CREATE TABLE
    # ========================================================

    create_sql = f"""
        CREATE TABLE IF NOT EXISTS runs (
            {", ".join(columns)}
        )
    """

    final_conn.execute(create_sql)
    final_conn.commit()


# ============================================================
# CREATE FINAL DB
# ============================================================

def create_final_db():

    # ========================================================
    # CONNECT TO HPO DB
    # ========================================================

    hpo_conn = sqlite3.connect(HPO_DB_PATH)
    hpo_cur = hpo_conn.cursor()

    # ========================================================
    # PREPARE FINAL DB DIRECTORY
    # ========================================================

    Path(FINAL_DB_PATH).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # CONNECT TO FINAL DB
    # ========================================================

    final_conn = sqlite3.connect(FINAL_DB_PATH)

    final_conn.execute("PRAGMA journal_mode=WAL")
    final_conn.execute("PRAGMA busy_timeout=30000")

    # ========================================================
    # CREATE TABLE
    # ========================================================

    create_table(final_conn)

    # ========================================================
    # GET DATASETS
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

    print(f"Found {len(datasets)} datasets.")

    # ========================================================
    # SELECT BEST HPO CONFIGURATION PER DATASET
    # ========================================================

    for dataset in datasets:

        print()
        print("=" * 70)
        print(f"Dataset: {dataset}")
        print("=" * 70)

        # ----------------------------------------------------
        # We only take train_data from the HPO DB to determine
        # the dataset directory.
        #
        # We explicitly use:
        #
        #     train.npz
        #     test.npz
        #
        # for the final experiment.
        # ----------------------------------------------------

        hpo_cur.execute("""
            SELECT
                id,
                config,
                train_data,
                MAE_avg,
                MASE_avg,
                MAE_Lin_avg,
                MASE_Lin_avg

            FROM runs

            WHERE dataset = ?
              AND MAE_avg IS NOT NULL

            ORDER BY MAE_avg ASC

            LIMIT 1
        """, (dataset,))

        row = hpo_cur.fetchone()

        if row is None:
            print(
                f"[WARNING] No completed HPO result "
                f"for {dataset} - skipping"
            )
            continue

        # ----------------------------------------------------
        # Exactly 7 columns selected above
        # ----------------------------------------------------

        (
            hpo_id,
            config_json,
            hpo_train_data,
            hpo_mae,
            hpo_mase,
            hpo_mae_lin,
            hpo_mase_lin
        ) = row

        # ====================================================
        # LOAD CONFIG
        # ====================================================

        config = json.loads(config_json)

        # ====================================================
        # DETERMINE FINAL DATA PATHS
        # ====================================================

        data_dir = Path(hpo_train_data).parent

        train_data = str(data_dir / "train.npz")
        test_data = str(data_dir / "test.npz")

        # ====================================================
        # CHECK FILES
        # ====================================================

        if not Path(train_data).exists():
            print(
                f"[WARNING] train.npz does not exist:\n"
                f"          {train_data}"
            )

        if not Path(test_data).exists():
            print(
                f"[WARNING] test.npz does not exist:\n"
                f"          {test_data}"
            )

        # ====================================================
        # CONFIG HASH
        # ====================================================

        h = config_hash(config, dataset)

        # ====================================================
        # INSERT FINAL CONFIGURATION
        # ====================================================

        final_conn.execute("""
            INSERT OR IGNORE INTO runs (

                config_hash,
                config,

                dataset,

                train_data,
                test_data,

                source_hpo_id,

                hpo_MAE_avg,
                hpo_MASE_avg,
                hpo_MAE_Lin_avg,
                hpo_MASE_Lin_avg,

                status

            )

            VALUES (
                ?, ?, ?, ?, ?,
                ?,
                ?, ?, ?, ?,
                ?
            )
        """, (
            h,
            json.dumps(config),

            dataset,

            train_data,
            test_data,

            hpo_id,

            hpo_mae,
            hpo_mase,
            hpo_mae_lin,
            hpo_mase_lin,

            "PENDING"
        ))

        final_conn.commit()

        # ====================================================
        # GET FINAL ID
        # ====================================================

        cur = final_conn.cursor()

        cur.execute("""
            SELECT id
            FROM runs
            WHERE config_hash = ?
        """, (h,))

        result = cur.fetchone()

        if result is None:
            print(
                f"[ERROR] Could not retrieve final ID "
                f"for {dataset}"
            )
            continue

        final_id = result[0]

        # ====================================================
        # PRINT
        # ====================================================

        print(f"Final ID:      {final_id}")
        print(f"HPO ID:        {hpo_id}")
        print(f"HPO MAE:       {hpo_mae:.6f}")
        print(f"HPO MASE:      {hpo_mase:.6f}")
        print(f"HPO MAE Lin:   {hpo_mae_lin:.6f}")
        print(f"HPO MASE Lin:  {hpo_mase_lin:.6f}")
        print(f"Final train:   {train_data}")
        print(f"Final test:    {test_data}")

    # ========================================================
    # CLOSE
    # ========================================================

    hpo_conn.close()
    final_conn.close()

    print()
    print("=" * 70)
    print("Final DB created:")
    print(FINAL_DB_PATH)
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    create_final_db()