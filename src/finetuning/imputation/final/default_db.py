import sqlite3
import json
import hashlib
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

SOURCE_DB_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/src/finetuning/imputation/final/"
    "imputation_bestConfigs_new.db"
)

DEFAULT_DB_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/src/finetuning/imputation/final/"
    "imputation_defaultConfigs.db"
)


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_CONFIG = {
    "num_train_epochs": 10,
    "per_device_train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-4,
    "warmup_ratio": 0.1,
    "masking_prob": 0.15,
    "mean_span_length": 8,
}


# ============================================================
# SEEDS
# ============================================================

SEEDS = [42, 43, 44, 45, 46]


# ============================================================
# CONFIG HASH
# ============================================================

def config_hash(config, dataset):

    return hashlib.sha256(
        json.dumps(
            {
                "config": config,
                "dataset": dataset,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


# ============================================================
# MAIN
# ============================================================

def create_default_db():

    source_path = Path(SOURCE_DB_PATH)
    output_path = Path(DEFAULT_DB_PATH)

    # --------------------------------------------------------
    # Check source DB
    # --------------------------------------------------------

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source DB does not exist:\n{source_path}"
        )

    # --------------------------------------------------------
    # Do not overwrite an existing DB accidentally
    # --------------------------------------------------------

    if output_path.exists():
        raise FileExistsError(
            f"Output DB already exists:\n{output_path}\n\n"
            f"Delete it manually first if you want to recreate it."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Copy entire database
    # --------------------------------------------------------

    source_conn = sqlite3.connect(
        str(source_path)
    )

    output_conn = sqlite3.connect(
        str(output_path)
    )

    print("=" * 70)
    print("Copying database")
    print("=" * 70)

    source_conn.backup(output_conn)

    output_conn.commit()

    source_conn.close()

    print(f"Source: {source_path}")
    print(f"Output: {output_path}")

    # --------------------------------------------------------
    # Connect to copied DB
    # --------------------------------------------------------

    conn = output_conn

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")

    # --------------------------------------------------------
    # Get all datasets
    # --------------------------------------------------------

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            dataset,
            train_data,
            test_data
        FROM runs
        ORDER BY id
    """)

    rows = cursor.fetchall()

    print()
    print(f"Found {len(rows)} datasets.")

    # --------------------------------------------------------
    # Update every dataset
    # --------------------------------------------------------

    for row in rows:

        run_id = row[0]
        dataset = row[1]
        train_data = row[2]
        test_data = row[3]

        print()
        print("-" * 70)
        print(f"Dataset: {dataset}")
        print(f"ID:      {run_id}")

        # ----------------------------------------------------
        # Create dataset-specific config hash
        # ----------------------------------------------------

        h = config_hash(
            DEFAULT_CONFIG,
            dataset,
        )

        # ----------------------------------------------------
        # Replace configuration and delete all results
        # ----------------------------------------------------

        update_values = {
            "config_hash": h,
            "config": json.dumps(
                DEFAULT_CONFIG,
                sort_keys=True,
            ),

            # HPO information
            "source_hpo_id": None,
            "hpo_MAE_avg": None,
            "hpo_MASE_avg": None,
            "hpo_MAE_Lin_avg": None,
            "hpo_MASE_Lin_avg": None,

            # Status
            "status": "PENDING",
        }

        # ----------------------------------------------------
        # Clear all per-seed results
        # ----------------------------------------------------

        for seed in SEEDS:

            update_values[f"model_path_{seed}"] = None

            update_values[f"MAE_0125_{seed}"] = None
            update_values[f"MASE_0125_{seed}"] = None
            update_values[f"MAE_Lin_0125_{seed}"] = None
            update_values[f"MASE_Lin_0125_{seed}"] = None

            update_values[f"MAE_025_{seed}"] = None
            update_values[f"MASE_025_{seed}"] = None
            update_values[f"MAE_Lin_025_{seed}"] = None
            update_values[f"MASE_Lin_025_{seed}"] = None

            update_values[f"MAE_0375_{seed}"] = None
            update_values[f"MASE_0375_{seed}"] = None
            update_values[f"MAE_Lin_0375_{seed}"] = None
            update_values[f"MASE_Lin_0375_{seed}"] = None

            update_values[f"MAE_050_{seed}"] = None
            update_values[f"MASE_050_{seed}"] = None
            update_values[f"MAE_Lin_050_{seed}"] = None
            update_values[f"MASE_Lin_050_{seed}"] = None

            update_values[f"MAE_avg_{seed}"] = None
            update_values[f"MASE_avg_{seed}"] = None
            update_values[f"MAE_Lin_avg_{seed}"] = None
            update_values[f"MASE_Lin_avg_{seed}"] = None

        # ----------------------------------------------------
        # Clear overall results
        # ----------------------------------------------------

        update_values["MAE_avg"] = None
        update_values["MAE_std"] = None

        update_values["MASE_avg"] = None
        update_values["MASE_std"] = None

        update_values["MAE_Lin_avg"] = None
        update_values["MAE_Lin_std"] = None

        update_values["MASE_Lin_avg"] = None
        update_values["MASE_Lin_std"] = None

        # ----------------------------------------------------
        # Clear inference times
        # ----------------------------------------------------

        for seed in SEEDS:
            update_values[f"inference_time_{seed}"] = None

        update_values["inference_time_avg"] = None
        update_values["inference_time_std"] = None

        # ----------------------------------------------------
        # Build UPDATE statement
        # ----------------------------------------------------

        set_clause = ", ".join(
            f"{column} = ?"
            for column in update_values
        )

        sql = f"""
            UPDATE runs
            SET {set_clause}
            WHERE id = ?
        """

        values = list(update_values.values())
        values.append(run_id)

        conn.execute(
            sql,
            values,
        )

        print("Configuration:")
        print(json.dumps(
            DEFAULT_CONFIG,
            indent=4,
        ))

        print(f"Train data: {train_data}")
        print(f"Test data:  {test_data}")
        print("Results cleared.")
        print("Status: PENDING")

    # --------------------------------------------------------
    # Commit
    # --------------------------------------------------------

    conn.commit()

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    cursor.execute("""
        SELECT
            COUNT(*)
        FROM runs
    """)

    count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT
            COUNT(*)
        FROM runs
        WHERE status = 'PENDING'
    """)

    pending_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT
            COUNT(*)
        FROM runs
        WHERE config IS NOT NULL
    """)

    config_count = cursor.fetchone()[0]

    # --------------------------------------------------------
    # Close
    # --------------------------------------------------------

    conn.close()

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("=" * 70)
    print("DEFAULT DATABASE CREATED")
    print("=" * 70)

    print(f"Output DB:       {output_path}")
    print(f"Total datasets:  {count}")
    print(f"Pending runs:    {pending_count}")
    print(f"Configs present: {config_count}")

    print()
    print("Default config:")
    print(json.dumps(
        DEFAULT_CONFIG,
        indent=4,
    ))

    print()
    print("All result columns have been cleared.")
    print("Original database was not modified.")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    create_default_db()