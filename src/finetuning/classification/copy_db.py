import sqlite3
from pathlib import Path


SOURCE_DB = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/classification_small_final.db"
)

TARGET_DB = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/classification.db"
)


def create_shared_config_db():
    target_path = Path(TARGET_DB)

    # Remove target DB if it already exists
    if target_path.exists():
        target_path.unlink()

    source_conn = sqlite3.connect(SOURCE_DB)
    target_conn = sqlite3.connect(TARGET_DB)

    source_cur = source_conn.cursor()
    target_cur = target_conn.cursor()

    # ---------------------------------------------------------------
    # Get the exact schema of the source table
    # ---------------------------------------------------------------

    source_cur.execute("""
        SELECT sql
        FROM sqlite_master
        WHERE type = 'table'
        AND name = 'runs'
    """)

    schema = source_cur.fetchone()

    if schema is None:
        raise RuntimeError("Table 'runs' does not exist in source DB.")

    # Create target table with exactly the same schema
    target_cur.execute(schema[0])

    # ---------------------------------------------------------------
    # Get all columns
    # ---------------------------------------------------------------

    source_cur.execute("PRAGMA table_info(runs)")
    columns = source_cur.fetchall()

    column_names = [column[1] for column in columns]

    # ---------------------------------------------------------------
    # Columns containing model-specific results
    # These are reset in the target DB.
    # Only columns that actually exist are used.
    # ---------------------------------------------------------------

    reset_columns = {
        "status": "pending",
        "model_path": None,
        "accuracy": None,
        "f1": None,
        "f1_score": None,
        "rmse": None,
        "mae": None,
        "mape": None,
        "mase": None,
        "auroc": None,
        "auprc": None,
        "wql": None,
    }

    reset_columns = {
        column: value
        for column, value in reset_columns.items()
        if column in column_names
    }

    # ---------------------------------------------------------------
    # Columns that are copied unchanged
    # ---------------------------------------------------------------

    copy_columns = [
        column
        for column in column_names
        if column != "id" and column not in reset_columns
    ]

    print("Copying columns:")
    for column in copy_columns:
        print(f"  {column}")

    print("\nResetting columns:")
    for column, value in reset_columns.items():
        print(f"  {column} -> {value}")

    # ---------------------------------------------------------------
    # Read source rows
    # ---------------------------------------------------------------

    select_columns = ", ".join(
        f'"{column}"' for column in copy_columns
    )

    source_cur.execute(f"""
        SELECT {select_columns}
        FROM runs
        ORDER BY id
    """)

    rows = source_cur.fetchall()

    # ---------------------------------------------------------------
    # Insert into target
    # ---------------------------------------------------------------

    insert_columns = copy_columns + list(reset_columns.keys())

    insert_column_string = ", ".join(
        f'"{column}"' for column in insert_columns
    )

    placeholders = ", ".join(
        ["?"] * len(insert_columns)
    )

    insert_rows = []

    for row in rows:
        reset_values = [
            reset_columns[column]
            for column in reset_columns
        ]

        insert_rows.append(
            tuple(row) + tuple(reset_values)
        )

    target_cur.executemany(
        f"""
        INSERT INTO runs (
            {insert_column_string}
        )
        VALUES (
            {placeholders}
        )
        """,
        insert_rows
    )

    target_conn.commit()

    print(f"\nCopied {len(rows)} rows.")
    print(f"Created: {TARGET_DB}")

    source_conn.close()
    target_conn.close()


if __name__ == "__main__":
    create_shared_config_db()