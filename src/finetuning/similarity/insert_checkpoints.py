import sqlite3
from pathlib import Path


DB_PATH = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/src/finetuning/similarity/similarity.db"
)

CHECKPOINT_ROOT = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/FineTunedModels/Similarity"
)


def main():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            dataset,
            config_hash,
            status,
            accuracy,
            auroc
        FROM runs
        WHERE accuracy IS NULL
           OR auroc IS NULL
    """)

    rows = cursor.fetchall()

    updated = []
    missing = []

    for run_id, dataset, config_hash, status, accuracy, auroc in rows:

        checkpoint = (
            CHECKPOINT_ROOT
            / str(config_hash)
            / "checkpoint-final"
        )

        if checkpoint.is_dir():

            cursor.execute("""
                UPDATE runs
                SET model_path = ?
                WHERE id = ?
            """, (
                str(checkpoint),
                run_id
            ))

            updated.append(run_id)

            print(
                f"[UPDATED] ID={run_id} | "
                f"{dataset} | "
                f"{config_hash}"
            )

        else:

            missing.append(run_id)

            print(
                f"[MISSING] ID={run_id} | "
                f"{dataset} | "
                f"{config_hash}"
            )

    conn.commit()
    conn.close()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Rows checked       : {len(rows)}")
    print(f"Model paths set    : {len(updated)}")
    print(f"Checkpoints missing: {len(missing)}")

    print("\nUpdated IDs:")
    print(",".join(map(str, updated)))

    print("\nMissing IDs:")
    print(",".join(map(str, missing)))


if __name__ == "__main__":
    main()