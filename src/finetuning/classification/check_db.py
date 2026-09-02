import sqlite3

DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/classification.db"


def get_pending_ids():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT id, dataset, status
        FROM runs
        WHERE status != 'DONE'
        ORDER BY id;
    """)

    rows = cur.fetchall()

    conn.close()

    return rows


if __name__ == "__main__":

    rows = get_pending_ids()

    print(f"Found {len(rows)} unfinished runs\n")

    for row in rows:
        print(f"ID: {row[0]}, Dataset: {row[1]}, Status: {row[2]}")