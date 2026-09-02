import sqlite3

DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/classification_allData.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            config_hash TEXT,
            config JSON,

            dataset TEXT,
            train_data TEXT,
            eval_data TEXT,

            model_path TEXT,

            accuracy REAL,

            f1 REAL,

            status TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(config_hash, dataset)
        );
    """)

    conn.commit()
    conn.close()

    print("DB initialized.")


if __name__ == "__main__":
    init_db()