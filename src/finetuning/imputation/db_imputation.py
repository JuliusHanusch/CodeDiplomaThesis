import sqlite3

DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/imputation/imputation.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_hash TEXT,
            config JSON,
            dataset TEXT,
            train_data TEXT,
            status TEXT,
            model_path TEXT,
            masking_ratio REAL,
            mean_span_length REAL,
            MAE REAL,
            MASE REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(config_hash, dataset)
        );
    """)

    conn.commit()
    conn.close()
    print("DB initialized.")


if __name__ == "__main__":
    init_db()