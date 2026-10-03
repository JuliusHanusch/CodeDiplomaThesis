import sqlite3

DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/imputation/imputation_allData_new.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_hash TEXT,
            config JSON,
            dataset TEXT,
            train_data TEXT,
            eval_data TEXT,
            status TEXT,

            model_path TEXT,

            -- 12.5% masking
            MAE_0125 REAL,
            MASE_0125 REAL,
            MAE_Lin_0125 REAL,
            MASE_Lin_0125 REAL,

            -- 25% masking
            MAE_025 REAL,
            MASE_025 REAL,
            MAE_Lin_025 REAL,
            MASE_Lin_025 REAL,

            -- 37.5% masking
            MAE_0375 REAL,
            MASE_0375 REAL,
            MAE_Lin_0375 REAL,
            MASE_Lin_0375 REAL,

            -- 50% masking
            MAE_050 REAL,
            MASE_050 REAL,
            MAE_Lin_050 REAL,
            MASE_Lin_050 REAL,

            -- Average across the four masking ratios
            MAE_avg REAL,
            MASE_avg REAL,
            MAE_Lin_avg REAL,
            MASE_Lin_avg REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(config_hash, dataset)
        );
    """)

    conn.commit()
    conn.close()
    print("DB initialized.")


if __name__ == "__main__":
    init_db()