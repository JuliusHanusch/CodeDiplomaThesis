import sqlite3

def getType(value):
    """
    Takes a value of a dict and return the sql type of the value
    """
    if isinstance(value, int):
        return "INTEGER"
    elif isinstance(value, float):
        return "REAL"
    elif isinstance(value, bool):
        return "BOOLEAN"
    else:
        return "TEXT"

def insertTable(table_name: str, row_data: dict, db_path: str = "AION.db"):
    """
    Takes a table name, adict and a db name, inserts the dict as one row into the table,
    creates table if not exists
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Check if the table exists
        cursor.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name=?;
        """, (table_name,))
        exists = cursor.fetchone()

        if not exists:
            # Create table with inferred types
            columns_sql = ", ".join([
                f"{col} {getType(val)}" for col, val in row_data.items()
            ])
            create_sql = f"""
                CREATE TABLE {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    {columns_sql}
                );
            """
            cursor.execute(create_sql)

        # Prepare INSERT statement
        columns = ", ".join(row_data.keys())
        placeholders = ", ".join(["?"] * len(row_data))
        insert_sql = f"""
            INSERT INTO {table_name} ({columns})
            VALUES ({placeholders});
        """
        cursor.execute(insert_sql, tuple(row_data.values()))
        conn.commit()
