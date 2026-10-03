import sqlite3
import hashlib
import json
import datetime
from pathlib import Path


def hash_dict(d: dict) -> str:
    """Coverts Simple Dict into hash 

    Args:
        d (dict): Dict that shall be hashed

    Returns:
        str: hashcode
    """
    # Not sure how more complex types get hashed (often represented as some kind of RANDOM id)
    # Hence Simplify first  
    d = make_dict_storable(d)  
    # Make Consistent ~ Convert dict to a sorted tuple 
    dict_tuple = tuple(sorted(d.items()))
    # Convert to binaries
    dict_string = str(dict_tuple).encode()
    # Hash binaries
    hash_object = hashlib.sha256(dict_string)
    
    return hash_object.hexdigest()


def make_dict_storable(advanced_dictionary: dict)->dict:
    """
    Takes in a dict with advanced values like Datatime, dicts, lists, etc. 
    and converts it into a dict that can be stored in an sqlite database meaning strings, numbers, etc.

    Args:
        advanced_dictionary (dict): dict with potentionally complex datatypes

    Returns:
        dict: A dictionary with only simple datatypes
    """
    simple_dict = {}
    for key, value in advanced_dictionary.items():
        if isinstance(value, (int, float, str)):
            pass
        elif isinstance(value, bool):
            value = 1 if value else 0
        elif isinstance(value, (bytes, Path, list, tuple)) or value is None:
            value = str(value)
        elif isinstance(value, (datetime.datetime, datetime.date)):
            value = value.strftime("%d-%m-%Y %H:%M:%S")
        elif isinstance(value, dict):
            value = json.dumps(value)
        #elif isinstance(value, str):
        else:
            raise NotImplementedError(f"dtype {type(value)} is currently not storable, but can likely be easily added in make_dict_storable()")
        simple_dict[key] = value

    return simple_dict


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
    row_data = make_dict_storable(row_data)
    try: 
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
    except sqlite3.OperationalError as e:
        # Inject missing Column
        with sqlite3.connect(db_path) as conn:
            if "no column named" in str(e).lower():
                print("A column seems to be missing try to insert it now")
                cursor = conn.cursor()
                # check if column exists
                cursor.execute(f"PRAGMA table_info({table_name})")
                cols = [info[1] for info in cursor.fetchall()]
                missing_cols = [col for col in row_data.keys() if col not in cols]
                for missing_col in missing_cols:
                    print(f"Inserted {missing_col} into {table_name}")
                    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {missing_col} {getType(row_data[missing_col])}")
                conn.commit()
                insertTable(table_name=table_name, row_data=row_data, db_path=db_path)
            else:
                raise 
