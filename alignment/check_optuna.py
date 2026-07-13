import sqlite3

def check_optuna_db(db: str):
    con = sqlite3.connect(db)
    cur = con.cursor()

    tables = [
        row[0]
        for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]

    print("TABLES")
    for t in tables:
        print(" -", t)

    for table in tables:
        print("\n" + "=" * 80)
        print("TABLE:", table)

        print("\nSCHEMA:")
        schema = cur.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        print(schema[0] if schema else "")

        cols = [row[1] for row in cur.execute(f"PRAGMA table_info({table})")]
        print("\nCOLUMNS:")
        print(cols)

        n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print("\nROW COUNT:", n)

        print("\nFIRST 10 ROWS:")
        rows = cur.execute(f"SELECT * FROM {table} LIMIT 10").fetchall()
        for row in rows:
            print(dict(zip(cols, row)))

    con.close()


db = "Results_v3/optuna_v2.db"
check_optuna_db(db)