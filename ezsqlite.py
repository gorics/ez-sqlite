import sqlite3
import time

def sqlite(db, sql_query, *params):
    while True:
        try:
            conn = sqlite3.connect(db)
            c = conn.cursor()
            c.execute(sql_query, params)
            result = c.fetchall()
            conn.commit()
            break
        except sqlite3.OperationalError as e:
            if 'locked' in str(e):
                time.sleep(0.1)
                continue
            else:
                raise
        finally:
            conn.close()
    return result

# Example usage
result = sqlite("example.db", "SELECT * FROM users WHERE username=?", ("user1",))
for row in result:
    print(row)
#박은성 쵝오
