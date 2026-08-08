import psycopg

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "face_finder"
DB_USER = "postgres"
DB_PASSWORD = "Jayphoto"

try:
    conn = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    print("PostgreSQL connection successful!")

    with conn.cursor() as cur:
        cur.execute("SELECT version();")
        version = cur.fetchone()
        print(version[0])

    conn.close()

except Exception as e:
    print("Database connection failed:")
    print(e)