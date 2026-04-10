import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def setup_db():
    try:
        # Connect to default postgres DB to create the new one
        conn = psycopg2.connect(
            dbname='postgres',
            user='postgres',
            password='admin123', # Updated password
            host='localhost'
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Check if database exists
        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'policy_db'")
        exists = cur.fetchone()
        
        if not exists:
            cur.execute('CREATE DATABASE policy_db')
            print("Successfully created database 'policy_db'")
        else:
            print("Database 'policy_db' already exists")
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error setting up database: {e}")
        print("Note: If you have a different password, please set it in config.json")

if __name__ == "__main__":
    setup_db()
