from compliance_engine.database import engine, Base
from compliance_engine import models

print("Dropping and recreating tables in PostgreSQL...")
try:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("Successfully recreated tables: ", Base.metadata.tables.keys())
except Exception as e:
    print(f"Error resetting tables: {e}")
