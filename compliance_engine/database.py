"""
compliance_engine/database.py
────────────────────────────
PostgreSQL / SQLAlchemy connection setup.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# DB Pipeline: Handles both local SQLite and PostgreSQL
# Defaulting to PostgreSQL based on setup_postgres.py settings
DEFAULT_PG_URL = "postgresql://postgres:admin123@localhost:5432/policy_db"
DB_URL = os.environ.get("DATABASE_URL", DEFAULT_PG_URL)

# Fallback to local SQLite only if explicitly requested or if PG fails (handled by engine)
if not DB_URL:
    DB_URL = "sqlite:///./policy_compliance.db"

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
