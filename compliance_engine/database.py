"""
compliance_engine/database.py
────────────────────────────
PostgreSQL / SQLAlchemy connection setup.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# DB Pipeline: Handles both local SQLite (fallback) and PostgreSQL
# Format: postgresql://user:password@localhost:5432/db_name
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:admin123@localhost:5432/policy_db")

# Fallback to local SQLite if Postgres URL is not provided
if not DB_URL or not DB_URL.startswith("postgresql"):
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
