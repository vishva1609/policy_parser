"""
compliance_engine/models.py
──────────────────────────
SQLAlchemy Models for PostgreSQL
"""
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, index=True)
    upload_date = Column(DateTime, default=datetime.utcnow)
    total_clauses = Column(Integer)
    
    clauses = relationship("Clause", back_populates="document", cascade="all, delete")

class Clause(Base):
    __tablename__ = "clauses"
    id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(Integer, ForeignKey("documents.id"))
    clause_id = Column(String, index=True)  # e.g. "A-1.2"
    title = Column(String)
    text = Column(Text)
    category = Column(String) # Security, Logging, etc.
    strength = Column(String) # mandatory, recommended
    
    document = relationship("Document", back_populates="clauses")

class ComparisonJob(Base):
    __tablename__ = "comparison_jobs"
    id = Column(String, primary_key=True, index=True) # UUID / Session ID
    status = Column(String) # queued, running, done, error
    current_step = Column(String) # parse, segment, embed, compare, score, export
    
    name_a = Column(String)
    name_b = Column(String)
    
    # Intermediate Data
    raw_text_a = Column(Text)
    raw_text_b = Column(Text)
    segments_a = Column(JSON)
    segments_b = Column(JSON)
    
    overall_score = Column(Float)
    results_json = Column(JSON) # Step compare/score output
    
    created_at = Column(DateTime, default=datetime.utcnow)
