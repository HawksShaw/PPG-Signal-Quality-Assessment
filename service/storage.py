import os
import json
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer
from sqlalchemy.orm import declarative_base, sessionmaker

database_url = "sqlite:///./quality_assessment.db"
object_store_dir = "./object_store"

os.makedirs(object_store_dir, exist_ok=True)
base = declarative_base()

class AssessmentRecord(base):
    __tablename__ = "assessments"

    id = Column(String, primary_key=True)
    subject_id = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String)
    confidence = Column(Float)
    reasons = Column(String)
    file_path = Column(String)

engine = create_engine(database_url, connect_args={"check_same_thread": False})
