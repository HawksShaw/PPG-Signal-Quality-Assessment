import os
import json
import uuid
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
from service.configs import settings, logger

database_url = settings["storage"]["database_url"]
object_store_dir = settings["storage"]["object_store_directory"]

os.makedirs(object_store_dir, exist_ok=True)
logger.info(f"--- Storage initialized ---\nDB: {database_url}, Object Store: {object_store_dir}")
base = declarative_base()

class AssessmentRecord(base):
    __tablename__ = "assessments"

    id = Column(String, primary_key=True)
    subject_id = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String)
    confidence = Column(Float)
    reasons = Column(String)
    metrics = Column(String)
    file_path = Column(String)

engine = create_engine(database_url, connect_args={"check_same_thread": False})
base.metadata.create_all(bind=engine)
LocalSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def save_assessment(metadata: dict, report: dict, raw_signals: dict):
    record_id = str(uuid.uuid4())

    filename = f"{record_id}.npy"
    file_path = os.path.join(object_store_dir, filename)

    np.save(file_path, raw_signals)

    db = LocalSession()
    try:
        record = AssessmentRecord(
            id = record_id,
            subject_id = metadata.get("subject_id", "unknown"),
            status = report["status"],
            confidence = report["confidence"],
            reasons = json.dumps(report["reasons"]),
            metrics = json.dumps(report["metrics"]),
            file_path = file_path
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record.id
    finally:
        db.close()
