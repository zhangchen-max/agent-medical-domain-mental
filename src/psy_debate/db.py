from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _build_url() -> str:
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    name = os.getenv("DB_NAME", "psy_debate")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}?charset=utf8mb4"


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(_build_url(), pool_pre_ping=True, echo=False)
    return _engine


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return _SessionLocal()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"

    student_id = Column(String(30), primary_key=True)
    first_visit = Column(DateTime, default=datetime.utcnow)
    last_visit = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    verbal_style = Column(String(20), nullable=True)
    total_sessions = Column(Integer, default=0)


class ClinicalPortrait(Base):
    __tablename__ = "clinical_portraits"

    student_id = Column(String(30), primary_key=True)
    portrait_json = Column(JSON, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SessionRecord(Base):
    __tablename__ = "session_records"

    session_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    student_id = Column(String(30), nullable=False, index=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    stage_reached = Column(String(50), nullable=True)
    turn_count = Column(Integer, default=0)
    risk_level = Column(String(20), nullable=True)
    report_json = Column(JSON, nullable=True)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create database and tables if not exist."""
    engine = get_engine()
    # Create database if not exists
    db_name = os.getenv("DB_NAME", "psy_debate")
    base_url = _build_url().rsplit(f"/{db_name}", 1)[0]
    tmp_engine = create_engine(base_url + "?charset=utf8mb4", pool_pre_ping=True)
    with tmp_engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4"))
        conn.commit()
    tmp_engine.dispose()
    Base.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def load_patient(student_id: str) -> dict[str, Any]:
    """
    Returns {'is_new': bool, 'verbal_style': str|None, 'portrait': dict|None}
    """
    with get_session() as db:
        patient = db.get(Patient, student_id)
        if patient is None:
            return {"is_new": True, "verbal_style": None, "portrait": None}

        portrait_row = db.get(ClinicalPortrait, student_id)
        portrait = portrait_row.portrait_json if portrait_row else None
        return {
            "is_new": False,
            "verbal_style": patient.verbal_style,
            "portrait": portrait,
        }


def upsert_patient(student_id: str, verbal_style: str | None = None) -> None:
    with get_session() as db:
        patient = db.get(Patient, student_id)
        if patient is None:
            patient = Patient(student_id=student_id, verbal_style=verbal_style)
            db.add(patient)
        else:
            patient.last_visit = datetime.utcnow()
            if verbal_style:
                patient.verbal_style = verbal_style
        db.commit()


def save_portrait(student_id: str, portrait: dict[str, Any]) -> None:
    """Upsert the clinical portrait — called after every turn."""
    # Only persist symptoms that are probable or above
    portrait_to_save = _filter_portrait_for_persistence(portrait)
    with get_session() as db:
        row = db.get(ClinicalPortrait, student_id)
        if row is None:
            row = ClinicalPortrait(student_id=student_id, portrait_json=portrait_to_save)
            db.add(row)
        else:
            row.portrait_json = portrait_to_save
            row.updated_at = datetime.utcnow()
        db.commit()


def create_session_record(student_id: str) -> str:
    session_id = str(uuid.uuid4())
    with get_session() as db:
        record = SessionRecord(session_id=session_id, student_id=student_id)
        db.add(record)
        # Increment total_sessions on patient
        patient = db.get(Patient, student_id)
        if patient:
            patient.total_sessions = (patient.total_sessions or 0) + 1
        db.commit()
    return session_id


def close_session_record(
    session_id: str,
    stage_reached: str,
    turn_count: int,
    risk_level: str,
    report: dict[str, Any] | None,
) -> None:
    with get_session() as db:
        record = db.get(SessionRecord, session_id)
        if record:
            record.end_time = datetime.utcnow()
            record.stage_reached = stage_reached
            record.turn_count = turn_count
            record.risk_level = risk_level
            record.report_json = report
            db.commit()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_portrait_for_persistence(portrait: dict[str, Any]) -> dict[str, Any]:
    """
    Only keep symptoms with status probable/confirmed.
    suspected症状不持久化，避免把错误信息带到下次会话。
    """
    import copy
    p = copy.deepcopy(portrait)
    symptoms = p.get("symptoms", {})
    filtered = {
        k: v for k, v in symptoms.items()
        if v.get("status") in ("probable", "confirmed")
    }
    p["symptoms"] = filtered
    return p
