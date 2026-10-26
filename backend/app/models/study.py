# ============================================================================
# STUDY.PY - MODELOS DE BD (SQLite compatible, sin PostgreSQL/JSONB/UUID)
# ============================================================================

from sqlalchemy import (
    Column, String, Integer, Float, Boolean,
    DateTime, ForeignKey, Text, Enum
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid as uuid_pkg
import enum
from app.database.base import Base


class DiseaseType(str, enum.Enum):
    PNEUMONIA    = "pneumonia"
    COVID        = "covid"
    TUBERCULOSIS = "tuberculosis"


class Study(Base):
    __tablename__ = "studies"

    id                  = Column(String(36), primary_key=True, default=lambda: str(uuid_pkg.uuid4()))
    uploaded_by         = Column(String(36), ForeignKey("users.id"), nullable=False)
    patient_identifier  = Column(String(255), nullable=True)
    patient_age         = Column(Integer, nullable=True)
    patient_sex         = Column(String(10), nullable=True)
    original_image_url  = Column(String(500), nullable=False)
    file_size           = Column(Integer, nullable=True)
    file_format         = Column(String(10), nullable=True)
    roi_coordinates     = Column(Text, nullable=True)   # JSON serializado como Text
    disease_type        = Column(Enum(DiseaseType), nullable=False)
    dicom_metadata      = Column(Text, nullable=True)   # JSON serializado como Text
    study_date          = Column(DateTime(timezone=True), server_default=func.now())
    created_at          = Column(DateTime(timezone=True), server_default=func.now())
    dicom_metadata      = Column(Text,    nullable=True)   # JSON con metadatos DICOM
    source              = Column(String(20), default="upload")  # "upload" o "dicom"

    predictions = relationship("Prediction", back_populates="study", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id                      = Column(String(36), primary_key=True, default=lambda: str(uuid_pkg.uuid4()))
    study_id                = Column(String(36), ForeignKey("studies.id"), nullable=False)
    prediction              = Column(String(100), nullable=False)
    confidence              = Column(Float, nullable=False)
    probabilities           = Column(Text, nullable=False)   # JSON serializado
    model_version           = Column(String(50), nullable=False)
    threshold_used          = Column(Float, nullable=True)
    processing_time_ms      = Column(Float, nullable=True)
    gradcam_enabled         = Column(Boolean, default=True)
    gradcam_image_url       = Column(String(500), nullable=True)
    yolo_image_url          = Column(String(500), nullable=True)   # NUEVO: YOLO
    segmentation_image_url  = Column(String(500), nullable=True)   # NUEVO: U-Net
    ensemble_detail         = Column(Text, nullable=True)          # NUEVO: detalle ensemble
    clinical_interpretation = Column(Text, nullable=True)          # JSON serializado
    risk_level              = Column(String(20), nullable=True)
    is_validated            = Column(Boolean, default=False)
    validated_by            = Column(String(36), ForeignKey("users.id"), nullable=True)
    validation_result       = Column(String(100), nullable=True)
    radiologist_notes       = Column(Text, nullable=True)
    validated_at            = Column(DateTime(timezone=True), nullable=True)
    predicted_at            = Column(DateTime(timezone=True), server_default=func.now())

    study = relationship("Study", back_populates="predictions")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id            = Column(String(36), primary_key=True, default=lambda: str(uuid_pkg.uuid4()))
    user_id       = Column(String(36), ForeignKey("users.id"), nullable=True)
    action        = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=True)
    resource_id   = Column(String(36), nullable=True)
    ip_address    = Column(String(45), nullable=True)
    user_agent    = Column(Text, nullable=True)
    meta_data      = Column(Text, nullable=True)   # JSON serializado
    created_at    = Column(DateTime(timezone=True), server_default=func.now())