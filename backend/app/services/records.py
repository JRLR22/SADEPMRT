"""Guardado de estudios, predicciones y registros de auditoría."""

import json
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.models.study import AuditLog, Prediction, Study
from config import settings

# la crea el lifespan de main.py al arrancar
storage_service = None


def _unique_name(prefix: str, ext: str = "png") -> str:
    ts  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:8]
    return f"{prefix}_{ts}_{uid}.{ext}"


def _log_audit(db: Session, user_id, action: str, resource_type: str, metadata: dict):
    audit = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        meta_data=json.dumps(metadata),
    )
    db.add(audit)


def _save_study_and_prediction(
    db, current_user, contents, filename_str, disease_type_enum,
    disease_str, prediction_label, confidence, probabilities,
    ensemble_detail, processing_time_ms,
    gradcam_url, yolo_url, seg_url, clinical, crop_points
):
    """Guarda Study + Prediction + AuditLog en BD."""
    image_url = storage_service.upload_image(contents, _unique_name("xray"), "xrays")

    study = Study(
        uploaded_by=current_user.id,
        original_image_url=image_url,
        file_size=len(contents),
        file_format=Path(filename_str).suffix.lstrip('.'),
        roi_coordinates=json.dumps(crop_points) if crop_points else None,
        disease_type=disease_type_enum,
    )
    db.add(study)
    db.flush()

    pred = Prediction(
        study_id=study.id,
        prediction=prediction_label,
        confidence=confidence,
        probabilities=json.dumps(probabilities),
        model_version="ensemble_v2.0",
        threshold_used=getattr(settings, f"{disease_str.upper()}_THRESHOLD", 0.5),
        processing_time_ms=processing_time_ms,
        gradcam_enabled=True,
        gradcam_image_url=gradcam_url,
        yolo_image_url=yolo_url,
        segmentation_image_url=seg_url,
        ensemble_detail=json.dumps(ensemble_detail),
        clinical_interpretation=json.dumps(clinical),
        risk_level=clinical['level'],
    )
    db.add(pred)

    _log_audit(db, current_user.id, "predict", "prediction",
               {'disease': disease_str, 'result': prediction_label,
                'confidence': round(confidence, 4)})
    db.commit()

    return study, pred
