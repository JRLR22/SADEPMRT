"""Consulta de estudios y validación por radiólogo."""

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.core.rate_limit import rate_limit_studies, rate_limit_validate
from app.database.base import get_db
from app.models.study import Prediction, Study
from app.models.user import User
from app.services.records import _log_audit

router = APIRouter(tags=["Estudios"])


@router.get("/studies")
async def get_studies(
    skip:         int     = Query(0,   ge=0),
    limit:        int     = Query(50,  ge=1, le=200),
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
    _rate:        None    = Depends(rate_limit_studies),
):
    from sqlalchemy.orm import joinedload

    query = db.query(Study)
    if current_user.role not in ['admin', 'radiologist']:
        query = query.filter(Study.uploaded_by == current_user.id)

    total   = query.count()
    studies = (
        query
        .options(joinedload(Study.predictions))
        .order_by(Study.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    result = []
    for s in studies:
        result.append({
            'id':                 str(s.id),
            'disease_type':       s.disease_type,
            'original_image_url': s.original_image_url,
            'file_size':          s.file_size,
            'file_format':        s.file_format,
            'created_at':         s.created_at.isoformat() + 'Z',
            'predictions': [{
                'id':                     str(p.id),
                'prediction':             p.prediction,
                'confidence':             p.confidence,
                'probabilities':          json.loads(p.probabilities) if p.probabilities else {},
                'ensemble_detail':        json.loads(p.ensemble_detail) if p.ensemble_detail else {},
                'gradcam_image_url':      p.gradcam_image_url,
                'yolo_image_url':         p.yolo_image_url,
                'segmentation_image_url': p.segmentation_image_url,
                'is_validated':           p.is_validated,
                'validation_result':      p.validation_result,
                'radiologist_notes':      p.radiologist_notes,
                'validated_at':           p.validated_at.isoformat() + 'Z' if p.validated_at else None,
                'predicted_at':           p.predicted_at.isoformat() + 'Z',
            } for p in s.predictions]
        })

    return {"total": total, "skip": skip, "limit": limit, "studies": result}


@router.get("/studies/{study_id}")
async def get_study(
    study_id:     str,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
    _rate:        None    = Depends(rate_limit_studies),
):
    from sqlalchemy.orm import joinedload

    s = db.query(Study).options(joinedload(Study.predictions)).filter(Study.id == study_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Estudio no encontrado")
    if s.uploaded_by != current_user.id and current_user.role not in ['admin', 'radiologist']:
        raise HTTPException(status_code=403, detail="Sin permiso")

    _log_audit(db, current_user.id, "view", "study", {'study_id': study_id})
    db.commit()

    return {
        'id':                  str(s.id),
        'uploaded_by':         str(s.uploaded_by),
        'patient_identifier':  s.patient_identifier,
        'patient_age':         s.patient_age,
        'patient_sex':         s.patient_sex,
        'disease_type':        s.disease_type,
        'original_image_url':  s.original_image_url,
        'file_size':           s.file_size,
        'file_format':         s.file_format,
        'roi_coordinates':     json.loads(s.roi_coordinates) if s.roi_coordinates else None,
        'created_at':          s.created_at.isoformat() + 'Z',
        'predictions': [{
            'id':                      str(p.id),
            'prediction':              p.prediction,
            'confidence':              p.confidence,
            'probabilities':           json.loads(p.probabilities) if p.probabilities else {},
            'ensemble_detail':         json.loads(p.ensemble_detail) if p.ensemble_detail else {},
            'model_version':           p.model_version,
            'threshold_used':          p.threshold_used,
            'processing_time_ms':      p.processing_time_ms,
            'gradcam_image_url':       p.gradcam_image_url,
            'yolo_image_url':          p.yolo_image_url,
            'segmentation_image_url':  p.segmentation_image_url,
            'clinical_interpretation': json.loads(p.clinical_interpretation) if p.clinical_interpretation else {},
            'risk_level':              p.risk_level,
            'is_validated':            p.is_validated,
            'validated_by':            str(p.validated_by) if p.validated_by else None,
            'validation_result':       p.validation_result,
            'radiologist_notes':       p.radiologist_notes,
            'validated_at':            p.validated_at.isoformat() + 'Z' if p.validated_at else None,
            'predicted_at':            p.predicted_at.isoformat() + 'Z',
        } for p in s.predictions]
    }


@router.put("/predictions/{prediction_id}/validate")
async def validate_prediction(
    prediction_id:     str,
    validation_result: str,
    radiologist_notes: str  = None,
    db:                Session = Depends(get_db),
    current_user:      User    = Depends(get_current_user),
    _rate:             None    = Depends(rate_limit_validate),
):
    if current_user.role not in ['admin', 'radiologist']:
        raise HTTPException(status_code=403, detail="Solo radiólogos pueden validar")

    pred = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not pred:
        raise HTTPException(status_code=404, detail="Predicción no encontrada")

    pred.is_validated      = True
    pred.validated_by      = current_user.id
    pred.validation_result = validation_result
    pred.radiologist_notes = radiologist_notes
    pred.validated_at      = datetime.utcnow()

    _log_audit(db, current_user.id, "validate", "prediction",
               {'prediction_id': prediction_id, 'result': validation_result})
    db.commit()

    return {
        'success':       True,
        'message':       'Predicción validada',
        'prediction_id': prediction_id,
        'validated_at':  pred.validated_at.isoformat() + 'Z',
    }
