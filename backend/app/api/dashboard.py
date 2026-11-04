"""Estadísticas del dashboard."""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.database.base import get_db
from app.models.study import Prediction, Study
from app.models.user import User
from app.services import inference

router = APIRouter(tags=["Dashboard"])


@router.get("/dashboard/stats")
async def dashboard_stats(
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    from sqlalchemy import func, case

    is_admin = current_user.role in ['admin', 'radiologist']

    # admin y radiólogo ven todos los estudios; el resto, solo los suyos
    study_filter = [] if is_admin else [Study.uploaded_by == current_user.id]

    total_studies = db.query(func.count(Study.id)).filter(*study_filter).scalar() or 0

    total_preds = (
        db.query(func.count(Prediction.id))
        .join(Study, Prediction.study_id == Study.id)
        .filter(*study_filter)
        .scalar() or 0
    )

    validated = (
        db.query(func.count(Prediction.id))
        .join(Study, Prediction.study_id == Study.id)
        .filter(*study_filter, Prediction.is_validated == True)
        .scalar() or 0
    )

    dicom_count = (
        db.query(func.count(Study.id))
        .filter(*study_filter, Study.file_format == 'dcm')
        .scalar() or 0
    )

    dist_rows = (
        db.query(
            Study.disease_type,
            func.count(Prediction.id).label('total'),
            func.sum(
                case((~Prediction.prediction.ilike('%normal%'), 1), else_=0)
            ).label('positive'),
            func.avg(Prediction.confidence).label('avg_confidence'),
        )
        .join(Prediction, Prediction.study_id == Study.id)
        .filter(*study_filter)
        .group_by(Study.disease_type)
        .all()
    )

    def normalize_disease(raw):
        # el enum puede llegar como 'DiseaseType.PNEUMONIA' o como 'pneumonia'
        r = str(raw).lower().replace('diseasetype.', '')
        if 'pneumonia' in r: return 'pneumonia'
        if 'covid'     in r: return 'covid'
        if 'tubercul'  in r: return 'tuberculosis'
        return r

    dist_map = {}
    for row in dist_rows:
        d    = normalize_disease(row.disease_type)
        pos  = int(row.positive or 0)
        tot  = int(row.total or 0)
        conf = float(row.avg_confidence or 0)
        dist_map[d] = {
            'disease':        d,
            'total':          tot,
            'positive':       pos,
            'negative':       tot - pos,
            'positive_rate':  round(pos / tot * 100, 1) if tot > 0 else 0,
            'avg_confidence': round(conf * 100, 1),
        }

    disease_distribution = [
        dist_map.get('pneumonia',    {'disease': 'pneumonia',    'total': 0, 'positive': 0, 'negative': 0, 'positive_rate': 0, 'avg_confidence': 0}),
        dist_map.get('covid',        {'disease': 'covid',        'total': 0, 'positive': 0, 'negative': 0, 'positive_rate': 0, 'avg_confidence': 0}),
        dist_map.get('tuberculosis', {'disease': 'tuberculosis', 'total': 0, 'positive': 0, 'negative': 0, 'positive_rate': 0, 'avg_confidence': 0}),
    ]

    today           = datetime.utcnow().date()
    thirty_days_ago = today - timedelta(days=30)

    timeline_rows = (
        db.query(
            func.date(Study.created_at).label('date'),
            func.count(Study.id).label('count'),
        )
        .filter(*study_filter, Study.created_at >= thirty_days_ago)
        .group_by(func.date(Study.created_at))
        .all()
    )

    timeline_map = {str(row.date): int(row.count) for row in timeline_rows}
    # los días sin estudios van con 0 para que la gráfica no tenga huecos
    timeline_full = [
        {
            'date':  (thirty_days_ago + timedelta(days=i)).isoformat() + 'Z',
            'count': timeline_map.get((thirty_days_ago + timedelta(days=i)).isoformat(), 0),
        }
        for i in range(30)
    ]

    return {
        "kpis": {
            "total_studies":      total_studies,
            "total_predictions":  total_preds,
            "validated":          validated,
            "pending_validation": total_preds - validated,
            "dicom_studies":      dicom_count,
        },
        "disease_distribution": disease_distribution,
        "timeline": timeline_full,
        "system_metrics": {
            "pneumonia_accuracy":    "92.75%",
            "covid_accuracy":        "98.50%",
            "tuberculosis_accuracy": "97.80%",
            "avg_accuracy":          "96.35%",
            "yolo_map50":            "39.3%" if inference.yolo_model is not None else None,
            "unet_dice":             "96.24%",
        }
    }
