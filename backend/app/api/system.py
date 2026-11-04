"""Estado del servicio."""

import os
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.services import inference
from config import settings

router = APIRouter(tags=["Sistema"])


@router.get("/health")
async def health_check():
    db_status = "ok"
    try:
        from app.database.base import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {str(e)}"

    models_status = {
        "pneumonia":    "pneumonia"    in inference.ensemble_models,
        "covid":        "covid"        in inference.ensemble_models,
        "tuberculosis": "tuberculosis" in inference.ensemble_models,
        "yolo":         inference.yolo_model is not None,
        "unet":         inference.unet_model is not None,
    }
    # yolo es opcional: su ausencia no marca el servicio como degradado
    all_models_ok = all(ok for name, ok in models_status.items() if name != "yolo")
    overall_ok    = db_status == "ok" and all_models_ok

    return JSONResponse(
        status_code=200 if overall_ok else 503,
        content={
            "status":       "healthy" if overall_ok else "degraded",
            "timestamp":    time.time(),
            "database":     db_status,
            "models_loaded": models_status,
            "models_dir":   "ok" if os.path.exists(os.getenv("MODELS_DIR", "./models")) else "missing",
            "version":      settings.APP_VERSION,
            "environment":  os.getenv("APP_ENV", "development"),
        }
    )
