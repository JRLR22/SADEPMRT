"""Endpoints de análisis: por enfermedad, automático y DICOM."""

import io
import json
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from loguru import logger
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.core.rate_limit import rate_limit_predict
from app.core.validation import validate_image_file
from app.database.base import get_db
from app.models.study import DiseaseType
from app.models.user import User
from app.services import inference, records
from app.services.inference import (
    dicom_to_pil, ensemble_predict, generate_gradcam, get_clinical_interpretation,
    get_validation_metrics, preprocess_image, run_segmentation, run_yolo,
)
from app.services.records import _save_study_and_prediction, _unique_name
from config import settings
from dicom_utils import format_dicom_date

router = APIRouter(tags=["Análisis"])


async def _predict(
    file, disease: str, disease_enum, class_names: list,
    threshold: float, crop_points, run_yolo_flag: bool,
    run_seg_flag: bool, db, current_user
):
    start    = time.time()
    contents = await file.read()

    validate_image_file(contents, file.filename, allow_dicom=False)

    points = json.loads(crop_points) if crop_points else None

    tensor, original_image = preprocess_image(contents)

    result          = ensemble_predict(disease, tensor)
    probs_avg       = result['probs_avg']
    predicted_class = result['predicted_class']
    confidence      = result['confidence']
    ensemble_detail = result['detail']

    prediction_label = class_names[predicted_class]
    probabilities    = {class_names[i]: float(probs_avg[i]) for i in range(len(class_names))}
    processing_ms    = (time.time() - start) * 1000

    gradcam_url   = None
    heatmap_bytes = generate_gradcam(disease, tensor, original_image, predicted_class)
    if heatmap_bytes:
        gradcam_url = records.storage_service.upload_image(
            heatmap_bytes, _unique_name("gradcam"), "gradcam"
        )

    yolo_url   = None
    detections = []
    if run_yolo_flag and inference.yolo_model is not None:
        yolo_bytes, detections = run_yolo(contents)
        yolo_url = records.storage_service.upload_image(
            yolo_bytes, _unique_name("yolo"), "yolo"
        )

    seg_url = None
    if run_seg_flag:
        seg_bytes = run_segmentation(contents)
        seg_url   = records.storage_service.upload_image(
            seg_bytes, _unique_name("seg"), "segmentation"
        )

    clinical = get_clinical_interpretation(confidence, disease)

    study, pred = _save_study_and_prediction(
        db, current_user, contents, file.filename, disease_enum,
        disease, prediction_label, confidence, probabilities,
        ensemble_detail, processing_ms,
        gradcam_url, yolo_url, seg_url, clinical, points
    )

    response = {
        "status":       "success",
        "disease_type": disease,
        "timestamp":    datetime.utcnow().isoformat() + 'Z',
        "study_id":     str(study.id),
        "prediction_id":str(pred.id),
        "result": {
            "prediction":              prediction_label,
            "confidence":              confidence,
            "probabilities":           probabilities,
            "threshold_used":          threshold,
            "clinical_interpretation": clinical,
        },
        "ensemble": {
            "models_used": ["DenseNet121", "ResNet50", "EfficientNet-B0"],
            "detail":      ensemble_detail,
        },
        "model_info": {
            "architecture":       "Ensemble (DenseNet121 + ResNet50 + EfficientNet-B0)",
            "validation_metrics": get_validation_metrics(disease),
        },
        "processing_time_ms": processing_ms,
        "roi_used": "manual" if points else "automatic",
    }

    if gradcam_url:
        response["gradcam"] = {"available": True, "heatmap_image": gradcam_url}

    if run_yolo_flag:
        response["yolo"] = {
            "available":  yolo_url is not None,
            "image_url":  yolo_url,
            "detections": detections,
            "total":      len(detections),
        }

    if run_seg_flag:
        response["segmentation"] = {
            "available": seg_url is not None,
            "image_url": seg_url,
        }

    return response


@router.post("/predict/pneumonia")
async def predict_pneumonia(
    file:        UploadFile = File(...),
    crop_points: str        = Query(None),
    run_yolo:    bool       = Query(True),
    run_seg:     bool       = Query(True),
    db:          Session    = Depends(get_db),
    current_user: User      = Depends(get_current_user),
    _rate:       None       = Depends(rate_limit_predict),
):
    try:
        return await _predict(
            file, "pneumonia", DiseaseType.PNEUMONIA,
            ["NORMAL", "PNEUMONIA"], settings.PNEUMONIA_THRESHOLD,
            crop_points, run_yolo, run_seg, db, current_user
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"predict/pneumonia: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/covid")
async def predict_covid(
    file:        UploadFile = File(...),
    crop_points: str        = Query(None),
    run_yolo:    bool       = Query(True),
    run_seg:     bool       = Query(True),
    db:          Session    = Depends(get_db),
    current_user: User      = Depends(get_current_user),
    _rate:       None       = Depends(rate_limit_predict),
):
    try:
        return await _predict(
            file, "covid", DiseaseType.COVID,
            ["Normal", "COVID"], settings.COVID_THRESHOLD,
            crop_points, run_yolo, run_seg, db, current_user
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"predict/covid: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/tuberculosis")
async def predict_tuberculosis(
    file:        UploadFile = File(...),
    crop_points: str        = Query(None),
    run_yolo:    bool       = Query(True),
    run_seg:     bool       = Query(True),
    db:          Session    = Depends(get_db),
    current_user: User      = Depends(get_current_user),
    _rate:       None       = Depends(rate_limit_predict),
):
    try:
        return await _predict(
            file, "tuberculosis", DiseaseType.TUBERCULOSIS,
            ["Normal", "Tuberculosis"], settings.TB_THRESHOLD,
            crop_points, run_yolo, run_seg, db, current_user
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"predict/tuberculosis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/auto")
async def predict_auto(
    file:          UploadFile = File(...),
    run_yolo_flag: bool       = Query(True,  alias="run_yolo"),
    run_seg_flag:  bool       = Query(True,  alias="run_seg"),
    db:            Session    = Depends(get_db),
    current_user:  User       = Depends(get_current_user),
    _rate:         None       = Depends(rate_limit_predict),
):
    """Analiza las 3 enfermedades automáticamente y devuelve ranking."""
    try:
        start    = time.time()
        contents = await file.read()

        validate_image_file(contents, file.filename, allow_dicom=False)

        tensor, original_image = preprocess_image(contents)

        configs = [
            ('pneumonia',    ['NORMAL', 'PNEUMONIA'],    DiseaseType.PNEUMONIA),
            ('covid',        ['Normal', 'COVID'],         DiseaseType.COVID),
            ('tuberculosis', ['Normal', 'Tuberculosis'],  DiseaseType.TUBERCULOSIS),
        ]

        results = {}
        for disease, class_names, _ in configs:
            result = ensemble_predict(disease, tensor)
            probs_avg       = result['probs_avg']
            predicted_class = result['predicted_class']
            results[disease] = {
                'label':              class_names[predicted_class],
                'confidence':         result['confidence'],
                'is_positive':        predicted_class == 1,
                'probability_disease':float(probs_avg[1]),
                'probabilities':      {class_names[i]: float(probs_avg[i]) for i in range(len(class_names))},
                'ensemble_detail':    result['detail'],
            }

        ranking = sorted(
            results.items(),
            key=lambda x: x[1]['probability_disease'],
            reverse=True
        )

        top_disease = ranking[0][0]
        top_result  = ranking[0][1]

        # grad-cam solo de la enfermedad más probable
        gradcam_url   = None
        heatmap_bytes = generate_gradcam(
            top_disease, tensor, original_image,
            1 if top_result['is_positive'] else 0
        )
        if heatmap_bytes:
            gradcam_url = records.storage_service.upload_image(
                heatmap_bytes, _unique_name("gradcam"), "gradcam"
            )

        yolo_url   = None
        detections = []
        if run_yolo_flag and inference.yolo_model is not None:
            yolo_bytes, detections = run_yolo(contents)
            yolo_url = records.storage_service.upload_image(
                yolo_bytes, _unique_name("yolo"), "yolo"
            )

        seg_url = None
        if run_seg_flag:
            seg_bytes = run_segmentation(contents)
            seg_url   = records.storage_service.upload_image(
                seg_bytes, _unique_name("seg"), "segmentation"
            )

        top_name   = top_disease.upper()
        top_conf   = top_result['confidence'] * 100
        conclusion = f"Mayor probabilidad de hallazgo: {top_name} ({top_conf:.1f}%)."
        if yolo_url:
            conclusion += f" YOLO detectó {len(detections)} hallazgo(s) en la radiografía."

        processing_ms = (time.time() - start) * 1000
        clinical      = get_clinical_interpretation(top_result['confidence'], top_disease)

        disease_enum_map = {
            'pneumonia':    DiseaseType.PNEUMONIA,
            'covid':        DiseaseType.COVID,
            'tuberculosis': DiseaseType.TUBERCULOSIS,
        }

        study, pred = _save_study_and_prediction(
            db, current_user, contents, file.filename,
            disease_enum_map[top_disease],
            top_disease, top_result['label'], top_result['confidence'],
            top_result['probabilities'], top_result['ensemble_detail'],
            processing_ms,
            gradcam_url, yolo_url, seg_url, clinical, None
        )

        return {
            "status":    "success",
            "mode":      "auto",
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "study_id":  str(study.id),
            "ranking":   [
                {"disease": d, "rank": i+1, **r}
                for i, (d, r) in enumerate(ranking)
            ],
            "conclusion":  conclusion,
            "gradcam":     {"available": True, "heatmap_image": gradcam_url,
                            "analyzed_disease": top_disease},
            "yolo":        {"available": yolo_url is not None, "image_url": yolo_url,
                            "detections": detections, "total": len(detections)},
            "segmentation":{"image_url": seg_url},
            "processing_time_ms": processing_ms,
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"predict/auto: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/dicom")
async def predict_dicom(
    file:          UploadFile = File(...),
    run_yolo_flag: bool       = Query(True, alias="run_yolo"),
    run_seg_flag:  bool       = Query(True, alias="run_seg"),
    db:            Session    = Depends(get_db),
    current_user:  User       = Depends(get_current_user),
    _rate:         None       = Depends(rate_limit_predict),
):
    if not file.filename.lower().endswith(".dcm"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .dcm")

    try:
        contents = await file.read()
        start    = time.time()

        validate_image_file(contents, file.filename, allow_dicom=True)

        pil_image, dicom_metadata = dicom_to_pil(contents)

        dicom_path = Path("uploads/dicom") / file.filename
        dicom_path.parent.mkdir(parents=True, exist_ok=True)
        dicom_path.write_bytes(contents)

        # el resto del pipeline trabaja con bytes de imagen, no con objetos pil
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        tensor, original_image = preprocess_image(png_bytes)

        configs = [
            ('pneumonia',    ['NORMAL', 'PNEUMONIA'],    DiseaseType.PNEUMONIA),
            ('covid',        ['Normal', 'COVID'],         DiseaseType.COVID),
            ('tuberculosis', ['Normal', 'Tuberculosis'],  DiseaseType.TUBERCULOSIS),
        ]

        results_dict = {}
        for disease, class_names, _ in configs:
            result    = ensemble_predict(disease, tensor)
            probs_avg = result['probs_avg']
            predicted_class = result['predicted_class']
            results_dict[disease] = {
                'label':              class_names[predicted_class],
                'confidence':         result['confidence'],
                'is_positive':        predicted_class == 1,
                'probability_disease':float(probs_avg[1]),
                'probabilities':      {class_names[i]: float(probs_avg[i]) for i in range(len(class_names))},
                'ensemble_detail':    result['detail'],
            }

        ranking = sorted(
            results_dict.items(),
            key=lambda x: x[1]['probability_disease'],
            reverse=True
        )
        ranking_list = [
            {"disease": d, "rank": i + 1, **r}
            for i, (d, r) in enumerate(ranking)
        ]

        top_disease = ranking[0][0]
        top_result  = ranking[0][1]

        gradcam_url   = None
        heatmap_bytes = generate_gradcam(
            top_disease, tensor, original_image,
            1 if top_result['is_positive'] else 0
        )
        if heatmap_bytes:
            gradcam_url = records.storage_service.upload_image(
                heatmap_bytes, _unique_name("gradcam"), "gradcam"
            )

        yolo_url   = None
        detections = []
        if run_yolo_flag and inference.yolo_model is not None:
            yolo_bytes, detections = run_yolo(png_bytes)
            yolo_url = records.storage_service.upload_image(
                yolo_bytes, _unique_name("yolo"), "yolo"
            )

        seg_url = None
        if run_seg_flag:
            seg_bytes = run_segmentation(png_bytes)
            seg_url   = records.storage_service.upload_image(
                seg_bytes, _unique_name("seg"), "segmentation"
            )

        clinical = get_clinical_interpretation(top_result['confidence'], top_disease)

        disease_enum_map = {
            'pneumonia':    DiseaseType.PNEUMONIA,
            'covid':        DiseaseType.COVID,
            'tuberculosis': DiseaseType.TUBERCULOSIS,
        }

        study, pred = _save_study_and_prediction(
            db, current_user, png_bytes, file.filename,
            disease_enum_map[top_disease],
            top_disease, top_result['label'], top_result['confidence'],
            top_result['probabilities'], top_result['ensemble_detail'],
            (time.time() - start) * 1000,
            gradcam_url, yolo_url, seg_url, clinical, None
        )

        display_meta = {
            **dicom_metadata,
            "study_date_formatted": format_dicom_date(dicom_metadata.get("study_date", "")),
            "file_name": file.filename,
        }

        conclusion = (
            f"Mayor probabilidad: {top_result['label']} "
            f"({top_result['probability_disease']*100:.1f}%). "
            + (f"YOLO detectó {len(detections)} hallazgo(s). " if yolo_url else "")
            + "Metadatos DICOM extraídos correctamente."
        )

        return {
            "status":         "success",
            "mode":           "dicom",
            "study_id":       str(study.id),
            "dicom_metadata": display_meta,
            "ranking":        ranking_list,
            "conclusion":     conclusion,
            "gradcam":        {"heatmap_image": gradcam_url, "analyzed_disease": top_disease},
            "yolo":           {"available": yolo_url is not None, "image_url": yolo_url,
                               "detections": detections, "total": len(detections)},
            "segmentation":   {"image_url": seg_url},
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"predict/dicom: {e}")
        raise HTTPException(status_code=500, detail=str(e))
