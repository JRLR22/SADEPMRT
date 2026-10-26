# ============================================================================
# MAIN.PY - SISTEMA COMPLETO LOCAL
# Ensemble (DenseNet + ResNet + EfficientNet) + YOLO + U-Net + Grad-CAM
# Sin Supabase — todo en SQLite + disco local
# ============================================================================

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
import time, uuid, json, cv2, os, shutil
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import pydicom
import io
from dicom_utils import dicom_to_pil as _dicom_to_pil_util, format_dicom_date
import json
import collections
import threading

from config import settings, validate_config
from gradcam import create_gradcam_for_densenet
from app.database.base import get_db
from app.models.user import User
from app.models.study import Study, Prediction, DiseaseType, AuditLog
from app.api.auth import router as auth_router, get_current_user
from app.services.storage import StorageService

from sqlalchemy import text

# ============================================================================
# CONSTANTES DE VALIDACIÓN
# ============================================================================

MAX_FILE_SIZE_MB    = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_MIME_TYPES  = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}
ALLOWED_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".dcm"}

# ============================================================================
# RATE LIMITER (in-memory, thread-safe, sliding window)
# ============================================================================

class RateLimiter:
    """
    Sliding-window rate limiter sin dependencias externas.
    Almacena timestamps de requests por clave (user_id o IP).
    """
    def __init__(self):
        self._store: dict[str, collections.deque] = {}
        self._lock  = threading.Lock()

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> tuple[bool, int]:
        """
        Retorna (allowed, retry_after_seconds).
        retry_after_seconds es 0 si allowed=True.
        """
        now = time.time()
        cutoff = now - window_seconds

        with self._lock:
            if key not in self._store:
                self._store[key] = collections.deque()

            dq = self._store[key]

            # Eliminar entradas fuera de la ventana
            while dq and dq[0] < cutoff:
                dq.popleft()

            if len(dq) >= max_requests:
                oldest = dq[0]
                retry_after = int(window_seconds - (now - oldest)) + 1
                return False, retry_after

            dq.append(now)
            return True, 0

    def cleanup(self):
        """Limpia claves inactivas (llamar periódicamente si el servidor es long-running)."""
        now = time.time()
        with self._lock:
            dead = [k for k, dq in self._store.items() if not dq or dq[-1] < now - 3600]
            for k in dead:
                del self._store[k]


_rate_limiter = RateLimiter()

# Límites configurables por tipo de operación
RATE_LIMITS = {
    "predict":    {"max_requests": 30,  "window_seconds": 3600},   # 30 predicciones/hora
    "report":     {"max_requests": 20,  "window_seconds": 3600},   # 20 reportes/hora
    "studies":    {"max_requests": 200, "window_seconds": 3600},   # 200 lecturas/hora
    "validate":   {"max_requests": 100, "window_seconds": 3600},   # 100 validaciones/hora
}


def check_rate_limit(request: Request, current_user: User, operation: str = "predict"):
    """
    Dependency que lanza 429 si el usuario superó su límite.
    Usa user_id como clave para que sea por cuenta, no por IP.
    """
    cfg = RATE_LIMITS.get(operation, RATE_LIMITS["predict"])
    key = f"{operation}:{current_user.id}"

    allowed, retry_after = _rate_limiter.is_allowed(
        key, cfg["max_requests"], cfg["window_seconds"]
    )

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error":        "rate_limit_exceeded",
                "message":      f"Límite de {cfg['max_requests']} requests/{cfg['window_seconds']//3600}h superado.",
                "retry_after":  retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )


# Factories para los distintos límites (necesario para Depends)
def rate_limit_predict(request: Request, current_user: User = Depends(get_current_user)):
    check_rate_limit(request, current_user, "predict")

def rate_limit_report(request: Request, current_user: User = Depends(get_current_user)):
    check_rate_limit(request, current_user, "report")

def rate_limit_studies(request: Request, current_user: User = Depends(get_current_user)):
    check_rate_limit(request, current_user, "studies")

def rate_limit_validate(request: Request, current_user: User = Depends(get_current_user)):
    check_rate_limit(request, current_user, "validate")


# ============================================================================
# VALIDACIÓN DE ARCHIVOS
# ============================================================================

def validate_image_file(contents: bytes, filename: str, allow_dicom: bool = False):
    """
    Valida tamaño, extensión y firma mágica del archivo.
    Lanza HTTPException 400 si algo falla.
    """
    # Tamaño
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Archivo demasiado grande. Máximo {MAX_FILE_SIZE_MB} MB."
        )

    if len(contents) < 8:
        raise HTTPException(status_code=400, detail="Archivo demasiado pequeño o vacío.")

    # Extensión
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Extensión '{ext}' no permitida. Permitidas: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # DICOM no necesita validación de imagen PIL
    if ext == ".dcm":
        if not allow_dicom:
            raise HTTPException(status_code=400, detail="DICOM no permitido en este endpoint.")
        return

    # Firma mágica (magic bytes) para imágenes
    magic_signatures = {
        b'\xff\xd8\xff':       "JPEG",
        b'\x89PNG\r\n\x1a\n': "PNG",
        b'BM':                  "BMP",
        b'II*\x00':             "TIFF (little-endian)",
        b'MM\x00*':             "TIFF (big-endian)",
        b'RIFF':                "WEBP",
    }

    detected = None
    for sig, fmt in magic_signatures.items():
        if contents[:len(sig)] == sig:
            detected = fmt
            break

    if detected is None:
        raise HTTPException(
            status_code=400,
            detail="El archivo no parece ser una imagen válida (firma no reconocida)."
        )

    # Intentar abrir con PIL como validación final
    try:
        img = Image.open(BytesIO(contents))
        img.verify()   # detecta archivos truncados/corruptos
    except Exception:
        raise HTTPException(status_code=400, detail="El archivo de imagen está corrupto o es inválido.")


# ============================================================================
# ESTADO GLOBAL
# ============================================================================

ensemble_models: dict = {}   # {disease: {densenet, resnet, efficient}}
yolo_model              = None
unet_model              = None
storage_service         = None

# ============================================================================
# CARGA DE MODELOS
# ============================================================================

def _load_densenet(path: Path, num_classes: int, device):
    model = tv_models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval()


def _load_resnet(path: Path, num_classes: int, device):
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval()


def _load_efficientnet(path: Path, num_classes: int, device):
    from efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval()


def _load_unet(path: Path, device):
    import segmentation_models_pytorch as smp
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation=None
    )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval()

# ============================================================================
# CICLO DE VIDA
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ensemble_models, yolo_model, unet_model, storage_service

    logger.info("🚀 Iniciando aplicación...")

    validate_config()

    storage_service = StorageService()
    device = torch.device(settings.DEVICE)
    logger.info(f"   Device: {device}")

    # ── Ensemble clasificación ───────────────────────────────────────────────
    configs = {
        'pneumonia': {
            'num_classes': 2,
            'densenet':  settings.PNEUMONIA_DENSENET_PATH,
            'resnet':    settings.PNEUMONIA_RESNET_PATH,
            'efficient': settings.PNEUMONIA_EFFICIENT_PATH,
        },
        'covid': {
            'num_classes': 2,
            'densenet':  settings.COVID_DENSENET_PATH,
            'resnet':    settings.COVID_RESNET_PATH,
            'efficient': settings.COVID_EFFICIENT_PATH,
        },
        'tuberculosis': {
            'num_classes': 2,
            'densenet':  settings.TB_DENSENET_PATH,
            'resnet':    settings.TB_RESNET_PATH,
            'efficient': settings.TB_EFFICIENT_PATH,
        },
    }

    for disease, cfg in configs.items():
        nc = cfg['num_classes']
        logger.info(f"   • Cargando ensemble {disease}...")
        ensemble_models[disease] = {
            'densenet':  _load_densenet(cfg['densenet'],  nc, device),
            'resnet':    _load_resnet(cfg['resnet'],    nc, device),
            'efficient': _load_efficientnet(cfg['efficient'], nc, device),
        }

    # ── YOLOv8 ──────────────────────────────────────────────────────────────
    logger.info("   • Cargando YOLOv8...")
    from ultralytics import YOLO
    yolo_model = YOLO(str(settings.YOLO_MODEL_PATH))

    # ── U-Net ────────────────────────────────────────────────────────────────
    logger.info("   • Cargando U-Net...")
    unet_model = _load_unet(settings.UNET_MODEL_PATH, device)

    logger.success("✅ Todos los modelos cargados")
    yield

    ensemble_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _rate_limiter.cleanup()

# ============================================================================
# APP
# ============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ============================================================================
# UTILIDADES
# ============================================================================

TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

UNET_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])


def _unique_name(prefix: str, ext: str = "png") -> str:
    """Genera un nombre de archivo único con timestamp + uuid."""
    ts  = datetime.utcnow().strftime("%Y%d%m_%H%M%S")
    uid = str(uuid.uuid4())[:8]
    return f"{prefix}_{ts}_{uid}.{ext}"


def preprocess_image(image_bytes: bytes):
    """Preprocesa la imagen para clasificación."""
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    tensor = TRANSFORM(image).unsqueeze(0)
    return tensor, image


def ensemble_predict(disease: str, tensor: torch.Tensor) -> dict:
    """
    Promedia las probabilidades de los 3 modelos (soft voting).
    Retorna: {probs_avg, predicted_class, confidence, detail}
    """
    device     = torch.device(settings.DEVICE)
    tensor     = tensor.to(device)
    models     = ensemble_models[disease]
    detail     = {}
    probs_list = []

    with torch.no_grad():
        for name, model in models.items():
            out   = model(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            probs_list.append(probs)
            detail[name] = probs.tolist()

    probs_avg       = np.mean(probs_list, axis=0)
    predicted_class = int(probs_avg.argmax())
    confidence      = float(probs_avg[predicted_class])

    return {
        'probs_avg':       probs_avg,
        'predicted_class': predicted_class,
        'confidence':      confidence,
        'detail':          detail,
    }


def generate_gradcam(disease: str, tensor: torch.Tensor, original_image: Image.Image,
                     predicted_class: int) -> bytes | None:
    """Genera Grad-CAM usando el DenseNet del ensemble."""
    try:
        model   = ensemble_models[disease]['densenet']
        gradcam = create_gradcam_for_densenet(model)
        cam     = gradcam.generate_cam(tensor.to(settings.DEVICE), target_class=predicted_class)
        heat    = gradcam.overlay_heatmap(original_image, cam, alpha=0.5)
        buf     = BytesIO()
        heat.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Grad-CAM error: {e}")
        return None


def run_yolo(image_bytes: bytes) -> tuple[bytes, list]:
    """
    Corre YOLOv8 sobre la imagen.
    Retorna (imagen_anotada_bytes, detecciones_list).
    """
    img_array = np.frombuffer(image_bytes, np.uint8)
    img_bgr   = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    results   = yolo_model(img_bgr, verbose=False)
    annotated = results[0].plot()

    detections = []
    for box in results[0].boxes:
        detections.append({
            'class_id':   int(box.cls),
            'class_name': yolo_model.names[int(box.cls)],
            'confidence': float(box.conf),
            'bbox':       box.xyxy[0].tolist(),
        })

    _, buf = cv2.imencode('.png', annotated)
    return buf.tobytes(), detections


def run_segmentation(image_bytes: bytes) -> bytes:
    """
    Corre U-Net sobre la imagen.
    Retorna máscara superpuesta como PNG bytes.
    """
    device = torch.device(settings.DEVICE)
    image  = Image.open(BytesIO(image_bytes)).convert('RGB')
    tensor = UNET_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(unet_model(tensor))
        mask = (pred > 0.5).float().cpu().squeeze().numpy()

    orig     = np.array(image.resize((512, 512)))
    green    = np.zeros_like(orig)
    green[:, :, 1] = 255
    mask_3ch = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)
    overlay  = cv2.addWeighted(orig, 0.7, cv2.bitwise_and(green, mask_3ch), 0.3, 0)

    _, buf = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def dicom_to_pil(dicom_bytes: bytes) -> tuple:
    """
    Convierte DICOM a imagen PIL + extrae metadatos.
    Delegado a dicom_utils para evitar duplicación.
    """
    return _dicom_to_pil_util(dicom_bytes)


def get_clinical_interpretation(confidence: float, disease: str) -> dict:
    thresh = {
        'tuberculosis': {'high': 0.90, 'medium': 0.75},
        'pneumonia':    {'high': 0.85, 'medium': 0.70},
        'covid':        {'high': 0.85, 'medium': 0.70},
    }.get(disease, {'high': 0.85, 'medium': 0.70})

    if confidence >= thresh['high']:
        return {
            'level':           'Alto',
            'color':           '#22c55e',
            'action_required': False,
            'message':         'Alta confianza diagnóstica.',
            'recommendation':  'Correlación clínica recomendada.',
        }
    elif confidence >= thresh['medium']:
        return {
            'level':           'Medio',
            'color':           '#f59e0b',
            'action_required': True,
            'message':         'Confianza moderada.',
            'recommendation':  'Evaluación detallada recomendada.',
        }
    else:
        return {
            'level':           'Bajo',
            'color':           '#ef4444',
            'action_required': True,
            'message':         'Baja confianza.',
            'recommendation':  '⚠️ Revisión manual obligatoria.',
        }


def get_validation_metrics(disease: str) -> dict:
    return {
        'pneumonia':    {'accuracy': '92.75%', 'auc': '0.9720', 'f1': '0.93'},
        'covid':        {'accuracy': '98.50%', 'auc': '0.9910', 'f1': '0.98'},
        'tuberculosis': {'accuracy': '97.80%', 'auc': '0.9850', 'f1': '0.97'},
    }.get(disease, {})


def _log_audit(db: Session, user_id, action: str, resource_type: str, metadata: dict):
    """Helper centralizado para audit logs."""
    audit = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        metadata=json.dumps(metadata),
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

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    db_status = "ok"
    try:
        from app.database.base import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {str(e)}"

    models_status = {
        "pneumonia":    "pneumonia"    in ensemble_models,
        "covid":        "covid"        in ensemble_models,
        "tuberculosis": "tuberculosis" in ensemble_models,
        "yolo":         yolo_model is not None,
        "unet":         unet_model is not None,
    }
    all_models_ok = all(models_status.values())
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


# ── Helper interno para predict ─────────────────────────────────────────────

async def _predict(
    file, disease: str, disease_enum, class_names: list,
    threshold: float, crop_points, run_yolo_flag: bool,
    run_seg_flag: bool, db, current_user
):
    start    = time.time()
    contents = await file.read()

    # Validar archivo
    validate_image_file(contents, file.filename, allow_dicom=False)

    points = json.loads(crop_points) if crop_points else None

    tensor, original_image = preprocess_image(contents)

    # Ensemble
    result          = ensemble_predict(disease, tensor)
    probs_avg       = result['probs_avg']
    predicted_class = result['predicted_class']
    confidence      = result['confidence']
    ensemble_detail = result['detail']

    prediction_label = class_names[predicted_class]
    probabilities    = {class_names[i]: float(probs_avg[i]) for i in range(len(class_names))}
    processing_ms    = (time.time() - start) * 1000

    # Grad-CAM
    gradcam_url   = None
    heatmap_bytes = generate_gradcam(disease, tensor, original_image, predicted_class)
    if heatmap_bytes:
        gradcam_url = storage_service.upload_image(
            heatmap_bytes, _unique_name("gradcam"), "gradcam"
        )

    # YOLO
    yolo_url   = None
    detections = []
    if run_yolo_flag:
        yolo_bytes, detections = run_yolo(contents)
        yolo_url = storage_service.upload_image(
            yolo_bytes, _unique_name("yolo"), "yolo"
        )

    # Segmentación U-Net
    seg_url = None
    if run_seg_flag:
        seg_bytes = run_segmentation(contents)
        seg_url   = storage_service.upload_image(
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


# ── Endpoints de predicción ──────────────────────────────────────────────────

@app.post("/predict/pneumonia")
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
        logger.error(f"❌ predict/pneumonia: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/covid")
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
        logger.error(f"❌ predict/covid: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/tuberculosis")
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
        logger.error(f"❌ predict/tuberculosis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Estudios ─────────────────────────────────────────────────────────────────

@app.get("/studies")
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


@app.get("/studies/{study_id}")
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


@app.put("/predictions/{prediction_id}/validate")
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


# ── Reporte PDF ───────────────────────────────────────────────────────────────

@app.get("/studies/{study_id}/report")
async def generate_pdf_report(
    study_id:     str,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
    _rate:        None    = Depends(rate_limit_report),
):
    try:
        from sqlalchemy.orm import joinedload
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle,
            Paragraph, Spacer, Image as RLImage,
            HRFlowable, KeepTogether
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        from reportlab.pdfgen import canvas as rl_canvas

        # ── Colores del sistema ─────────────────────────────────────────────
        BLUE       = colors.HexColor('#1a56db')
        BLUE_LT    = colors.HexColor('#eff4ff')
        BLUE_MID   = colors.HexColor('#c7d7f9')
        RED        = colors.HexColor('#e02424')
        RED_LT     = colors.HexColor('#fef2f2')
        GREEN      = colors.HexColor('#0d9f6e')
        GREEN_LT   = colors.HexColor('#ecfdf5')
        GRAY_TEXT  = colors.HexColor('#4a5e7a')
        GRAY_DIM   = colors.HexColor('#8fa3bf')
        GRAY_BG    = colors.HexColor('#f8fafc')
        DARK       = colors.HexColor('#0f1f38')
        WHITE      = colors.white
        BORDER     = colors.HexColor('#e2e8f0')

        # ── Carga del estudio ───────────────────────────────────────────────
        s = (
            db.query(Study)
            .options(joinedload(Study.predictions))
            .filter(Study.id == study_id)
            .first()
        )
        if not s:
            raise HTTPException(status_code=404, detail="Estudio no encontrado")
        if s.uploaded_by != current_user.id and current_user.role not in ['admin', 'radiologist']:
            raise HTTPException(status_code=403, detail="Sin permiso")
        if not s.predictions:
            raise HTTPException(status_code=400, detail="Sin predicciones")

        _log_audit(db, current_user.id, "report", "study", {'study_id': study_id})
        db.commit()

        pred     = s.predictions[0]
        probs    = json.loads(pred.probabilities)           if pred.probabilities           else {}
        clinical = json.loads(pred.clinical_interpretation) if pred.clinical_interpretation else {}

        is_normal     = 'normal' in pred.prediction.lower()
        result_color  = GREEN    if is_normal else RED
        result_bg     = GREEN_LT if is_normal else RED_LT
        result_label  = 'NORMAL — Sin hallazgos patológicos' if is_normal else f'POSITIVO — {pred.prediction.upper()}'

        disease_display = {
            'COVID':        'COVID-19',
            'PNEUMONIA':    'Neumonía',
            'TUBERCULOSIS': 'Tuberculosis',
        }.get(str(s.disease_type).upper().replace('DISEASETYPE.', ''), str(s.disease_type))

        # Hora local Mazatlán para el reporte impreso
        from datetime import timezone
        import pytz
        mzt    = pytz.timezone('America/Mazatlan')
        fecha  = s.created_at.replace(tzinfo=timezone.utc).astimezone(mzt)
        fecha_str  = fecha.strftime('%d/%m/%Y %H:%M')
        gen_str    = datetime.now(mzt).strftime('%d/%m/%Y %H:%M:%S')

        def url_to_path(url: str):
            if not url:
                return None
            for marker in ['localhost:8000/uploads/', '127.0.0.1:8000/uploads/']:
                if marker in url:
                    relative = url.split(marker)[-1]
                    return os.path.join('uploads', relative.replace('/', os.sep))
            # Ruta relativa directa
            if url.startswith('uploads/'):
                return url
            return None

        # ── Clases de canvas para header/footer ────────────────────────────
        class HeaderFooterCanvas(rl_canvas.Canvas):
            """Dibuja header azul y footer en cada página."""
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.pages = []

            def showPage(self):
                self.pages.append(dict(self.__dict__))
                self._startPage()

            def save(self):
                page_count = len(self.pages)
                for i, page in enumerate(self.pages):
                    self.__dict__.update(page)
                    self._draw_header_footer(i + 1, page_count)
                    super().showPage()
                super().save()

            def _draw_header_footer(self, page_num, page_count):
                w, h = letter

                # ── Barra azul superior ──
                self.setFillColor(BLUE)
                self.rect(0, h - 0.55 * inch, w, 0.55 * inch, fill=1, stroke=0)

                # Nombre del sistema en la barra
                self.setFillColor(WHITE)
                self.setFont('Helvetica-Bold', 11)
                self.drawString(0.5 * inch, h - 0.35 * inch, 'rt26 — Sistema de Detección Pulmonar IA v2.0')

                # Nro de página derecha
                self.setFont('Helvetica', 9)
                self.drawRightString(w - 0.5 * inch, h - 0.35 * inch, f'Página {page_num} de {page_count}')

                # ── Línea decorativa azul claro bajo el header ──
                self.setStrokeColor(BLUE_MID)
                self.setLineWidth(0.5)
                self.line(0.5 * inch, h - 0.6 * inch, w - 0.5 * inch, h - 0.6 * inch)

                # ── Footer ──
                self.setStrokeColor(BORDER)
                self.setLineWidth(0.5)
                self.line(0.5 * inch, 0.55 * inch, w - 0.5 * inch, 0.55 * inch)

                self.setFillColor(GRAY_DIM)
                self.setFont('Helvetica', 8)
                self.drawString(0.5 * inch, 0.35 * inch,
                    f'Generado: {gen_str}  •  Este reporte es una herramienta de apoyo — no sustituye criterio médico')
                self.drawRightString(w - 0.5 * inch, 0.35 * inch,
                    f'Estudio ID: {str(s.id)[:8].upper()}')

        # ── Documento ──────────────────────────────────────────────────────
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.6 * inch,
            leftMargin=0.6 * inch,
            topMargin=1.0 * inch,      # espacio para el header azul
            bottomMargin=0.85 * inch,  # espacio para el footer
        )

        # ── Estilos ────────────────────────────────────────────────────────
        styles = getSampleStyleSheet()

        def section_header(text):
            """Retorna la fila de encabezado de sección con fondo azul."""
            t = Table([[Paragraph(
                f'<font color="white"><b>{text}</b></font>',
                ParagraphStyle('sh', fontName='Helvetica-Bold', fontSize=10,
                               textColor=WHITE, leading=14)
            )]], colWidths=[6.8 * inch])
            t.setStyle(TableStyle([
                ('BACKGROUND',    (0,0),(-1,-1), BLUE),
                ('TOPPADDING',    (0,0),(-1,-1), 7),
                ('BOTTOMPADDING', (0,0),(-1,-1), 7),
                ('LEFTPADDING',   (0,0),(-1,-1), 10),
                ('RIGHTPADDING',  (0,0),(-1,-1), 10),
                ('ROUNDEDCORNERS',(0,0),(-1,-1), [4, 4, 0, 0]),
            ]))
            return t

        def info_table(rows, col_widths=None):
            """Tabla de datos con filas alternadas."""
            cw = col_widths or [1.8 * inch, 5.0 * inch]
            t  = Table(rows, colWidths=cw)
            style = [
                ('FONTNAME',      (0,0),(0,-1), 'Helvetica-Bold'),
                ('FONTNAME',      (1,0),(1,-1), 'Helvetica'),
                ('FONTSIZE',      (0,0),(-1,-1), 9),
                ('TEXTCOLOR',     (0,0),(0,-1), DARK),
                ('TEXTCOLOR',     (1,0),(1,-1), GRAY_TEXT),
                ('TOPPADDING',    (0,0),(-1,-1), 6),
                ('BOTTOMPADDING', (0,0),(-1,-1), 6),
                ('LEFTPADDING',   (0,0),(-1,-1), 10),
                ('RIGHTPADDING',  (0,0),(-1,-1), 10),
                ('GRID',          (0,0),(-1,-1), 0.5, BORDER),
                ('ROWBACKGROUNDS',(0,0),(-1,-1), [WHITE, GRAY_BG]),
            ]
            t.setStyle(TableStyle(style))
            return t

        sp = lambda n: Spacer(1, n)

        # ── Contenido ──────────────────────────────────────────────────────
        elements = []

        # ── Título principal ────────────────────────────────────────────────
        elements.append(sp(0.15 * inch))
        elements.append(Paragraph(
            'REPORTE DE ANÁLISIS RADIOLÓGICO',
            ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=18,
                           textColor=DARK, alignment=TA_CENTER, spaceAfter=4,
                           letterSpacing=0.5)
        ))
        elements.append(Paragraph(
            'Diagnóstico asistido por IA — Ensemble de 3 modelos de Deep Learning',
            ParagraphStyle('sub', fontName='Helvetica', fontSize=10,
                           textColor=GRAY_DIM, alignment=TA_CENTER)
        ))
        elements.append(sp(0.25 * inch))

        # ── RESULTADO DESTACADO (cajón grande arriba) ───────────────────────
        conf_pct = f'{pred.confidence * 100:.1f}%'
        risk_lvl = clinical.get('level', '—')
        risk_msg = clinical.get('recommendation', '—')

        result_table = Table(
            [[
                Paragraph(
                    f'<font size="11" color="{result_color.hexval() if hasattr(result_color,"hexval") else "#0d9f6e"}"><b>DIAGNÓSTICO</b></font><br/>'
                    f'<font size="16"><b>{result_label}</b></font>',
                    ParagraphStyle('res', fontName='Helvetica', fontSize=16,
                                   textColor=result_color, leading=22)
                ),
                Paragraph(
                    f'<font size="9" color="#8fa3bf">Confianza del modelo</font><br/>'
                    f'<font size="22"><b>{conf_pct}</b></font>',
                    ParagraphStyle('conf', fontName='Helvetica-Bold', fontSize=22,
                                   textColor=result_color, alignment=TA_CENTER, leading=28)
                ),
            ]],
            colWidths=[4.5 * inch, 2.3 * inch]
        )
        result_table.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), result_bg),
            ('TOPPADDING',    (0,0),(-1,-1), 14),
            ('BOTTOMPADDING', (0,0),(-1,-1), 14),
            ('LEFTPADDING',   (0,0),(-1,-1), 14),
            ('RIGHTPADDING',  (0,0),(-1,-1), 14),
            ('ROUNDEDCORNERS',(0,0),(-1,-1), [6, 6, 6, 6]),
            ('BOX',           (0,0),(-1,-1), 1.5, result_color),
            ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
            ('LINEAFTER',     (0,0),(0,-1),  0.5, result_color),
        ]))
        elements.append(result_table)
        elements.append(sp(0.22 * inch))

        # ── INFORMACIÓN DEL ESTUDIO ─────────────────────────────────────────
        elements.append(section_header('INFORMACIÓN DEL ESTUDIO'))
        elements.append(info_table([
            ['ID del estudio:',  str(s.id)],
            ['Fecha del estudio:', fecha_str],
            ['Tipo de análisis:', disease_display],
            ['Paciente:',        s.patient_identifier or 'No especificado'],
            ['Modelo utilizado:', 'Ensemble (DenseNet121 + ResNet50 + EfficientNet-B0)'],
            ['Tiempo de proc.:', f'{pred.processing_time_ms:.0f} ms'],
        ]))
        elements.append(sp(0.22 * inch))

        # ── IMÁGENES (2x2 más grandes) ──────────────────────────────────────
        img_items = [
            (s.original_image_url,        'Radiografía Original'),
            (pred.gradcam_image_url,       'Grad-CAM (activación)'),
            (pred.yolo_image_url,          'YOLO (detección)'),
            (pred.segmentation_image_url,  'U-Net (segmentación)'),
        ]

        img_size   = 1.4 * inch
        label_style = ParagraphStyle('lbl', fontName='Helvetica-Bold', fontSize=8,
                                     textColor=DARK, alignment=TA_CENTER)
        note_style  = ParagraphStyle('note', fontName='Helvetica', fontSize=7,
                                     textColor=GRAY_DIM, alignment=TA_CENTER)

        def make_img_cell(url, label, note=''):
            path = url_to_path(url) if url else None
            if path and os.path.exists(path):
                img = RLImage(path, width=img_size, height=img_size)
                img.hAlign = 'CENTER'
                return [img, Paragraph(label, label_style),
                        Paragraph(note, note_style) if note else Spacer(1, 1)]
            else:
                placeholder = Table(
                    [[Paragraph('N/D', ParagraphStyle('nd', fontName='Helvetica',
                                fontSize=9, textColor=GRAY_DIM, alignment=TA_CENTER))]],
                    colWidths=[img_size], rowHeights=[img_size]
                )
                placeholder.setStyle(TableStyle([
                    ('BACKGROUND', (0,0),(-1,-1), GRAY_BG),
                    ('BOX',        (0,0),(-1,-1), 0.5, BORDER),
                    ('VALIGN',     (0,0),(-1,-1), 'MIDDLE'),
                ]))
                return [placeholder, Paragraph(label, label_style)]

        row1 = [make_img_cell(*img_items[0], 'Imagen sin procesar'),
                make_img_cell(*img_items[1], 'Rojo = mayor atencion IA')]
        row2 = [make_img_cell(*img_items[2], 'Bounding boxes'),
                make_img_cell(*img_items[3], 'Mascara de segmentacion')]

        def pair_to_table(left_cell, right_cell):
            col_w = (6.8 * inch) / 2
            t = Table(
                [[left_cell[0],  right_cell[0]],
                 [left_cell[1],  right_cell[1]],
                 [left_cell[2] if len(left_cell) > 2 else Spacer(1,1),
                  right_cell[2] if len(right_cell) > 2 else Spacer(1,1)]],
                colWidths=[col_w, col_w]
            )
            t.setStyle(TableStyle([
                ('ALIGN',         (0,0),(-1,-1), 'CENTER'),
                ('VALIGN',        (0,0),(  -1,0), 'BOTTOM'),
                ('TOPPADDING',    (0,0),(-1,-1), 4),
                ('BOTTOMPADDING', (0,0),(-1,-1), 2),
            ]))
            return t

        elements.append(section_header('IMÁGENES DEL ANÁLISIS'))
        elements.append(KeepTogether([
            sp(6),
            pair_to_table(row1[0], row1[1]),
            sp(8),
            pair_to_table(row2[0], row2[1]),
            sp(0.22 * inch),
        ]))

        # ── PROBABILIDADES ──────────────────────────────────────────────────
        elements.append(section_header('PROBABILIDADES POR CLASE'))
        elements.append(sp(4))

        prob_rows = [
            [
                Paragraph(f'<b>{label}</b>',
                          ParagraphStyle('pl', fontName='Helvetica-Bold', fontSize=9, textColor=DARK)),
                Paragraph(f'{p*100:.2f}%',
                          ParagraphStyle('pv', fontName='Helvetica-Bold', fontSize=9,
                                         textColor=result_color if 'normal' not in label.lower() else GRAY_TEXT,
                                         alignment=TA_CENTER)),
            ]
            for label, p in probs.items()
        ]

        pt = Table(prob_rows, colWidths=[5.1 * inch, 1.7 * inch])
        pt.setStyle(TableStyle([
            ('FONTSIZE',      (0,0),(-1,-1), 9),
            ('TOPPADDING',    (0,0),(-1,-1), 7),
            ('BOTTOMPADDING', (0,0),(-1,-1), 7),
            ('LEFTPADDING',   (0,0),(-1,-1), 12),
            ('RIGHTPADDING',  (0,0),(-1,-1), 12),
            ('GRID',          (0,0),(-1,-1), 0.5, BORDER),
            ('ROWBACKGROUNDS',(0,0),(-1,-1), [WHITE, GRAY_BG]),
            ('ALIGN',         (1,0),(1,-1),  'CENTER'),
        ]))
        elements.append(pt)
        elements.append(sp(0.22 * inch))

        # ── NIVEL DE CONFIANZA CLÍNICA ──────────────────────────────────────
        if clinical:
            lvl_color = {
                'Alto':  GREEN,
                'Medio': colors.HexColor('#d97706'),
                'Bajo':  RED,
            }.get(risk_lvl, GRAY_TEXT)

            clin_table = Table(
                [[
                    Paragraph(
                        f'<b>Nivel de confianza clínica: {risk_lvl}</b><br/>'
                        f'<font size="8" color="#4a5e7a">{clinical.get("message","")}</font>',
                        ParagraphStyle('cl', fontName='Helvetica', fontSize=10,
                                       textColor=lvl_color, leading=14)
                    ),
                    Paragraph(
                        f'<font size="8"><b>Recomendacion:</b></font><br/>'
                        f'<font size="8" color="#4a5e7a">{risk_msg}</font>',
                        ParagraphStyle('cr', fontName='Helvetica', fontSize=9,
                                       textColor=DARK, leading=13)
                    ),
                ]],
                colWidths=[3.4 * inch, 3.4 * inch]
            )
            clin_table.setStyle(TableStyle([
                ('BACKGROUND',    (0,0),(0,-1), BLUE_LT),
                ('BACKGROUND',    (1,0),(1,-1), GRAY_BG),
                ('TOPPADDING',    (0,0),(-1,-1), 10),
                ('BOTTOMPADDING', (0,0),(-1,-1), 10),
                ('LEFTPADDING',   (0,0),(-1,-1), 12),
                ('RIGHTPADDING',  (0,0),(-1,-1), 12),
                ('BOX',           (0,0),(-1,-1), 0.5, BLUE_MID),
                ('LINEBEFORE',    (1,0),(1,-1),  0.5, BLUE_MID),
                ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
            ]))
            elements.append(section_header('INTERPRETACIÓN CLÍNICA'))
            elements.append(sp(4))
            elements.append(clin_table)
            elements.append(sp(0.22 * inch))

        # ── AVISO LEGAL (inline, no página separada) ────────────────────────
        aviso = Table(
            [[Paragraph(
                '<b>AVISO LEGAL:</b> Este sistema es una herramienta de apoyo diagnóstico basada en '
                'inteligencia artificial. Los resultados <b>NO</b> sustituyen el criterio clínico de un '
                'radiólogo o médico certificado. Todo estudio debe ser revisado por un profesional de la '
                'salud antes de tomar decisiones clínicas. El sistema no está certificado bajo NOM-024-SSA3.',
                ParagraphStyle('av', fontName='Helvetica', fontSize=8,
                               textColor=colors.HexColor('#92400e'), leading=12,
                               alignment=TA_JUSTIFY)
            )]],
            colWidths=[6.8 * inch]
        )
        aviso.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), colors.HexColor('#fffbeb')),
            ('BOX',           (0,0),(-1,-1), 0.5, colors.HexColor('#d97706')),
            ('LEFTPADDING',   (0,0),(-1,-1), 12),
            ('RIGHTPADDING',  (0,0),(-1,-1), 12),
            ('TOPPADDING',    (0,0),(-1,-1), 10),
            ('BOTTOMPADDING', (0,0),(-1,-1), 10),
            ('ROUNDEDCORNERS',(0,0),(-1,-1), [4, 4, 4, 4]),
        ]))
        elements.append(aviso)

        # ── Build ───────────────────────────────────────────────────────────
        doc.build(elements, canvasmaker=HeaderFooterCanvas)

        buffer.seek(0)
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        return StreamingResponse(
            buffer,
            media_type='application/pdf',
            headers={'Content-Disposition': f'attachment; filename=reporte_{s.disease_type}_{ts}.pdf'}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'PDF error: {e}')
        raise HTTPException(status_code=500, detail=str(e))


# ── predict/auto ─────────────────────────────────────────────────────────────

@app.post("/predict/auto")
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

        # Grad-CAM del más probable
        gradcam_url   = None
        heatmap_bytes = generate_gradcam(
            top_disease, tensor, original_image,
            1 if top_result['is_positive'] else 0
        )
        if heatmap_bytes:
            gradcam_url = storage_service.upload_image(
                heatmap_bytes, _unique_name("gradcam"), "gradcam"
            )

        # YOLO
        yolo_url   = None
        detections = []
        if run_yolo_flag:
            yolo_bytes, detections = run_yolo(contents)
            yolo_url = storage_service.upload_image(
                yolo_bytes, _unique_name("yolo"), "yolo"
            )

        # Segmentación
        seg_url = None
        if run_seg_flag:
            seg_bytes = run_segmentation(contents)
            seg_url   = storage_service.upload_image(
                seg_bytes, _unique_name("seg"), "segmentation"
            )

        top_name   = top_disease.upper()
        top_conf   = top_result['confidence'] * 100
        conclusion = f"Mayor probabilidad de hallazgo: {top_name} ({top_conf:.1f}%). "
        conclusion += f"YOLO detectó {len(detections)} hallazgo(s) en la radiografía."

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
            "yolo":        {"image_url": yolo_url, "detections": detections,
                            "total": len(detections)},
            "segmentation":{"image_url": seg_url},
            "processing_time_ms": processing_ms,
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ predict/auto: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── predict/dicom ─────────────────────────────────────────────────────────────

@app.post("/predict/dicom")
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

        # Convertir DICOM → PIL
        pil_image, dicom_metadata = dicom_to_pil(contents)

        # Guardar DICOM original
        dicom_path = Path("uploads/dicom") / file.filename
        dicom_path.parent.mkdir(parents=True, exist_ok=True)
        dicom_path.write_bytes(contents)

        # PIL → PNG bytes para el pipeline
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

        # Grad-CAM
        gradcam_url   = None
        heatmap_bytes = generate_gradcam(
            top_disease, tensor, original_image,
            1 if top_result['is_positive'] else 0
        )
        if heatmap_bytes:
            gradcam_url = storage_service.upload_image(
                heatmap_bytes, _unique_name("gradcam"), "gradcam"
            )

        # YOLO
        yolo_url   = None
        detections = []
        if run_yolo_flag:
            yolo_bytes, detections = run_yolo(png_bytes)
            yolo_url = storage_service.upload_image(
                yolo_bytes, _unique_name("yolo"), "yolo"
            )

        # Segmentación
        seg_url = None
        if run_seg_flag:
            seg_bytes = run_segmentation(png_bytes)
            seg_url   = storage_service.upload_image(
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
            f"YOLO detectó {len(detections)} hallazgo(s). "
            "Metadatos DICOM extraídos correctamente."
        )

        return {
            "status":         "success",
            "mode":           "dicom",
            "study_id":       str(study.id),
            "dicom_metadata": display_meta,
            "ranking":        ranking_list,
            "conclusion":     conclusion,
            "gradcam":        {"heatmap_image": gradcam_url, "analyzed_disease": top_disease},
            "yolo":           {"image_url": yolo_url, "detections": detections, "total": len(detections)},
            "segmentation":   {"image_url": seg_url},
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ predict/dicom: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Dashboard stats ───────────────────────────────────────────────────────────

@app.get("/dashboard/stats")
async def dashboard_stats(
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    from sqlalchemy import func, case

    is_admin = current_user.role in ['admin', 'radiologist']

    # ── Filtro base por usuario ──────────────────────────────────────────
    study_filter = [] if is_admin else [Study.uploaded_by == current_user.id]

    # ── KPIs principales ─────────────────────────────────────────────────
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

    # ── Distribución por enfermedad ───────────────────────────────────────
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

    # ── Timeline últimos 30 días ──────────────────────────────────────────
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
            "yolo_map50":            "39.3%",
            "unet_dice":             "96.24%",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)