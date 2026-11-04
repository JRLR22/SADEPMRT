"""Carga de los modelos e inferencia: ensemble, Grad-CAM, YOLO y U-Net."""

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image

from config import settings
from dicom_utils import dicom_to_pil as _dicom_to_pil_util
from gradcam import create_gradcam_for_densenet

# los carga load_models() al arrancar la api
ensemble_models: dict = {}   # {disease: {densenet, resnet, efficient}}
yolo_model              = None
unet_model              = None


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


def load_models(device):
    """Carga los tres ensembles, la U-Net y, si está, YOLO."""
    global yolo_model, unet_model

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
        logger.info(f"   Cargando ensemble {disease}...")
        ensemble_models[disease] = {
            'densenet':  _load_densenet(cfg['densenet'],  nc, device),
            'resnet':    _load_resnet(cfg['resnet'],    nc, device),
            'efficient': _load_efficientnet(cfg['efficient'], nc, device),
        }

    if settings.YOLO_MODEL_PATH.exists():
        logger.info("   Cargando YOLOv8...")
        from ultralytics import YOLO
        yolo_model = YOLO(str(settings.YOLO_MODEL_PATH))
    else:
        logger.warning("   YOLOv8 no encontrado: la detección queda desactivada")

    logger.info("   Cargando U-Net...")
    unet_model = _load_unet(settings.UNET_MODEL_PATH, device)


# las radiografías son en gris, pero los clasificadores parten de imagenet:
# esperan 3 canales y su normalización
TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# la u-net se entrenó con 1 canal a 512x512
UNET_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])


def preprocess_image(image_bytes: bytes):
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    tensor = TRANSFORM(image).unsqueeze(0)
    return tensor, image


def ensemble_predict(disease: str, tensor: torch.Tensor) -> dict:
    """Soft voting: promedia las probabilidades de los 3 modelos."""
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
    gradcam = None
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
    finally:
        if gradcam is not None:
            gradcam.remove_hooks()


def run_yolo(image_bytes: bytes) -> tuple[bytes, list]:
    """Devuelve (png anotado, lista de detecciones)."""
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
    """Devuelve la máscara pulmonar superpuesta en verde, como png."""
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
    """Convierte DICOM a imagen PIL y extrae sus metadatos."""
    return _dicom_to_pil_util(dicom_bytes)


def get_clinical_interpretation(confidence: float, disease: str) -> dict:
    # tb exige más confianza, igual que su umbral de decisión (TB_THRESHOLD = 0.75)
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
    # valores fijos; no se recalculan en ejecución
    return {
        'pneumonia':    {'accuracy': '92.75%', 'auc': '0.9720', 'f1': '0.93'},
        'covid':        {'accuracy': '98.50%', 'auc': '0.9910', 'f1': '0.98'},
        'tuberculosis': {'accuracy': '97.80%', 'auc': '0.9850', 'f1': '0.97'},
    }.get(disease, {})
