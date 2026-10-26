# ============================================================================
# CONFIG.PY - CONFIGURACIÓN LOCAL (SIN SUPABASE)
# ============================================================================

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):

    # General
    APP_NAME:    str = "Lung Disease Detection API"
    APP_VERSION: str = "2.0.0"
    DEBUG:       bool = True
    ENVIRONMENT: str = "development"

    # Servidor
    HOST:    str = "0.0.0.0"
    PORT:    int = 8000
    WORKERS: int = 1

    # CORS
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:19006",
        "http://localhost:8081",
    ]

    # Base de datos — SQLite local
    DATABASE_URL: str = "sqlite:///./lung_detection.db"

    # Modelos de IA
    MODELS_DIR: Path = Path(__file__).parent / "models"

    # ── Clasificación (ensemble: 3 modelos por enfermedad) ──────────────────
    # Pneumonia
    PNEUMONIA_DENSENET_PATH:   Path = MODELS_DIR / "pneumonia_densenet121.pth"
    PNEUMONIA_RESNET_PATH:     Path = MODELS_DIR / "pneumonia_resnet50.pth"
    PNEUMONIA_EFFICIENT_PATH:  Path = MODELS_DIR / "pneumonia_efficientnet_b0.pth"

    # COVID-19
    COVID_DENSENET_PATH:   Path = MODELS_DIR / "covid_19_densenet121.pth"
    COVID_RESNET_PATH:     Path = MODELS_DIR / "covid_19_resnet50.pth"
    COVID_EFFICIENT_PATH:  Path = MODELS_DIR / "covid_19_efficientnet_b0.pth"

    # Tuberculosis
    TB_DENSENET_PATH:          Path = MODELS_DIR / "tuberculosis_densenet121.pth"
    TB_RESNET_PATH:            Path = MODELS_DIR / "tuberculosis_resnet50.pth"
    TB_EFFICIENT_PATH:         Path = MODELS_DIR / "tuberculosis_efficientnet_b0.pth"

    # ── Detección YOLO ───────────────────────────────────────────────────────
    YOLO_MODEL_PATH: Path = MODELS_DIR / "yolov8_chest_best.pt"

    # ── Segmentación U-Net ───────────────────────────────────────────────────
    UNET_MODEL_PATH: Path = MODELS_DIR / "unet_resnet34_best.pth"

    # Device
    DEVICE: str = "cpu"   # Cambiar a "cuda" si hay GPU NVIDIA
    MODEL_IMAGE_SIZE: int = 224

    # Thresholds
    PNEUMONIA_THRESHOLD: float = 0.5
    COVID_THRESHOLD:     float = 0.5
    TB_THRESHOLD:        float = 0.75

    # Seguridad
    SECRET_KEY:                  str = "cambia-esta-clave-en-produccion"
    ALGORITHM:                   str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Archivos
    MAX_FILE_SIZE:       int = 10 * 1024 * 1024   # 10 MB
    ALLOWED_EXTENSIONS:  set = {".jpg", ".jpeg", ".png"}

    class Config:
        env_file      = ".env"
        case_sensitive = True


settings = Settings()


def validate_config():
    """Valida que los modelos existan antes de arrancar."""
    required = {
        "Pneumonia DenseNet":    settings.PNEUMONIA_DENSENET_PATH,
        "Pneumonia ResNet":      settings.PNEUMONIA_RESNET_PATH,
        "Pneumonia EfficientNet":settings.PNEUMONIA_EFFICIENT_PATH,
        "COVID DenseNet":        settings.COVID_DENSENET_PATH,
        "COVID ResNet":          settings.COVID_RESNET_PATH,
        "COVID EfficientNet":    settings.COVID_EFFICIENT_PATH,
        "TB DenseNet":           settings.TB_DENSENET_PATH,
        "TB ResNet":             settings.TB_RESNET_PATH,
        "TB EfficientNet":       settings.TB_EFFICIENT_PATH,
        "YOLOv8":                settings.YOLO_MODEL_PATH,
        "U-Net":                 settings.UNET_MODEL_PATH,
    }

    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "\n❌ MODELOS NO ENCONTRADOS:\n" +
            "\n".join(f"  - {m}" for m in missing) +
            f"\n\n💡 Copia los .pth y .pt a: {settings.MODELS_DIR}\n"
        )

    if settings.SECRET_KEY == "cambia-esta-clave-en-produccion":
        print("AVISO: se usa el SECRET_KEY por defecto. Define uno propio en .env antes de publicar.")

    print("✅ Configuración validada")