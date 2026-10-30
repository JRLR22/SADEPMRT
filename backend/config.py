from pydantic_settings import BaseSettings
from pathlib import Path

# claves de ejemplo del repo (config, Dockerfile, .env.example) o vacía si
# docker-compose recibe SECRET_KEY sin valor; con ellas cualquiera puede firmar tokens
PLACEHOLDER_SECRET_KEYS = {
    "",
    "cambia-esta-clave-en-produccion",
    "change-me-in-production",
    "CAMBIA-ESTO-POR-UNA-CLAVE-SEGURA",
    "CHANGE-THIS-IN-PRODUCTION",
}


class Settings(BaseSettings):

    APP_NAME:    str = "Lung Disease Detection API"
    APP_VERSION: str = "2.0.0"
    DEBUG:       bool = True
    ENVIRONMENT: str = "development"
    APP_ENV:     str = "development"   # el dockerfile y docker-compose lo ponen en "production"

    HOST:    str = "0.0.0.0"
    PORT:    int = 8000
    WORKERS: int = 1

    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:19006",
        "http://localhost:8081",
    ]

    DATABASE_URL: str = "sqlite:///./lung_detection.db"

    MODELS_DIR: Path = Path(__file__).parent / "models"

    # clasificación: 3 modelos por enfermedad
    PNEUMONIA_DENSENET_PATH:   Path = MODELS_DIR / "pneumonia_densenet121.pth"
    PNEUMONIA_RESNET_PATH:     Path = MODELS_DIR / "pneumonia_resnet50.pth"
    PNEUMONIA_EFFICIENT_PATH:  Path = MODELS_DIR / "pneumonia_efficientnet_b0.pth"

    COVID_DENSENET_PATH:   Path = MODELS_DIR / "covid_19_densenet121.pth"
    COVID_RESNET_PATH:     Path = MODELS_DIR / "covid_19_resnet50.pth"
    COVID_EFFICIENT_PATH:  Path = MODELS_DIR / "covid_19_efficientnet_b0.pth"

    TB_DENSENET_PATH:          Path = MODELS_DIR / "tuberculosis_densenet121.pth"
    TB_RESNET_PATH:            Path = MODELS_DIR / "tuberculosis_resnet50.pth"
    TB_EFFICIENT_PATH:         Path = MODELS_DIR / "tuberculosis_efficientnet_b0.pth"

    YOLO_MODEL_PATH: Path = MODELS_DIR / "yolov8_chest_best.pt"

    UNET_MODEL_PATH: Path = MODELS_DIR / "unet_resnet34_best.pth"

    DEVICE: str = "cpu"   # "cuda" si hay gpu nvidia
    MODEL_IMAGE_SIZE: int = 224

    PNEUMONIA_THRESHOLD: float = 0.5
    COVID_THRESHOLD:     float = 0.5
    TB_THRESHOLD:        float = 0.75

    SECRET_KEY:                  str = "cambia-esta-clave-en-produccion"
    ALGORITHM:                   str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    MAX_FILE_SIZE:       int = 10 * 1024 * 1024
    ALLOWED_EXTENSIONS:  set = {".jpg", ".jpeg", ".png"}

    class Config:
        env_file      = ".env"
        case_sensitive = True


settings = Settings()


def validate_config():
    """Valida la clave de firma y que los modelos existan antes de arrancar."""
    if settings.SECRET_KEY in PLACEHOLDER_SECRET_KEYS:
        if settings.APP_ENV == "production":
            raise RuntimeError(
                "\nSECRET_KEY no definido: en producción no se arranca con la clave de ejemplo.\n"
                "Pásalo como variable de entorno (por ejemplo en el .env de docker-compose).\n"
                'Genera uno con: python -c "import secrets; print(secrets.token_urlsafe(48))"\n'
            )
        print("AVISO: SECRET_KEY no definido; se usa una clave de ejemplo que solo vale para desarrollo. "
              "Defínelo en backend/.env.")

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
            "\nMODELOS NO ENCONTRADOS:\n" +
            "\n".join(f"  - {m}" for m in missing) +
            f"\n\nCopia los .pth y .pt a: {settings.MODELS_DIR}\n"
        )

    print("Configuración validada")