"""Validación de los archivos subidos antes de analizarlos."""

from io import BytesIO
from pathlib import Path

from fastapi import HTTPException
from PIL import Image

MAX_FILE_SIZE_MB    = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_MIME_TYPES  = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}
ALLOWED_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".dcm"}


def validate_image_file(contents: bytes, filename: str, allow_dicom: bool = False):
    """Valida tamaño, extensión y firma del archivo; lanza 400 si algo falla."""
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Archivo demasiado grande. Máximo {MAX_FILE_SIZE_MB} MB."
        )

    if len(contents) < 8:
        raise HTTPException(status_code=400, detail="Archivo demasiado pequeño o vacío.")

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Extensión '{ext}' no permitida. Permitidas: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # pil no abre dicom; ese formato se valida al convertirlo
    if ext == ".dcm":
        if not allow_dicom:
            raise HTTPException(status_code=400, detail="DICOM no permitido en este endpoint.")
        return

    # la extensión no basta: se comprueba la firma real del archivo
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

    try:
        img = Image.open(BytesIO(contents))
        img.verify()   # detecta archivos truncados/corruptos
    except Exception:
        raise HTTPException(status_code=400, detail="El archivo de imagen está corrupto o es inválido.")
