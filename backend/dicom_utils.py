import io
import pydicom
import numpy as np
from PIL import Image
from pydicom.errors import InvalidDicomError


def dicom_to_pil(dicom_bytes: bytes) -> tuple[Image.Image, dict]:
    """
    Convierte bytes DICOM a imagen PIL normalizada + metadatos del estudio.
    Maneja DICOM con y sin pixel data comprimido.
    """
    try:
        dcm = pydicom.dcmread(io.BytesIO(dicom_bytes))
    except InvalidDicomError:
        raise ValueError("El archivo no es un DICOM válido.")

    # --- Extraer metadatos ---
    metadata = {
        "patient_name":    str(getattr(dcm, "PatientName",    "Desconocido")),
        "patient_id":      str(getattr(dcm, "PatientID",      "")),
        "patient_age":     str(getattr(dcm, "PatientAge",     "")),
        "patient_sex":     str(getattr(dcm, "PatientSex",     "")),
        "study_date":      str(getattr(dcm, "StudyDate",      "")),
        "study_time":      str(getattr(dcm, "StudyTime",      "")),
        "modality":        str(getattr(dcm, "Modality",       "CR")),
        "institution":     str(getattr(dcm, "InstitutionName","Sin institución")),
        "manufacturer":    str(getattr(dcm, "Manufacturer",   "")),
        "study_description": str(getattr(dcm, "StudyDescription", "")),
        "series_description": str(getattr(dcm, "SeriesDescription", "")),
        "body_part":       str(getattr(dcm, "BodyPartExamined", "")),
        "kvp":             str(getattr(dcm, "KVP", "")),
        "exposure":        str(getattr(dcm, "Exposure", "")),
    }

    # --- Convertir pixel array ---
    try:
        img_array = dcm.pixel_array
    except Exception as e:
        raise ValueError(f"No se pudo leer pixel array del DICOM: {e}")

    img_array = img_array.astype(np.float32)

    # Aplicar VOI LUT si existe (Window Center/Width)
    if hasattr(dcm, "WindowCenter") and hasattr(dcm, "WindowWidth"):
        wc = float(dcm.WindowCenter) if not isinstance(dcm.WindowCenter, pydicom.multival.MultiValue) \
             else float(dcm.WindowCenter[0])
        ww = float(dcm.WindowWidth) if not isinstance(dcm.WindowWidth, pydicom.multival.MultiValue) \
             else float(dcm.WindowWidth[0])
        img_min = wc - ww / 2
        img_max = wc + ww / 2
        img_array = np.clip(img_array, img_min, img_max)

    # Normalizar a 0-255
    arr_min, arr_max = img_array.min(), img_array.max()
    if arr_max > arr_min:
        img_array = (img_array - arr_min) / (arr_max - arr_min) * 255.0
    img_array = img_array.astype(np.uint8)

    # Aplicar PhotometricInterpretation si es MONOCHROME1 (invertir)
    photometric = str(getattr(dcm, "PhotometricInterpretation", "MONOCHROME2"))
    if photometric == "MONOCHROME1":
        img_array = 255 - img_array

    # Convertir a PIL RGB
    if len(img_array.shape) == 2:
        pil_image = Image.fromarray(img_array, mode="L").convert("RGB")
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
        pil_image = Image.fromarray(img_array, mode="RGB")
    else:
        pil_image = Image.fromarray(img_array).convert("RGB")

    return pil_image, metadata


def format_dicom_date(date_str: str) -> str:
    """Convierte '20240315' a '15/03/2024'"""
    if len(date_str) == 8:
        return f"{date_str[6:8]}/{date_str[4:6]}/{date_str[0:4]}"
    return date_str