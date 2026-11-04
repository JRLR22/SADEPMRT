import numpy as np
import pytest

from conftest import dicom_bytes
from dicom_utils import dicom_to_pil, format_dicom_date


def gray(dcm):
    image, meta = dicom_to_pil(dcm)
    return np.asarray(image.convert("L"), dtype=int), meta


def test_metadatos_y_normalizacion():
    arr, meta = gray(dicom_bytes())
    assert meta["patient_id"] == "P-001" and meta["modality"] == "CR"
    assert arr.min() == 0 and arr.max() == 255
    assert arr[0, 0] < arr[0, -1]


def test_monochrome1_se_invierte():
    normal, _ = gray(dicom_bytes())
    inverted, _ = gray(dicom_bytes("MONOCHROME1"))
    assert np.array_equal(inverted, 255 - normal)


def test_ventana_recorta_los_extremos():
    arr, _ = gray(dicom_bytes(window=(2000, 1000)))
    # fuera de [1500, 2500] queda saturado: columnas 0-23 a negro, 40-63 a blanco
    assert (arr[:, :20] == 0).all() and (arr[:, -20:] == 255).all()


def test_archivo_que_no_es_dicom():
    with pytest.raises(ValueError):
        dicom_to_pil(b"esto no es un dicom")


def test_formato_de_fecha():
    assert format_dicom_date("20240315") == "15/03/2024"
    assert format_dicom_date("") == ""
