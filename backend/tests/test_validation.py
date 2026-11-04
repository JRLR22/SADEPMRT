import pytest
from fastapi import HTTPException

from conftest import png_bytes
from app.core.validation import MAX_FILE_SIZE_BYTES, validate_image_file


def rejects(contents, filename, **kw):
    with pytest.raises(HTTPException) as e:
        validate_image_file(contents, filename, **kw)
    assert e.value.status_code == 400
    return e.value.detail


def test_imagen_valida():
    validate_image_file(png_bytes(), "rx.png")


def test_archivo_demasiado_grande():
    assert "demasiado grande" in rejects(b"\x89PNG\r\n\x1a\n" + b"0" * MAX_FILE_SIZE_BYTES, "rx.png")


def test_archivo_vacio():
    assert "vacío" in rejects(b"abc", "rx.png")


def test_extension_no_permitida():
    assert "no permitida" in rejects(png_bytes(), "rx.gif")


def test_dicom_solo_donde_se_permite():
    assert "DICOM no permitido" in rejects(b"0" * 200, "rx.dcm")
    validate_image_file(b"0" * 200, "rx.dcm", allow_dicom=True)


def test_extension_falsa():
    # un archivo de texto renombrado a .png no pasa por la firma
    assert "firma no reconocida" in rejects(b"esto no es una imagen", "rx.png")


def test_imagen_truncada():
    assert "corrupto" in rejects(png_bytes()[:60], "rx.png")
