"""Entorno aislado para las pruebas: base de datos, carpeta de trabajo y modelos propios."""

import io
import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
SAMPLES = BACKEND.parent / "frontend-react2" / "public" / "samples"

# antes de importar nada del backend: config y base.py leen el entorno al importarse,
# y main.py crea uploads/ relativo a la carpeta actual
WORK = Path(tempfile.mkdtemp(prefix="rt26-tests-"))
os.chdir(WORK)
os.environ.update({
    "DATABASE_URL": f"sqlite:///{(WORK / 'test.db').as_posix()}",
    "SECRET_KEY": "clave-solo-para-las-pruebas-0123456789",
    "APP_ENV": "development",
    "ACCESS_TOKEN_EXPIRE_MINUTES": "60",
})
sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torchvision.models as tv_models  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

import main  # noqa: E402
from app.core import rate_limit  # noqa: E402
from app.services import inference, records  # noqa: E402
from app.core.security import get_password_hash  # noqa: E402
from app.database.base import Base, SessionLocal, engine  # noqa: E402
from app.models.user import User, UserRole  # noqa: E402
from app.services.storage import StorageService  # noqa: E402

PASSWORD = "contrasena-de-prueba"


def build_random_models():
    """Mismas arquitecturas que en producción, con pesos aleatorios y fijos."""
    from efficientnet_pytorch import EfficientNet
    import segmentation_models_pytorch as smp

    torch.manual_seed(0)
    # una instancia por arquitectura, compartida por las tres enfermedades, para ahorrar memoria
    trio = {
        "densenet": tv_models.densenet121(weights=None, num_classes=2).eval(),
        "resnet": tv_models.resnet50(weights=None, num_classes=2).eval(),
        "efficient": EfficientNet.from_name("efficientnet-b0", num_classes=2).eval(),
    }
    unet = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1, activation=None).eval()
    return {d: trio for d in ("pneumonia", "covid", "tuberculosis")}, unet


RANDOM_ENSEMBLE, RANDOM_UNET = build_random_models()


def install_random_models():
    inference.ensemble_models.clear()
    inference.ensemble_models.update(RANDOM_ENSEMBLE)
    inference.unet_model = RANDOM_UNET
    inference.yolo_model = None


@pytest.fixture(scope="session")
def client():
    Base.metadata.create_all(bind=engine)
    records.storage_service = StorageService()
    install_random_models()
    # sin "with": no se ejecuta el lifespan, que cargaría los modelos reales
    return TestClient(main.app)


@pytest.fixture(autouse=True)
def clean_state():
    rate_limit._rate_limiter._store.clear()
    yolo = inference.yolo_model
    yield
    inference.yolo_model = yolo


_counter = iter(range(1, 10**6))


@pytest.fixture
def make_user(client):
    """Crea un usuario con el rol pedido y devuelve sus cabeceras de autenticación."""
    def factory(role: str = "viewer"):
        email = f"{role}{next(_counter)}@pruebas.com"
        with SessionLocal() as db:
            db.add(User(email=email, hashed_password=get_password_hash(PASSWORD),
                        full_name=f"Usuario {role}", role=UserRole(role), is_active=True))
            db.commit()
        r = client.post("/auth/login", json={"email": email, "password": PASSWORD})
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['access_token']}"}
    return factory


@pytest.fixture
def xray_bytes():
    return (SAMPLES / "ejemplo-2.jpg").read_bytes()


def png_bytes(size=(64, 64), value=128):
    buf = io.BytesIO()
    Image.fromarray(np.full(size, value, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def dicom_bytes(photometric="MONOCHROME2", window=None):
    """DICOM sintético de 64x64 con un degradado, para no depender de archivos reales."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset("sintetico", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.PatientName, ds.PatientID, ds.Modality, ds.StudyDate = "Prueba^Sintetica", "P-001", "CR", "20240315"
    ds.Rows = ds.Columns = 64
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, photometric
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit, ds.PixelRepresentation = 15, 0
    if window:
        ds.WindowCenter, ds.WindowWidth = window
    ds.PixelData = np.tile(np.arange(64, dtype=np.uint16) * 64, (64, 1)).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf)
    return buf.getvalue()


class FakeYOLO:
    """Imita la interfaz de ultralytics que usa run_yolo."""
    names = {0: "Nodule/Mass", 3: "Cardiomegaly"}

    class _Box:
        def __init__(self, cls, conf, xyxy):
            self.cls, self.conf, self.xyxy = torch.tensor(cls), torch.tensor(conf), torch.tensor([xyxy])

    class _Result:
        def __init__(self, img):
            self.img = img
            self.boxes = [FakeYOLO._Box(3, 0.81, [10.0, 20.0, 200.0, 220.0]),
                          FakeYOLO._Box(0, 0.42, [50.0, 60.0, 80.0, 90.0])]

        def plot(self):
            return self.img

    def __call__(self, img, verbose=False):
        return [FakeYOLO._Result(img)]
