"""Descarga de Hugging Face Hub los modelos que falten en backend/models."""

import argparse
import hashlib
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS_DIR = Path(__file__).parent / "models"
REPO_ID = "JRLR22/rt26-models"

# commit fijo y sha256 por archivo: los .pth se abren con pickle, que puede ejecutar
# código, así que solo se acepta exactamente lo que se revisó al publicar
REVISION = "c0e51eb575905d69b7d737e15fbc41a05b5febf1"
MODELS = {
    "pneumonia_densenet121.pth": "fc89cb2bc9f0e624630901f8a5ad4973789a64c4a89fcc0cb26e51f5d13f0ca5",
    "pneumonia_resnet50.pth": "eeaaa8099317dedcf3802e83395de32720be8c50515e02d84674bcce34634cde",
    "pneumonia_efficientnet_b0.pth": "dab40ed148637e85e365db52b92d56b9f6e5800a244950e7e4b18e0ecc888abb",
    "covid_19_densenet121.pth": "5468eab24cff58a1e1212b3d2cf090b7c4a583f98c5906f9d02eeace8dbf6b21",
    "covid_19_resnet50.pth": "d5059ed2354243192d3137cbecc50a9609693ac98dd89518f23ffb752431d591",
    "covid_19_efficientnet_b0.pth": "a3872f2c010a3745f402083cd37e0a4b9d63480f2a7910ea3ec597aa3d97015a",
    "tuberculosis_densenet121.pth": "e3f9671002c8a7293f788a5f3ce207364bee56788ff2d381fcff631c1b0d8ed8",
    "tuberculosis_resnet50.pth": "e362e6fd889a16405b53d4c35512129865f11ec97a11f2a3825bca228c3170d9",
    "tuberculosis_efficientnet_b0.pth": "7c72a68035ad8499ec2448737fb45f09dd8f71b60f84635f294239dd2d2db1b8",
    "unet_resnet34_best.pth": "f53c41dd62b26849a8010a949eec50e827cc5c577e787040b5cdd7afef176ab4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="comprueba también el sha256 de los archivos que ya existen")
    args = parser.parse_args()

    errors = []
    for name, expected in MODELS.items():
        path = MODELS_DIR / name
        downloaded = False

        if path.exists():
            if not args.verify:
                print(f"ya existe: {name}")
                continue
        else:
            print(f"descargando: {name}")
            try:
                hf_hub_download(repo_id=REPO_ID, filename=name, revision=REVISION, local_dir=MODELS_DIR)
            except Exception as e:
                errors.append(f"{name}: no se pudo descargar ({e})")
                continue
            downloaded = True

        if sha256(path) == expected:
            print(f"verificado: {name}")
            continue

        # uno ya existente puede ser un modelo propio: se avisa, pero no se borra
        if downloaded:
            path.unlink()
        errors.append(f"{name}: el sha256 no coincide")

    if errors:
        print("\nERROR:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1

    print("\nModelos listos.")
    # los datos de entrenamiento de yolo (vinbigdata) son solo para uso no comercial
    if not (MODELS_DIR / "yolov8_chest_best.pt").exists():
        print("YOLO no se distribuye; la detección queda desactivada (ver README).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
